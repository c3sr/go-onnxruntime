package onnxruntime

// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"
import (
	"context"
	"runtime"
	"strings"
	"time"
	"unsafe"

	"github.com/Unknwon/com"
	"github.com/c3sr/dlframework/framework/options"
	cupti "github.com/c3sr/go-cupti"
	nvidiasmi "github.com/c3sr/nvidia-smi"
	"github.com/c3sr/tracer"
	"github.com/k0kubun/pp"
	opentracing "github.com/opentracing/opentracing-go"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

type Predictor struct {
	ctx               C.ORT_PredictorContext
	options           *options.Options
	cu                *cupti.CUPTI
	startingTimeSlice []int64
	endingTimeSlice   []int64
	ctxSlice          []context.Context
	predictSpanSlice  []opentracing.Span
}

func New(ctx context.Context, opts ...options.Option) (*Predictor, error) {
	defer PanicOnError()

	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_new")
	defer span.Finish()

	options := options.New(opts...)
	modelFile := string(options.Graph())
	if !com.IsFile(modelFile) {
		return nil, errors.Errorf("file %s not found", modelFile)
	}

	device := fromDevice(options)
	if device == UnknownDeviceKind {
		return nil, errors.New("invalid device")
	}

	cModelFile := C.CString(modelFile)
	defer C.free(unsafe.Pointer(cModelFile))

	pred := &Predictor{
		ctx:     C.ORT_NewPredictor(cModelFile, C.ORT_DeviceKind(device), C.bool(options.TraceLevel() >= tracer.FRAMEWORK_TRACE)),
		options: options,
	}

	runtime.SetFinalizer(pred, func(p *Predictor) {
		p.Close()
	})

	return pred, GetError()
}

func fromDevice(opts *options.Options) DeviceKind {
	device := CPUDeviceKind
	if opts.UsesGPU() {
		if !nvidiasmi.HasGPU {
			return UnknownDeviceKind
		}
		device = CUDADeviceKind
	}
	return device
}

func (p *Predictor) addinput(ten *tensor.Dense) {
	shape := make([]int64, len(ten.Shape()))
	for i, s := range ten.Shape() {
		shape[i] = int64(s)
	}
	var shapePtr *C.int64_t
	shapePtr = (*C.int64_t)(unsafe.Pointer(&shape[0]))

	C.ORT_AddInput(p.ctx, ten.Pointer(), shapePtr, C.int(len(shape)), fromType(ten))

	runtime.KeepAlive(shape)
}

func (p *Predictor) Predict(ctx context.Context, inputs []tensor.Tensor) error {
	defer PanicOnError()
	if len(inputs) < 1 {
		return errors.New("input nil or empty")
	}

	C.ORT_PredictorClear(p.ctx)

	for _, input := range inputs {
		dense, ok := input.(*tensor.Dense)
		if !ok {
			return errors.New("expecting a dense tensor")
		}
		p.addinput(dense)
	}

	predictSpan, ctx := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict")

	if tracer.GetLevel() < tracer.FRAMEWORK_TRACE {
		defer predictSpan.Finish()
	}

	err := p.cuptiStart(ctx)
	if err != nil {
		return err
	}

	defer p.cuptiClose()

	if tracer.GetLevel() >= tracer.FRAMEWORK_TRACE {
		p.predictSpanSlice = append(p.predictSpanSlice, predictSpan)
		p.ctxSlice = append(p.ctxSlice, ctx)
		p.startingTimeSlice = append(p.startingTimeSlice, time.Now().UnixNano())
	}

	C.ORT_PredictorRun(p.ctx)

	if tracer.GetLevel() >= tracer.FRAMEWORK_TRACE {
		p.endingTimeSlice = append(p.endingTimeSlice, time.Now().UnixNano())
	}

	return GetError()
}

func (p *Predictor) ReadPredictionOutput(ctx context.Context) ([]tensor.Tensor, error) {
	defer PanicOnError()

	span, _ := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_read_predicted_output")
	defer span.Finish()

	C.ORT_PredictorConvertOutput(p.ctx)

	cNumOutputs := int(C.ORT_PredictorNumOutputs(p.ctx))

	if cNumOutputs == 0 {
		return nil, errors.New("zero number of tensors")
	}

	res := make([]tensor.Tensor, cNumOutputs)

	for i := 0; i < cNumOutputs; i++ {
		cPredictions := C.ORT_PredictorGetOutput(p.ctx, C.int(i))
		// The allocated memory will be deleted when destructor of predictor in c++ is called
		res[i] = ortValueToTensor(cPredictions)
	}

	if err := GetError(); err != nil {
		return nil, err
	}

	return res, nil
}

func (p *Predictor) Close() {
	if p == nil {
		return
	}

	if p.options.TraceLevel() >= tracer.FRAMEWORK_TRACE {
		C.ORT_EndProfiling(p.ctx)
		start_time := int64(C.ORT_ProfilingGetStartTime(p.ctx))

		profBuffer, err := p.ReadProfile()
		if err != nil {
			pp.Println(err)
			return
		}

		t, err := NewTrace(profBuffer, start_time)
		if err != nil {
			panic(err)
		}

		tSlice, err := SplitTrace(t, p.startingTimeSlice, p.endingTimeSlice)
		if err != nil {
			panic(err)
		}

		for batchNum, ctx := range p.ctxSlice {
			tSlice[batchNum].Publish(ctx, tracer.FRAMEWORK_TRACE)
			p.predictSpanSlice[batchNum].FinishWithOptions(opentracing.FinishOptions{
				FinishTime: time.Unix(0, p.endingTimeSlice[batchNum]),
			})
		}

		// clear records
		p.startingTimeSlice = nil
		p.endingTimeSlice = nil
		p.ctxSlice = nil
		p.predictSpanSlice = nil
	}

	if p.ctx != nil {
		C.ORT_PredictorDelete(p.ctx)
	}
	p.ctx = nil

}

func (p *Predictor) cuptiStart(ctx context.Context) error {
	if p.options.TraceLevel() < tracer.SYSTEM_LIBRARY_TRACE {
		return nil
	}
	metrics := []string{}
	if p.options.GPUMetrics() != "" {
		metrics = strings.Split(p.options.GPUMetrics(), ",")
	}

	cu, err := cupti.New(cupti.Context(ctx),
		cupti.SamplingPeriod(0),
		cupti.Metrics(metrics),
	)
	if err != nil {
		return err
	}

	p.cu = cu
	return nil
}

func (p *Predictor) cuptiClose() {
	if p.cu == nil {
		return
	}
	p.cu.Wait()
	p.cu.Close()
	p.cu = nil
}
