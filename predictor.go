package onnxruntime

// #include <stdlib.h>
// #include "cbits/predictor.hpp"
import "C"
import (
	"context"
	"runtime"
	"strings"
	"unsafe"

	"github.com/Unknwon/com"
	"github.com/k0kubun/pp"
	"github.com/pkg/errors"
	"github.com/rai-project/dlframework/framework/options"
	cupti "github.com/rai-project/go-cupti"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	"github.com/rai-project/tracer"
	"gorgonia.org/tensor"
)

type Predictor struct {
	ctx     C.ORT_PredictorContext
	options *options.Options
	cu      *cupti.CUPTI
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
		ctx:     C.ORT_NewPredictor(cModelFile, C.ORT_DeviceKind(device)),
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

	for _, input := range inputs {
		dense, ok := input.(*tensor.Dense)
		if !ok {
			return errors.New("expecting a dense tensor")
		}
		p.addinput(dense)
	}

	predictSpan, ctx := tracer.StartSpanFromContext(ctx, tracer.MODEL_TRACE, "c_predict")
	defer predictSpan.Finish()

	if p.options.TraceLevel() >= tracer.FRAMEWORK_TRACE {
		defer func() {
			pp.Println("Remember to resume parsing profiler.")
			// TODO: add trace.go
			// start_time := int64(C.ORT_ProfilingGetStartTime(p.ctx))

			// Note: The pointer from C is already freed in p.ReadProfile()
			// profBuffer, err := p.ReadProfile()
			// if err != nil {
			// 	pp.Println(err)
			// 	return
			// }
			// t, err := NewTrace(profBuffer, start_time)
			// if err != nil {
			// 	panic(err)
			// 	return
			// }
			// t.Publish(ctx, tracer.FRAMEWORK_TRACE)
		}()
	}

	err := p.cuptiStart(ctx)
	if err != nil {
		return err
	}
	defer p.cuptiClose()

	C.ORT_PredictorRun(p.ctx)

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
