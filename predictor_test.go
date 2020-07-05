package onnxruntime

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/GeertJohan/go-sourcepath"
	"github.com/rai-project/config"
	"github.com/rai-project/dlframework/framework/options"
	nvidiasmi "github.com/rai-project/nvidia-smi"
	_ "github.com/rai-project/tracer/all"
	gotensor "gorgonia.org/tensor"
)

var (
	batchSize     = 2
	shape         = []int{1, 3, 224, 224}
	thisDir       = sourcepath.MustAbsoluteDir()
	onnxModelPath = filepath.Join(thisDir, "examples", "_fixtures", "torchvision_alexnet", "torchvision_alexnet.onnx")
)

func TestOnnxruntimePredictor(t *testing.T) {

	var input []float32
	size := 1
	for _, sz := range shape {
		size *= sz
	}

	for i := 0; i < batchSize; i++ {
		for j := 0; j < size; j++ {
			input = append(input, float32(i))
		}
	}

	device := options.CPU_DEVICE
	if nvidiasmi.HasGPU {
		device = options.CUDA_DEVICE
	}

	ctx := context.Background()

	opts := options.New(options.Context(ctx),
		options.Graph([]byte(onnxModelPath)),
		options.Device(device, 0),
		options.BatchSize(batchSize))

	predictor, err := New(
		ctx,
		options.WithOptions(opts),
	)

	if err != nil {
		t.Errorf("Onnxruntime predictor initialization failed %v", err)
	}

	defer predictor.Close()

	dims := shape
	dims[0] = batchSize

	err = predictor.Predict(ctx, []gotensor.Tensor{
		gotensor.New(
			gotensor.Of(gotensor.Float32),
			gotensor.WithBacking(input),
			gotensor.WithShape(dims...),
		),
	})

	if err != nil {
		t.Errorf("Onnxruntime predictor predicting failed %v", err)
	}
}

func TestMain(m *testing.M) {
	config.Init(
		config.AppName("carml"),
		config.VerboseMode(true),
		config.DebugMode(true),
	)

	os.Exit(m.Run())
}
