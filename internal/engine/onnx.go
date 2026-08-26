package engine

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

type ONNXEngine struct {
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
	mu           sync.Mutex
}

func InitRuntime(sharedLibPath string) error {
	ort.SetSharedLibraryPath(sharedLibPath)
	return ort.InitializeEnvironment()
}

func CleanupRuntime() {
	ort.DestroyEnvironment()
}

func NewONNXEngine(modelPath string) (*ONNXEngine, error) {
	inputNames := []string{"float_input"}
	outputNames := []string{"probabilities"}

	inputShape := ort.NewShape(1, 19)
	outputShape := ort.NewShape(1, 2)

	// Tensors are pinned to the C heap for the lifetime of the session to avoid
	// per-request CGO allocations; must be explicitly destroyed on error or Close.
	inputData := make([]float32, 19)
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}

	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("failed to create output tensor: %w", err)
	}

	session, err := ort.NewAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		[]ort.Value{inputTensor},
		[]ort.Value{outputTensor},
		nil,
	)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("failed to create ONNX session: %w", err)
	}

	return &ONNXEngine{
		session:      session,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
	}, nil
}

func (e *ONNXEngine) Predict(features []float32) (float32, error) {
	if len(features) != 19 {
		return 0.0, fmt.Errorf("invalid feature length: expected 19, got %d", len(features))
	}

	// Serialize access to the shared in-place C tensor buffers across concurrent RPCs.
	e.mu.Lock()
	defer e.mu.Unlock()

	copy(e.inputTensor.GetData(), features)

	err := e.session.Run()
	if err != nil {
		return 0.0, fmt.Errorf("ONNX inference execution failed: %w", err)
	}

	results := e.outputTensor.GetData()
	if len(results) < 2 {
		return 0.0, fmt.Errorf("unexpected output shape from model")
	}

	// Output layout: [0] = P(legit), [1] = P(chargeback)
	return results[1], nil
}

func (e *ONNXEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	var err error
	if e.session != nil {
		err = e.session.Destroy()
	}
	if e.inputTensor != nil {
		e.inputTensor.Destroy()
	}
	if e.outputTensor != nil {
		e.outputTensor.Destroy()
	}
	return err
}
