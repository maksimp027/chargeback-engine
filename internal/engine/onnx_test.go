package engine

import (
	"os"
	"testing"
)

func resolveTestPaths() (string, string) {
	dllPath := os.Getenv("ONNX_LIB_PATH")
	if dllPath == "" {
		if _, err := os.Stat("../../libonnxruntime.so"); err == nil {
			dllPath = "../../libonnxruntime.so"
		} else {
			dllPath = "/usr/lib/libonnxruntime.so"
		}
	}
	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		modelPath = "../../model_artifacts/chargeback_model.onnx"
	}
	return dllPath, modelPath
}

func TestONNXEngine(t *testing.T) {
	dllPath, modelPath := resolveTestPaths()

	if err := InitRuntime(dllPath); err != nil {
		t.Fatalf("Failed to init runtime: %v", err)
	}
	defer CleanupRuntime()

	engine, err := NewONNXEngine(modelPath)
	if err != nil {
		t.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()

	features := make([]float32, 19)
	score, err := engine.Predict(features)
	if err != nil {
		t.Fatalf("Prediction failed: %v", err)
	}

	if score < 0.0 || score > 1.0 {
		t.Fatalf("Score out of bounds [0, 1]: %f", score)
	}
}

func BenchmarkONNXEngine_Predict(b *testing.B) {
	dllPath, modelPath := resolveTestPaths()

	if err := InitRuntime(dllPath); err != nil {
		b.Fatalf("Failed to init runtime: %v", err)
	}
	defer CleanupRuntime()

	engine, err := NewONNXEngine(modelPath)
	if err != nil {
		b.Fatalf("Failed to create engine: %v", err)
	}
	defer engine.Close()

	features := make([]float32, 19)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := engine.Predict(features); err != nil {
			b.Fatalf("Prediction failed: %v", err)
		}
	}
}
