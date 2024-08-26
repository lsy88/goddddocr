package goddddocr

import (
	"fmt"
	ort "github.com/yalue/onnxruntime_go"
	"path/filepath"
	"runtime"
)

func init() {
	ort.SetSharedLibraryPath(getSharedLibPath())
	err := ort.InitializeEnvironment()
	if err != nil {
		panic(err)
	}
}

func newSession(modelPath string, inputName, outputName []string) (*ort.DynamicAdvancedSession, error) {
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("error creating ORT session options: %w", err)
	}
	defer options.Destroy()

	session, err := ort.NewDynamicAdvancedSession(modelPath, inputName, outputName, options)
	if err != nil {
		return nil, fmt.Errorf("error creating ORT session: %w", err)
	}
	return session, nil
}

// 获取链接库 https://github.com/microsoft/onnxruntime/releases
func getSharedLibPath() string {
	lib := filepath.Join(".", "lib_onnx")
	switch runtime.GOOS {
	case "windows":
		if runtime.GOARCH == "amd64" {
			return filepath.Join(lib, "onnxruntime.dll")
		}
	case "darwin":
		if runtime.GOARCH == "arm64" {
			return filepath.Join(lib, "onnxruntime_arm64.dylib")
		}
		if runtime.GOARCH == "amd64" {
			return filepath.Join(lib, "onnxruntime_amd64.dylib")
		}
	case "linux":
		if runtime.GOARCH == "arm64" {
			return filepath.Join(lib, "onnxruntime_arm64.so")
		}
		return filepath.Join(lib, "onnxruntime_amd64.so")
	}

	panic("get lib onnxruntime failed")
}
