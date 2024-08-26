package goddddocr

import (
	"fmt"
	"testing"
)

func TestNewGOcr(t *testing.T) {
	ocr, err := NewGOcr()
	if err != nil {
		fmt.Println(err.Error())
		return
	}
	base64ImageData := "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"
	ddddOcr, err := ocr.Probe(base64ImageData)
	if err != nil {
		fmt.Println(err.Error())
		return
	}
	fmt.Println("ocr = ", ddddOcr)
}
