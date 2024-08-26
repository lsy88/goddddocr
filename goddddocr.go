package goddddocr

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"github.com/disintegration/imaging"
	ort "github.com/yalue/onnxruntime_go"
	"gonum.org/v1/gonum/mat"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
)

type GoDDDDOcr struct {
	Ocr            bool
	Det            bool
	Old            bool
	Beta           bool
	UseGPU         bool
	DeviceId       int
	ImportOnnxPath string
	charsetsPath   string
	_word          bool
	_resize        []int
	_charsetRange  []int
	_channel       int
	_charset       []string
	_graphPath     string
	_useImportOnnx bool
	_ortSession    *ort.DynamicAdvancedSession
}

// NewGOcr 暂时只支持识别
func NewGOcr() (*GoDDDDOcr, error) {
	return NewGoDDDDOcr(true, false, false, true, false, -1, "", "")
}

func NewGoDDDDOcr(ocr, det, old, beta, useGpu bool, deviceId int, importOnnxPath, charsetsPath string) (*GoDDDDOcr, error) {
	dd := &GoDDDDOcr{}
	dd._useImportOnnx = false
	dd._word = false
	dd._resize = make([]int, 0)
	dd._charsetRange = make([]int, 0)
	dd._channel = 1

	if importOnnxPath != "" {
		det = false
		ocr = false
		dd._graphPath = importOnnxPath

		f, err := os.Open(charsetsPath)
		defer f.Close()
		if err != nil {
			return dd, err
		}
		c, err := io.ReadAll(f)
		if err != nil {
			return dd, err
		}
		var info map[string]interface{}

		err = json.Unmarshal(c, &info)
		if err != nil {
			return dd, err
		}

		dd._charset = info["charset"].([]string)
		dd._word = info["word"].(bool)
		dd._resize = info["image"].([]int)
		dd._channel = info["channel"].(int)
		dd._useImportOnnx = true
	}

	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return dd, errors.New("get path failed")
	}

	if det {
		ocr = false
		dd._graphPath = filepath.Join(filepath.Dir(filename), "model", "common_det.onnx")
		dd._charset = nil
	}

	if ocr {
		if !beta {
			dd._graphPath = filepath.Join(filepath.Dir(filename), "model", "common_old.onnx")
			dd._charset = CommonOldCharSet
		} else {
			dd._graphPath = filepath.Join(filepath.Dir(filename), "model", "common.onnx")
			dd._charset = CommonCharSet
		}
	}
	dd.Det = det

	if ocr || det || dd._useImportOnnx {
		// 确定输入输出name
		var inputName, outputName []string
		inputs, outputs, err := ort.GetInputOutputInfo(dd._graphPath)
		for _, input := range inputs {
			inputName = append(inputName, input.Name)
		}
		for _, output := range outputs {
			outputName = append(outputName, output.Name)
		}
		session, err := newSession(dd._graphPath, inputName, outputName)
		if err != nil {
			return dd, nil
		}
		dd._ortSession = session
	}

	return dd, nil
}

// 将img转为灰度矩阵
func imageToMatrix(img image.Image) *mat.Dense {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	matrix := mat.NewDense(height, width, nil)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			gray := color.RGBA{uint8(r >> 8), uint8(g >> 8), uint8(b >> 8), 255}
			matrix.Set(y, x, float64(gray.R)/255.0)
		}
	}

	return matrix
}

// 将矩阵转为一维张量作为输入数据
func flatteningFloat32(matrix *mat.Dense, width int) []float32 {
	tensorData := make([]float32, 64*width)
	for i, v := range matrix.RawMatrix().Data {
		tensorData[i] = float32(v)
	}
	return tensorData
}

func removeBase64Prefix(imgData string) string {
	prefixes := []string{
		"data:image/png;base64,",
		"data:image/jpeg;base64,",
		// ...
	}

	for _, prefix := range prefixes {
		imgData = strings.TrimPrefix(imgData, prefix)
	}

	return imgData
}

func (d *GoDDDDOcr) ProbePath(path string) (ocr string, err error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	imgData, err := io.ReadAll(f)
	if err != nil {
		return "", err
	}
	return d.Probe(string(imgData))
}

// Probe 识别ocr
func (d *GoDDDDOcr) Probe(imgData string) (ocr string, err error) {
	imgByte, err := base64.StdEncoding.DecodeString(removeBase64Prefix(imgData))
	if err != nil {
		imgByte = []byte(imgData)
	}

	img, _, err := image.Decode(bytes.NewReader(imgByte))
	if err != nil {
		return "", err
	}
	// 等比例缩小图片为高度64像素
	width := int(float64(img.Bounds().Dx()) * (float64(64) / float64(img.Bounds().Dy())))
	resizedImage := imaging.Resize(img, width, 64, imaging.Lanczos)

	// 处理img为灰度矩阵
	matrix := imageToMatrix(resizedImage)

	inputData := flatteningFloat32(matrix, width)
	// 构造输入-四维张量
	// 1: 第一个维度的大小是 1，通常用于表示批量大小（batch size）。在深度学习中，批量大小通常用于一次处理多个样本，这里 1 表示每次处理一个样本。
	// 1: 第二个维度的大小是 1，通常用于表示通道数。在图像处理任务中，这个维度可以表示图像的颜色通道（例如，1 代表灰度图像，3 代表 RGB 图像）。
	// 64: 第三个维度的大小是 64，这个维度可以表示图像的高度（或特征图的高度）。
	// int64(width): 第四个维度的大小是图像的宽度（width），用于表示图像的宽度或特征图的宽度
	inputShape := ort.NewShape(1, 1, 64, int64(width))
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return "", err
	}
	defer inputTensor.Destroy()

	// shape为暂时指定,失败后会植入正确shape
	var shape = []int64{1, 1}

retry:
	// 构造输出张量
	outputShape := ort.NewShape(shape...)
	outputTensor, err := ort.NewEmptyTensor[int64](outputShape)
	if err != nil {
		return "", err
	}
	defer outputTensor.Destroy()

	err = d._ortSession.Run([]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor})
	if err != nil {
		// 处理已知的错误并重新计算
		if strings.Contains(err.Error(), "OrtValue shape verification failed. Current shape:") {
			regexPattern := `Requested shape:{(\d+),(\d+)}`
			regex := regexp.MustCompile(regexPattern)
			matches := regex.FindStringSubmatch(err.Error())
			num1, _ := strconv.Atoi(matches[1])
			num2, _ := strconv.Atoi(matches[2])
			shape = []int64{int64(num1), int64(num2)}
			goto retry
		}
		log.Fatalln("calculate result error", err)
	}
	defer d._ortSession.Destroy()

	var result []string
	lastItem := int64(0)
	for _, item := range outputTensor.GetData() {
		if int64(item) == lastItem {
			continue
		} else {
			lastItem = int64(item)
		}
		if item != 0 {
			result = append(result, d._charset[int64(item)])
		}
	}

	return strings.Join(result, ""), nil
}
