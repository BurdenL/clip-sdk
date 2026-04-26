package clipsdk

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"

	"io"

	"golang.org/x/image/draw"
)

var (
	clipMean = [3]float32{0.48145466, 0.4578275, 0.40821073}
	clipStd  = [3]float32{0.26862954, 0.26130258, 0.27577711}
)

func preprocessImage(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("decode error: %w", err)
	}

	size := 224
	resized := image.NewRGBA(image.Rect(0, 0, size, size))
	draw.CatmullRom.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Src, nil)

	tensor := make([]float32, 3*size*size)

	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()

			idx := y*size + x
			tensor[idx] = (float32(r)/65535 - clipMean[0]) / clipStd[0]
			tensor[size*size+idx] = (float32(g)/65535 - clipMean[1]) / clipStd[1]
			tensor[2*size*size+idx] = (float32(b)/65535 - clipMean[2]) / clipStd[2]
		}
	}
	return tensor, nil
}

// PreprocessImageStream 与原始 preprocessImage 逻辑完全一致
// 区别在于接收 io.Reader 提高通用性
func PreprocessImageStream(r io.Reader) ([]float32, error) {
	// 1. 解码图片流
	img, _, err := image.Decode(r)
	if err != nil {
		return nil, fmt.Errorf("decode error: %w", err)
	}

	// 2. 缩放至 224x224 (使用 CatmullRom 算法保持高质量)
	size := 224
	resized := image.NewRGBA(image.Rect(0, 0, size, size))
	draw.CatmullRom.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Src, nil)

	// 3. 准备 Tensor 容器 (3 * 224 * 224)
	tensor := make([]float32, 3*size*size)

	// 4. 归一化并从 HWC 转换为 CHW 格式
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			// RGBA() 返回的是 0-65535 的 uint32
			r, g, b, _ := resized.At(x, y).RGBA()

			idx := y*size + x
			// R 通道
			tensor[idx] = (float32(r)/65535 - clipMean[0]) / clipStd[0]
			// G 通道
			tensor[size*size+idx] = (float32(g)/65535 - clipMean[1]) / clipStd[1]
			// B 通道
			tensor[2*size*size+idx] = (float32(b)/65535 - clipMean[2]) / clipStd[2]
		}
	}
	return tensor, nil
}
