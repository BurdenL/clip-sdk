package clipsdk

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"

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
