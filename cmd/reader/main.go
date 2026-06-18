package main

import (
	"fmt"
	"path/filepath"

	clipsdk "github.com/BurdenL/clip-sdk" // 替换为你的实际包导入路径
)

func main() {
	// 1. 指定输出目录（如果不存在，代码内部会自动创建）
	outputDir := "./assets"

	// 模拟一些 512 维的特征向量 (例如 CLIP-ViT-B/32 的标准输出维度)
	dim := 512

	// 2. 创建 EmbeddingWriter 实例
	binPath := filepath.Join(outputDir, "image_index.bin")
	txtPath := filepath.Join(outputDir, "image_index.txt")

	records, err := clipsdk.ReadEmbeddingsFromPath(binPath, txtPath)
	if err != nil {
		fmt.Printf("ReadEmbeddingsFromPath failed: %v", err)
	}

	// if len(records) != 2 {
	// 	fmt.Printf("expected 2 records, got %d", len(records))
	// }

	for i, rec := range records {

		fmt.Printf("Record %d: Filename=%s, EmbeddingDim=%d\n", i+1, rec.Filename, len(rec.Embedding))
		expectedName := fmt.Sprintf("image_%d.jpg", i+1)
		if rec.Filename != expectedName {
			fmt.Printf("expected filename %s, got %s\n", expectedName, rec.Filename)
		}
		if len(rec.Embedding) != dim {
			fmt.Printf("expected embedding dim %d, got %d\n", dim, len(rec.Embedding))
		}
	}

	fmt.Println("\n开始流式读取...")

	count := 0
	err = clipsdk.StreamEmbeddings(binPath, txtPath, func(rec clipsdk.EmbeddingRecord) error {
		count++

		// fmt.Printf("Record %d: Filename=%s, EmbeddingDim=%d\n", count, rec.Filename, len(rec.Embedding))

		if rec.Filename != fmt.Sprintf("image_%d.jpg", count) {
			fmt.Printf("expected filename image_%d.jpg, got %s\n", count, rec.Filename)
		}
		if len(rec.Embedding) != dim {
			fmt.Printf("expected embedding dim %d, got %d\n", dim, len(rec.Embedding))
		}
		return nil
	})

	if err != nil {
		fmt.Printf("StreamEmbeddings failed: %v", err)
	}

}
