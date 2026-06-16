package main

import (
	"fmt"
	"log"
	"math/rand"

	clipsdk "github.com/BurdenL/clip-sdk" // 替换为你的实际包导入路径
)

func main() {
	// 1. 指定输出目录（如果不存在，代码内部会自动创建）
	outputDir := "./output"

	// 2. 创建 EmbeddingWriter 实例
	writer, err := clipsdk.NewEmbeddingWriter(outputDir, "test") // 这里可以根据需要传入不同的 tag 来区分不同批次的输出
	if err != nil {
		log.Fatalf("创建 EmbeddingWriter 失败: %v", err)
	}

	// 模拟一些 512 维的特征向量 (例如 CLIP-ViT-B/32 的标准输出维度)
	dim := 512

	fmt.Println("开始写入特征向量...")

	// 3. 示例：单条数据添加 (Add)
	mockEmbedding1 := make([]float32, dim)
	for i := 0; i < dim; i++ {
		mockEmbedding1[i] = rand.Float32()
	}

	err = writer.Add("image_001.jpg", mockEmbedding1)
	if err != nil {
		log.Fatalf("单条添加失败: %v", err)
	}

	// 4. 示例：批量数据添加 (AddBatch)
	var batchRecords []clipsdk.EmbeddingRecord

	for i := 2; i <= 5; i++ {
		mockEmbedding := make([]float32, dim)
		for j := 0; j < dim; j++ {
			mockEmbedding[j] = rand.Float32()
		}

		record := clipsdk.EmbeddingRecord{
			Filename:  fmt.Sprintf("image_%03d.jpg", i),
			Embedding: mockEmbedding,
		}
		batchRecords = append(batchRecords, record)
	}

	err = writer.AddBatch(batchRecords)
	if err != nil {
		log.Fatalf("批量添加失败: %v", err)
	}

	// 5. 核心步骤：必须调用 Close()！
	// 此时内部才会将缓存刷入磁盘，并回到文件头写入总数 (count=5) 和维度 (dim=512)
	err = writer.Close()
	if err != nil {
		log.Fatalf("关闭并回写 Header 失败: %v", err)
	}

	fmt.Println("写入完成！成功生成 image_index_test.bin 和 image_index_test.txt（位于 output 目录）")
}
