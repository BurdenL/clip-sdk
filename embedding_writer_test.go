package clipsdk

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

func TestEmbeddingWriter(t *testing.T) {
	// 1. 使用 t.TempDir() 创建测试用的临时目录，测试完成后 Go 会自动清理
	tmpDir := t.TempDir()

	// 2. 初始化 Writer
	writer, err := NewEmbeddingWriter(tmpDir, "test")
	if err != nil {
		t.Fatalf("Failed to create EmbeddingWriter: %v", err)
	}

	dim := 512

	// 3. 测试单条添加 (Add)
	mockEmbedding1 := make([]float32, dim)
	for i := 0; i < dim; i++ {
		mockEmbedding1[i] = rand.Float32()
	}
	err = writer.Add("image_001.jpg", mockEmbedding1)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	// 4. 测试批量添加 (AddBatch)
	var batchRecords []EmbeddingRecord
	for i := 2; i <= 5; i++ {
		mockEmbedding := make([]float32, dim)
		for j := 0; j < dim; j++ {
			mockEmbedding[j] = rand.Float32()
		}
		batchRecords = append(batchRecords, EmbeddingRecord{
			Filename:  fmt.Sprintf("image_%03d.jpg", i),
			Embedding: mockEmbedding,
		})
	}
	err = writer.AddBatch(batchRecords)
	if err != nil {
		t.Fatalf("AddBatch failed: %v", err)
	}

	// 5. 关闭 Writer，触发 Header 回写
	err = writer.Close()
	if err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// ==========================================
	// 6. 验证逻辑：读取生成的文件，确保数据 100% 正确
	// ==========================================

	// 验证 bin 文件
	binPath := filepath.Join(tmpDir, "embeddings.bin")
	binFile, err := os.Open(binPath)
	if err != nil {
		t.Fatalf("Failed to open generated bin file: %v", err)
	}
	defer binFile.Close()

	var gotCount, gotDim uint32
	// 读取 count
	if err := binary.Read(binFile, binary.LittleEndian, &gotCount); err != nil {
		t.Fatalf("Read count failed: %v", err)
	}
	// 读取 dim
	if err := binary.Read(binFile, binary.LittleEndian, &gotDim); err != nil {
		t.Fatalf("Read dim failed: %v", err)
	}

	// 断言 Header 是否正确
	if gotCount != 5 {
		t.Errorf("Expect count=5, got=%d", gotCount)
	}
	if gotDim != uint32(dim) {
		t.Errorf("Expect dim=%d, got=%d", dim, gotDim)
	}

	// 验证 txt 文件大小或内容
	txtPath := filepath.Join(tmpDir, "filenames.txt")
	txtContent, err := os.ReadFile(txtPath)
	if err != nil {
		t.Fatalf("Failed to read txt file: %v", err)
	}

	// 预期生成的文本内容
	expectTxt := "image_001.jpg\nimage_002.jpg\nimage_003.jpg\nimage_004.jpg\nimage_005.jpg\n"
	if string(txtContent) != expectTxt {
		t.Errorf("Txt content mismatch.\nExpect:\n%s\nGot:\n%s", expectTxt, string(txtContent))
	}
}
