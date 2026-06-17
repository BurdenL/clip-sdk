package clipsdk

import (
	"fmt"
	"path/filepath"
	"testing"
)

// TestReadEmbeddingsFromPath 验证从文件路径读取 bin/txt 并返回完整 EmbeddingRecord 列表。
func TestReadEmbeddingsFromPath(t *testing.T) {
	tmpDir := t.TempDir()

	writer, err := NewEmbeddingWriter(tmpDir, "test")
	if err != nil {
		t.Fatalf("failed to create EmbeddingWriter: %v", err)
	}

	dim := 4
	for i := 0; i < 2; i++ {
		emb := make([]float32, dim)
		for j := 0; j < dim; j++ {
			emb[j] = float32(i+j) + 1.0
		}
		if err := writer.Add(fmt.Sprintf("image_%d.jpg", i+1), emb); err != nil {
			t.Fatalf("writer add failed: %v", err)
		}
	}

	if err := writer.Close(); err != nil {
		t.Fatalf("writer close failed: %v", err)
	}

	binPath := filepath.Join(tmpDir, "image_index_test.bin")
	txtPath := filepath.Join(tmpDir, "image_index_test.txt")

	records, err := ReadEmbeddingsFromPath(binPath, txtPath)
	if err != nil {
		t.Fatalf("ReadEmbeddingsFromPath failed: %v", err)
	}

	if len(records) != 2 {
		t.Fatalf("expected 2 records, got %d", len(records))
	}

	for i, rec := range records {
		expectedName := fmt.Sprintf("image_%d.jpg", i+1)
		if rec.Filename != expectedName {
			t.Fatalf("expected filename %s, got %s", expectedName, rec.Filename)
		}
		if len(rec.Embedding) != dim {
			t.Fatalf("expected embedding dim %d, got %d", dim, len(rec.Embedding))
		}
	}
}

// TestStreamEmbeddings 验证流式读取接口 StreamEmbeddings 能按顺序消费每条记录。
func TestStreamEmbeddings(t *testing.T) {
	tmpDir := t.TempDir()

	writer, err := NewEmbeddingWriter(tmpDir, "stream")
	if err != nil {
		t.Fatalf("failed to create EmbeddingWriter: %v", err)
	}

	dim := 4
	for i := 0; i < 3; i++ {
		emb := make([]float32, dim)
		for j := 0; j < dim; j++ {
			emb[j] = float32(i+j) + 1.0
		}
		if err := writer.Add(fmt.Sprintf("image_%d.jpg", i+1), emb); err != nil {
			t.Fatalf("writer add failed: %v", err)
		}
	}

	if err := writer.Close(); err != nil {
		t.Fatalf("writer close failed: %v", err)
	}

	binPath := filepath.Join(tmpDir, "image_index_stream.bin")
	txtPath := filepath.Join(tmpDir, "image_index_stream.txt")

	count := 0
	err = StreamEmbeddings(binPath, txtPath, func(rec EmbeddingRecord) error {
		count++
		if rec.Filename != fmt.Sprintf("image_%d.jpg", count) {
			t.Fatalf("expected filename image_%d.jpg, got %s", count, rec.Filename)
		}
		if len(rec.Embedding) != dim {
			t.Fatalf("expected embedding dim %d, got %d", dim, len(rec.Embedding))
		}
		return nil
	})
	if err != nil {
		t.Fatalf("StreamEmbeddings failed: %v", err)
	}

	if count != 3 {
		t.Fatalf("expected 3 streamed records, got %d", count)
	}
}
