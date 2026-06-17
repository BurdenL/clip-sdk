package clipsdk

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
)

// ReadEmbeddings 从已经打开的 bin 和 txt 文件中读取全部 embedding 记录。
//
// binFile 需要包含头部信息: count(uint32) + dim(uint32)，随后按顺序写入每条 float32 向量。
// txtFile 每一行对应一张图片的文件名，行数必须与 bin 中的 count 保持一致。
func ReadEmbeddings(binFile *os.File, txtFile *os.File) (records []EmbeddingRecord, err error) {
	if binFile == nil || txtFile == nil {
		return nil, fmt.Errorf("binFile and txtFile must not be nil")
	}

	// 先回到文件开头，确保可重复读取。
	if _, err := binFile.Seek(0, 0); err != nil {
		return nil, err
	}
	if _, err := txtFile.Seek(0, 0); err != nil {
		return nil, err
	}

	var count, dim uint32
	if err := binary.Read(binFile, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("read bin header failed: %w", err)
	}
	if err := binary.Read(binFile, binary.LittleEndian, &dim); err != nil {
		return nil, fmt.Errorf("read bin header failed: %w", err)
	}

	scanner := bufio.NewScanner(txtFile)
	for scanner.Scan() {
		filename := strings.TrimSpace(scanner.Text())
		if filename == "" {
			return nil, fmt.Errorf("empty filename line in txt file")
		}

		embedding, err := readEmbedding(binFile, int(dim))
		if err != nil {
			return nil, err
		}

		records = append(records, EmbeddingRecord{
			Filename:  filename,
			Embedding: embedding,
		})
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	if uint32(len(records)) != count {
		return nil, fmt.Errorf("record count mismatch: txt=%d bin=%d", len(records), count)
	}

	return records, nil
}

// ReadEmbeddingsFromPath 打开路径指定的 bin/txt 文件，并读取所有 embedding 记录。
func ReadEmbeddingsFromPath(binPath, txtPath string) ([]EmbeddingRecord, error) {
	binFile, err := os.Open(binPath)
	if err != nil {
		return nil, err
	}
	defer binFile.Close()

	txtFile, err := os.Open(txtPath)
	if err != nil {
		return nil, err
	}
	defer txtFile.Close()

	return ReadEmbeddings(binFile, txtFile)
}

// NewEmbeddingStream 打开 bin/txt 文件并创建一个可增量读取的流。
func NewEmbeddingStream(binPath, txtPath string) (*EmbeddingStream, error) {
	binFile, err := os.Open(binPath)
	if err != nil {
		return nil, err
	}

	txtFile, err := os.Open(txtPath)
	if err != nil {
		binFile.Close()
		return nil, err
	}

	stream, err := NewEmbeddingStreamFromFiles(binFile, txtFile)
	if err != nil {
		binFile.Close()
		txtFile.Close()
		return nil, err
	}
	return stream, nil
}

// NewEmbeddingStreamFromFiles 使用已经打开的 bin/txt 文件创建流对象。
// 该方法只读取 header，不会一次性加载全部嵌入数据，因此适合大文件场景。
func NewEmbeddingStreamFromFiles(binFile, txtFile *os.File) (*EmbeddingStream, error) {
	if binFile == nil || txtFile == nil {
		return nil, fmt.Errorf("binFile and txtFile must not be nil")
	}

	// 将文件位置重置到开头，确保 header 读取正确。
	if _, err := binFile.Seek(0, 0); err != nil {
		return nil, err
	}
	if _, err := txtFile.Seek(0, 0); err != nil {
		return nil, err
	}

	var count, dim uint32
	if err := binary.Read(binFile, binary.LittleEndian, &count); err != nil {
		return nil, fmt.Errorf("read bin header failed: %w", err)
	}
	if err := binary.Read(binFile, binary.LittleEndian, &dim); err != nil {
		return nil, fmt.Errorf("read bin header failed: %w", err)
	}

	return &EmbeddingStream{
		binFile:    binFile,
		txtFile:    txtFile,
		txtScanner: bufio.NewScanner(txtFile),
		count:      count,
		dim:        dim,
	}, nil
}

// EmbeddingStream 支持流式读取 bin/txt 中的 embedding 记录，避免一次性加载全部数据。
type EmbeddingStream struct {
	binFile    *os.File
	txtFile    *os.File
	txtScanner *bufio.Scanner
	count      uint32
	dim        uint32
	readCount  uint32
}

// Next 读取下一条记录，如果读取完成则返回 io.EOF。
func (s *EmbeddingStream) Next() (EmbeddingRecord, error) {
	if s == nil {
		return EmbeddingRecord{}, fmt.Errorf("stream is nil")
	}

	if s.txtScanner == nil {
		return EmbeddingRecord{}, fmt.Errorf("stream not initialized")
	}

	if !s.txtScanner.Scan() {
		if err := s.txtScanner.Err(); err != nil {
			return EmbeddingRecord{}, err
		}
		return EmbeddingRecord{}, io.EOF
	}

	filename := strings.TrimSpace(s.txtScanner.Text())
	if filename == "" {
		return EmbeddingRecord{}, fmt.Errorf("empty filename line in txt file")
	}

	embedding, err := readEmbedding(s.binFile, int(s.dim))
	if err != nil {
		return EmbeddingRecord{}, err
	}

	s.readCount++
	return EmbeddingRecord{Filename: filename, Embedding: embedding}, nil
}

// Close 关闭流所使用的文件资源。
func (s *EmbeddingStream) Close() error {
	if s == nil {
		return nil
	}

	var err error
	if s.binFile != nil {
		err = s.binFile.Close()
	}
	if s.txtFile != nil {
		if err2 := s.txtFile.Close(); err == nil {
			err = err2
		}
	}
	return err
}

// StreamEmbeddings 通过回调逐条处理 embedding 记录，适合大文件流式消费。
func StreamEmbeddings(binPath, txtPath string, handler func(EmbeddingRecord) error) error {
	stream, err := NewEmbeddingStream(binPath, txtPath)
	if err != nil {
		return err
	}
	defer stream.Close()

	for {
		rec, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if err := handler(rec); err != nil {
			return err
		}
	}
	return nil
}

// readEmbedding 读取一个固定维度的 embedding 向量，直接从 bin 文件当前位置读取。
func readEmbedding(binFile *os.File, dim int) ([]float32, error) {
	if dim <= 0 {
		return nil, fmt.Errorf("invalid embedding dimension: %d", dim)
	}

	emb := make([]float32, dim)
	if err := binary.Read(binFile, binary.LittleEndian, emb); err != nil {
		return nil, err
	}
	return emb, nil
}
