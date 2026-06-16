package clipsdk

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// EmbeddingWriter 用于将特征向量和对应的文件名写入磁盘，支持批量添加和线程安全
type EmbeddingWriter struct {
	binFile   *os.File      // 结构: Header + Embeddings
	binWriter *bufio.Writer // 用于缓冲写入binFile

	txtFile   *os.File      // 结构: 每行一个文件名
	txtWriter *bufio.Writer // 用于缓冲写入txtFile

	count uint32 // 已写入的记录数
	dim   uint32 // 特征向量维度

	closed bool

	mu sync.Mutex // 保护count和dim的更新，以及Close方法的调用
}

// NewEmbeddingWriter 创建一个新的EmbeddingWriter，输出目录为outputDir
func NewEmbeddingWriter(outputDir string, tag string) (w *EmbeddingWriter, err error) {
	err = os.MkdirAll(outputDir, 0755)
	if err != nil {
		return nil, err
	}

	binPath := filepath.Join(outputDir, fmt.Sprintf("image_index_%s.bin", tag))
	txtPath := filepath.Join(outputDir, fmt.Sprintf("image_index_%s.txt", tag))

	binFile, err := os.Create(binPath)
	if err != nil {
		return nil, err
	}

	txtFile, err := os.Create(txtPath)
	if err != nil {
		binFile.Close()
		return nil, err
	}

	// 错误处理：如果后续初始化失败，安全关闭文件
	defer func() {
		if err != nil {
			binFile.Close()
			txtFile.Close()
		}
	}()

	// 【修复 3】: 初始化 bufio.Writer
	binWriter := bufio.NewWriter(binFile)
	txtWriter := bufio.NewWriter(txtFile)

	// 【修复 2】: 在 binFile 开头预留 8 字节的 Header 空间 (count 4字节 + dim 4字节)
	// 这样可以防止 Close() 里的 Seek(0,0) 回写时覆盖前几个向量的数据
	var placeholder [8]byte
	_, err = binWriter.Write(placeholder[:])
	if err != nil {
		return nil, err
	}

	// 【修复 1】: 正常返回组装后的对象，并闭合函数
	return &EmbeddingWriter{
		binFile:   binFile,
		binWriter: binWriter,
		txtFile:   txtFile,
		txtWriter: txtWriter,
	}, nil
} // <--- 这里之前漏掉了结束大括号

// Add 添加一个文件的特征向量和对应的文件名
func (w *EmbeddingWriter) Add(filename string, embedding []float32) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	return w.addLocked(filename, embedding)
}

type EmbeddingRecord struct {
	Filename  string
	Embedding []float32
}

func (w *EmbeddingWriter) AddBatch(records []EmbeddingRecord) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	for _, record := range records {
		err := w.addLocked(record.Filename, record.Embedding)
		if err != nil {
			return err
		}
	}

	return nil
}

// addLocked 在持有锁的情况下添加记录，确保线程安全
func (w *EmbeddingWriter) addLocked(filename string, embedding []float32) error {
	if w.closed {
		return fmt.Errorf("writer already closed")
	}

	if len(embedding) == 0 {
		return fmt.Errorf("empty embedding")
	}

	if w.dim == 0 {
		w.dim = uint32(len(embedding))
	}

	if uint32(len(embedding)) != w.dim {
		return fmt.Errorf("embedding dimension mismatch: expect=%d actual=%d", w.dim, len(embedding))
	}

	// 写入二进制向量
	err := binary.Write(w.binWriter, binary.LittleEndian, embedding)
	if err != nil {
		return err
	}

	// 写入文本文件名
	_, err = w.txtWriter.WriteString(filename + "\n")
	if err != nil {
		return err
	}

	w.count++
	return nil
}

// Close 关闭文件并回写Header
func (w *EmbeddingWriter) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.closed {
		return nil
	}

	w.closed = true

	// 必须先刷新缓冲，确保所有数据都从内存写入到操作系统的文件流中
	if err := w.txtWriter.Flush(); err != nil {
		return err
	}

	if err := w.binWriter.Flush(); err != nil {
		return err
	}

	// 回写 Header 前，先确保之前 append 的向量数据落盘（或者利用 Seek 穿透缓冲）
	// 因为用的是 binFile (底层文件对象) 进行 Seek 和 Write，所以上面必须执行过 binWriter.Flush()
	_, err := w.binFile.Seek(0, 0)
	if err != nil {
		return err
	}

	// 写入最终的 count
	if err := binary.Write(w.binFile, binary.LittleEndian, w.count); err != nil {
		return err
	}

	// 写入最终的 dim
	if err := binary.Write(w.binFile, binary.LittleEndian, w.dim); err != nil {
		return err
	}

	// 最终同步并关闭
	if err := w.binFile.Sync(); err != nil {
		return err
	}

	if err := w.binFile.Close(); err != nil {
		return err
	}

	if err := w.txtFile.Close(); err != nil {
		return err
	}

	return nil
}
