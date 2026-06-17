package clipsdk

import (
	"fmt"
	"math"
	"os"
	"sort"
)

func sortMatches(m []Match) {
	sort.Slice(m, func(i, j int) bool {
		return m[i].Similarity > m[j].Similarity
	})
}

// ValidateEmbeddingFormat 判断 emb 是否符合 ExtractEmbedding 导出的格式。
// expectedDim > 0 时同时检查向量长度，否则只检查格式和归一化。
func ValidateEmbeddingFormat(emb []float32, expectedDim int) error {
	if len(emb) == 0 {
		return fmt.Errorf("empty embedding")
	}

	if expectedDim > 0 && len(emb) != expectedDim {
		return fmt.Errorf("embedding dim mismatch: expect=%d actual=%d", expectedDim, len(emb))
	}

	var sum float32
	for _, v := range emb {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("embedding contains invalid value: %v", v)
		}
		sum += v * v
	}

	norm := float32(math.Sqrt(float64(sum)))
	if norm == 0 {
		return fmt.Errorf("zero embedding is not valid")
	}

	if math.Abs(float64(norm-1.0)) > 1e-3 {
		return fmt.Errorf("embedding is not normalized: norm=%.6f", norm)
	}

	return nil
}

// FileCheck 检查路径是否为有效的可读文件
func FileCheck(path string) error {
	// 1. 获取文件状态
	info, err := os.Stat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("文件不存在: %s", path)
		}
		return fmt.Errorf("无法访问文件: %w", err)
	}

	// 2. 检查是否为目录
	if info.IsDir() {
		return fmt.Errorf("路径是一个目录而非文件: %s", path)
	}

	// 3. 检查文件大小
	if info.Size() == 0 {
		return fmt.Errorf("文件内容为空: %s", path)
	}

	// 4. (可选) 检查文件后缀，防止加载非图片文件
	// ext := strings.ToLower(filepath.Ext(path))
	// if ext != ".jpg" && ext != ".jpeg" && ext != ".png" {
	//     return fmt.Errorf("不支持的文件格式: %s", ext)
	// }

	return nil
}
