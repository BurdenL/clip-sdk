package clipsdk

import (
	"fmt"
	"os"
	"sort"
)

func sortMatches(m []Match) {
	sort.Slice(m, func(i, j int) bool {
		return m[i].Similarity > m[j].Similarity
	})
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
