# Clip SDK (Go)



## 推送指定版本
```cmd
    git tag v0.1.0

    git push origin v0.1.0
```

## 安装
```cmd
go get github.com/BurdenL/clip-sdk@latest
```
## 使用
```golang
client, _ := clipsdk.NewClient(clipsdk.Config{
ModelPath: "clip.onnx",
IndexBin:  "index.bin",
IndexTxt:  "index.txt",
ORTLib:    "libonnxruntime.so",
})

results, _ := client.SearchImage("test.jpg", 5)

// 提取特征向量示例
f, err := os.Open(image)
if err != nil {
    return
}
defer f.Close()

embedding, err := client.ExtractEmbeddingByReader(f)
if err != nil {
    fmt.Println("提取特征向量失败:", err)
    return
}
fmt.Printf("特征向量长度: %d\n", len(embedding))

// 范围搜索示例
f2, err := os.Open(image)
if err != nil {
    return
}
defer f2.Close()

scope := 0.7

t0 = time.Now()
results, err = client.SearchScopeByReader(f2, float32(scope))
if err != nil {
    fmt.Println("查询失败:", err)
    return
}
elapsed = time.Since(t0)

fmt.Printf("\n使用图片流查询: %s (耗时 %v)\n\nTop-%f:\n", image, elapsed, scope)
for i, r := range results {
    fmt.Printf("  #%d: %s  sim=%.4f\n", i+1, r.Name, r.Similarity)
}


```

更多方法示例

    cmd/clipcli/main.go

## 特征向量写入（EmbeddingWriter）

库内提供 `EmbeddingWriter` 用于将提取到的特征向量写入磁盘，通常配合批量提取流程使用。使用示例：

```golang
writer, err := clipsdk.NewEmbeddingWriter("./output", "batch1")
if err != nil {
		return err
}
defer writer.Close() // 必须 Close() 来回写 Header

// 单条写入
_ = writer.Add("image_001.jpg", embedding)

// 批量写入
records := []clipsdk.EmbeddingRecord{{Filename: "image_002.jpg", Embedding: emb2}}
_ = writer.AddBatch(records)
```

生成文件说明（假设 tag 为 `batch1`，输出目录为 `./output`）：
- 二进制文件：`image_index_batch1.bin`，结构为：
	- Header（8 字节）：`count` (uint32, little-endian) + `dim` (uint32, little-endian)
	- 随后按行存放每个向量的原始 float32 值，按小端序连续写入，向量顺序与文本文件一致。
- 文本文件：`image_index_batch1.txt`，每行对应一个图片文件名（与二进制向量顺序一一对应）。

注意事项：
- 写入完成后必须调用 `Close()`，函数会刷新缓冲区并回写文件头的 `count` 与 `dim`。
- 写入时会做维度检查，所有向量必须具有相同维度。



图片流处理
```golang

import (
	"encoding/base64"
	"strings"
	"net/http"
)

func handleBase64InMultipart(w http.ResponseWriter, r *http.Request, client *clipsdk.Client) {
	// 1. 解析表单
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		http.Error(w, "无法解析表单", http.StatusBadRequest)
		return
	}

	// 2. 获取 Base64 字符串 (假设字段名是 "image_base64")
	base64Str := r.FormValue("image_base64")
	if base64Str == "" {
		http.Error(w, "Base64 数据为空", http.StatusBadRequest)
		return
	}

	// 3. 处理 Data URL 前缀 (例如 "data:image/jpeg;base64,")
	// 如果你的数据包含这个前缀，必须先去掉它，否则 Base64 解码会报错
	if i := strings.Index(base64Str, ","); i != -1 {
		base64Str = base64Str[i+1:]
	}

	// 4. ✅ 核心高效逻辑：使用 base64.NewDecoder
	// 将字符串包装成一个 io.Reader
	base64Reader := strings.NewReader(base64Str)
	decoder := base64.NewDecoder(base64.StdEncoding, base64Reader)

	// 5. 直接传入你的接口
	// 此时 decoder 本身就是 io.Reader，它会在被读取时才动态解码
	matches, err := client.SearchImageByReader(decoder, 5)
	if err != nil {
		http.Error(w, "图片搜索失败: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// ... 返回结果
}

```