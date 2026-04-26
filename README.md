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

// 范围搜索示例
f, err := os.Open(image)
if err != nil {
    return
}
defer f.Close()

scope := 0.7

t0 = time.Now()
results, err = client.SearchScopeByReader(f, float32(scope))
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