# Clip SDK (Go)

## 安装

go get github.com/BurdenL/clip-sdk@latest

## 使用

client, _ := clipsdk.NewClient(clipsdk.Config{
ModelPath: "clip.onnx",
IndexBin:  "index.bin",
IndexTxt:  "index.txt",
ORTLib:    "libonnxruntime.so",
})

results, _ := client.SearchImage("test.jpg", 5)