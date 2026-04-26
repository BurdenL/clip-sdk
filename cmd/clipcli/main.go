package main

import (
	"encoding/binary"
	"fmt"
	"os"
	"time"

	clipsdk "github.com/BurdenL/clip-sdk"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		return
	}

	switch os.Args[1] {

	case "search":
		// ONNX 推理 + 检索
		// 用法:
		// clipcli search <model.onnx> <index.bin> <index.txt> <image.jpg> <onnxruntime.so> [topk]
		if len(os.Args) < 7 {
			fmt.Println("参数不足")
			usage()
			return
		}

		topK := 5
		if len(os.Args) >= 8 {
			fmt.Sscanf(os.Args[7], "%d", &topK)
		}

		runFullSearch(
			os.Args[2],
			os.Args[3],
			os.Args[4],
			os.Args[5],
			os.Args[6],
			topK,
		)

	case "query":
		// 纯向量检索
		// clipcli query <index.bin> <index.txt> <query_embedding.bin> [topk]
		if len(os.Args) < 5 {
			fmt.Println("参数不足")
			usage()
			return
		}

		topK := 5
		if len(os.Args) >= 6 {
			fmt.Sscanf(os.Args[5], "%d", &topK)
		}

		runQuery(
			os.Args[2],
			os.Args[3],
			os.Args[4],
			topK,
		)

	case "bench":
		// 性能测试
		// clipcli bench <index.bin> <index.txt>
		if len(os.Args) < 4 {
			fmt.Println("参数不足")
			usage()
			return
		}

		runBenchmark(os.Args[2], os.Args[3])

	default:
		usage()
	}
}

func usage() {
	fmt.Println("Clip SDK CLI")
	fmt.Println()
	fmt.Println("用法:")
	fmt.Println("  search <model.onnx> <index.bin> <index.txt> <image.jpg> <onnxruntime.so> [topk]")
	fmt.Println("  query  <index.bin> <index.txt> <query_embedding.bin> [topk]")
	fmt.Println("  bench  <index.bin> <index.txt>")
}

//
// ====================== 功能实现 ======================
//

// ONNX 推理 + 搜索
func runFullSearch(model, bin, txt, image, ortLib string, topK int) {
	// 初始化客户端
	client, err := clipsdk.NewClient(clipsdk.Config{
		ModelPath: model,
		IndexBin:  bin,
		IndexTxt:  txt,
		ORTLib:    ortLib,
	})
	if err != nil {
		fmt.Println("初始化失败:", err)
		return
	}
	defer client.Close()

	// 查询示例
	t0 := time.Now()
	results, err := client.SearchImageByPath(image, topK)
	if err != nil {
		fmt.Println("查询失败:", err)
		return
	}
	elapsed := time.Since(t0)

	fmt.Printf("查询: %s (耗时 %v)\n\nTop-%d:\n", image, elapsed, topK)
	for i, r := range results {
		fmt.Printf("  #%d: %s  sim=%.4f\n", i+1, r.Name, r.Similarity)
	}

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

}

// 纯向量查询
func runQuery(bin, txt, queryFile string, topK int) {
	engine, err := clipsdk.NewPureGoEngine(bin, txt)
	if err != nil {
		fmt.Println("加载索引失败:", err)
		return
	}

	query := make([]float32, 512) // 默认512维
	f, err := os.Open(queryFile)
	if err != nil {
		fmt.Println("读取查询向量失败:", err)
		return
	}
	defer f.Close()

	binary.Read(f, binary.LittleEndian, query)

	t0 := time.Now()
	results := engine.SearchTopK(query, topK)
	elapsed := time.Since(t0)

	fmt.Printf("查询耗时: %v\n\nTop-%d:\n", elapsed, topK)
	for i, r := range results {
		fmt.Printf("  #%d: %s  sim=%.4f\n", i+1, r.Name, r.Similarity)
	}
}

// 性能测试
func runBenchmark(bin, txt string) {
	engine, err := clipsdk.NewPureGoEngine(bin, txt)
	if err != nil {
		fmt.Println("加载失败:", err)
		return
	}

	query := engine.SearchTopK(engine.Index().Embeddings[0], 1)[0]

	iterations := 10000
	t0 := time.Now()

	for i := 0; i < iterations; i++ {
		engine.SearchTopK([]float32{query.Similarity}, 1)
	}

	elapsed := time.Since(t0)

	fmt.Printf("QPS: %.0f\n", float64(iterations)/elapsed.Seconds())
}
