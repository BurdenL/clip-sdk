package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
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
	case "embed":
		// 提取特征向量
		// clipcli embed <model.onnx> <image.jpg> <embedding.bin> <onnxruntime.so>
		if len(os.Args) < 6 {
			fmt.Println("参数不足")
			usage()
			return
		}

		runEmbed(
			os.Args[2],
			os.Args[3],
			os.Args[4],
			os.Args[5],
		)
	case "embedall":
		// 批量提取特征向量
		// clipcli embedall <model.onnx> <index.bin> <index.txt> <image_dir> <output_dir> <onnxruntime.so>
		if len(os.Args) < 8 {
			fmt.Println("参数不足")
			usage()
			return
		}

		runAllEmbed(
			os.Args[2],
			os.Args[3],
			os.Args[4],
			os.Args[5],
			os.Args[6],
			os.Args[7],
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
	fmt.Println("  embed  <model.onnx> <image.jpg> <embedding.bin> <onnxruntime.so>")
	fmt.Println("  embedall <model.onnx>  <index.bin> <index.txt>  <image_dir> <output_dir> <onnxruntime.so>")
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
	}
	elapsed := time.Since(t0)

	fmt.Printf("查询: %s (耗时 %v)\n\nTop-%d:\n", image, elapsed, topK)
	for i, r := range results {
		fmt.Printf("  #%d: %s  sim=%.4f\n", i+1, r.Name, r.Similarity)
	}

	// 范围搜索示例
	data, err := os.ReadFile(image)
	if err != nil {
		fmt.Println("读取图片失败:", err)
		return
	}

	scope := 0.7

	t0 = time.Now()
	results, err = client.SearchScopeByReader(bytes.NewReader(data), float32(scope))
	if err != nil {
		fmt.Println("查询失败:", err)
		return
	}
	elapsed = time.Since(t0)

	fmt.Printf("\n使用图片流查询: %s (耗时 %v)\n\nTop-%f:\n", image, elapsed, scope)
	for i, r := range results {
		fmt.Printf("  #%d: %s  sim=%.4f\n", i+1, r.Name, r.Similarity)
	}

	// 提取特征向量示例
	if emb, err := client.ExtractEmbeddingByReader(bytes.NewReader(data)); err != nil {
		fmt.Println("提取特征向量失败:", err)

	} else {
		fmt.Printf("\n使用图片流查询提取的特征向量: %v\n", emb)
	}

	// // 范围搜索示例
	// baseFile, err := os.Open(".\\source\\img_base64.txt")
	// if err != nil {
	// 	fmt.Println("打开base64文件失败:", err)
	// 	return
	// }
	// defer baseFile.Close()

	// decoder := base64.NewDecoder(base64.StdEncoding, baseFile)

	// t0 = time.Now()
	// results, err = client.SearchScopeByReader(decoder, float32(scope))
	// if err != nil {
	// 	fmt.Println("查询失败:", err)

	// }
	// elapsed = time.Since(t0)

	// fmt.Printf("\n使用base64后的图片流查询: %s (耗时 %v)\n\nTop-%f:\n", image, elapsed, scope)
	// for i, r := range results {
	// 	fmt.Printf("  #%d: %s  sim=%.4f\n", i+1, r.Name, r.Similarity)
	// }

	// if emb, err := client.ExtractEmbeddingByReader(decoder); err != nil {
	// 	fmt.Println("提取特征向量失败:", err)
	// 	return
	// } else {
	// 	fmt.Printf("\n使用base64后的图片流提取的特征向量: %v\n", emb)
	// }
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

func runEmbed(model, image, output, ortLib string) {
	client, err := clipsdk.NewClient(clipsdk.Config{
		ModelPath: model,
		ORTLib:    ortLib,
	})
	if err != nil {
		fmt.Println("初始化失败:", err)
		return
	}
	defer client.Close()

	data, err := os.ReadFile(image)
	if err != nil {
		fmt.Println("读取图片失败:", err)
		return
	}

	embedding, err := client.ExtractEmbeddingByReader(bytes.NewReader(data))
	if err != nil {
		fmt.Println("提取特征向量失败:", err)
		return
	}

	f, err := os.Create(output)
	if err != nil {
		fmt.Println("创建输出文件失败:", err)
		return
	}
	defer f.Close()

	binary.Write(f, binary.LittleEndian, embedding)

	fmt.Printf("特征向量已保存到: %s\n", output)
}

func runAllEmbed(model, bin, txt, imagedir, output, ortLib string) {
	// 1. 初始化 CLIP 客户端
	client, err := clipsdk.NewClient(clipsdk.Config{
		ModelPath: model,
		IndexBin:  bin,
		IndexTxt:  txt,
		ORTLib:    ortLib,
	})
	if err != nil {
		fmt.Println("初始化 CLIP 客户端失败:", err)
		return
	}
	defer client.Close()

	// 2. 创建 EmbeddingWriter 实例（直接使用传入的 output 目录）
	writer, err := clipsdk.NewEmbeddingWriter(output, "test") // 这里可以根据需要传入不同的 tag 来区分不同批次的输出
	if err != nil {
		log.Fatalf("创建 EmbeddingWriter 失败: %v", err)
	}
	// 确保函数退出时，一定会回写 Header 并关闭文件
	defer func() {
		if err := writer.Close(); err != nil {
			log.Printf("延迟关闭 EmbeddingWriter 失败: %v", err)
		}
	}()

	// 3. 读取图片目录下的所有文件
	files, err := os.ReadDir(imagedir)
	if err != nil {
		fmt.Printf("读取图片目录 [%s] 失败: %v\n", imagedir, err)
		return
	}

	// 支持的图片后缀集合
	supportedExts := map[string]bool{
		".jpg":  true,
		".jpeg": true,
		".png":  true,
		".bmp":  true,
		".webp": true,
	}

	fmt.Println("开始批量提取特征向量...")
	t0 := time.Now()
	successCount := 0

	// 4. 循环处理每张图片
	for _, file := range files {
		// 跳过子目录
		if file.IsDir() {
			continue
		}

		filename := file.Name()
		ext := strings.ToLower(filepath.Ext(filename))

		// 过滤非图片文件
		if !supportedExts[ext] {
			continue
		}

		// 拼接完整的图片绝对/相对路径
		fullPath := filepath.Join(imagedir, filename)

		// 读取图片字节流
		data, err := os.ReadFile(fullPath)
		if err != nil {
			fmt.Printf("⚠️ 跳过 - 读取图片失败 [%s]: %v\n", filename, err)
			continue
		}

		// 提取特征向量
		embedding, err := client.ExtractEmbeddingByReader(bytes.NewReader(data))
		if err != nil {
			fmt.Printf("⚠️ 跳过 - 提取特征向量失败 [%s]: %v\n", filename, err)
			continue
		}

		// 写入二进制和文本索引 (内部会自动处理线程安全和 count 计数)
		err = writer.Add(filename, embedding)
		if err != nil {
			fmt.Printf("❌ 写入特征库失败 [%s]: %v\n", filename, err)
			return // 写入底层的 I/O 错误通常是致命的（如同磁盘满了），直接退出
		}

		successCount++
		if successCount%100 == 0 {
			fmt.Printf(" 已处理 %d 张图片...\n", successCount)
		}
	}

	elapsed := time.Since(t0)
	fmt.Printf("\n特征提取完毕！\n")
	fmt.Printf("成功处理图片数: %d 张\n", successCount)
	fmt.Printf("总耗时: %v (平均 %.2f ms/张)\n", elapsed, float64(elapsed.Milliseconds())/float64(successCount))
	fmt.Printf("特征向量和索引文件名已保存到目录: %s\n", output)
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
