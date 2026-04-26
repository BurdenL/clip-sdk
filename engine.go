package clipsdk

import (
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

type CLIPSearchEngine struct {
	index *ImageIndex

	// worker pool
	workers chan *worker
	wg      sync.WaitGroup

	// config
	batchSize int
}

type worker struct {
	session      *ort.AdvancedSession
	inputTensor  *ort.Tensor[float32]
	outputTensor *ort.Tensor[float32]
	inputShape   ort.Shape
	outputShape  ort.Shape
}

// ====================== 初始化 ======================

func NewCLIPSearchEngine(onnxPath, indexBin, indexTxt, onnxLib string) (*CLIPSearchEngine, error) {

	fmt.Printf("Loading ONNX model from %s\n", onnxPath)
	fmt.Printf("Loading index from %s and %s\n", indexBin, indexTxt)
	fmt.Printf("Using ONNX Runtime library: %s\n", onnxLib)
	ort.SetSharedLibraryPath(onnxLib)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("初始化 ONNX Runtime 失败: %w", err)
	}

	idx, err := LoadIndex(indexBin, indexTxt)
	if err != nil {
		return nil, err
	}

	engine := &CLIPSearchEngine{
		index:     idx,
		batchSize: 1, // 默认单张（可改成 8/16）
	}

	// worker 数 = CPU 核心数
	numWorkers := runtime.NumCPU()
	engine.workers = make(chan *worker, numWorkers)

	for i := 0; i < numWorkers; i++ {
		w, err := newWorker(onnxPath, idx.Dim, engine.batchSize)
		if err != nil {
			return nil, err
		}
		engine.workers <- w
	}

	return engine, nil
}

// 创建 worker（每个 worker 独立 session）
func newWorker(onnxPath string, dim int, batch int) (*worker, error) {
	inputShape := ort.NewShape(int64(batch), 3, 224, 224)
	outputShape := ort.NewShape(int64(batch), int64(dim))

	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, err
	}

	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, err
	}

	session, err := ort.NewAdvancedSession(
		onnxPath,
		[]string{"image"},
		[]string{"embedding"},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		nil,
	)
	if err != nil {
		return nil, err
	}

	return &worker{
		session:      session,
		inputTensor:  inputTensor,
		outputTensor: outputTensor,
		inputShape:   inputShape,
		outputShape:  outputShape,
	}, nil
}

// ====================== 关闭 ======================

func (e *CLIPSearchEngine) Close() {
	close(e.workers)
	for w := range e.workers {
		w.session.Destroy()
		w.inputTensor.Destroy()
		w.outputTensor.Destroy()
	}
	ort.DestroyEnvironment()
}

// ====================== 核心推理 ======================

// 单张图（线程安全）
// ExtractEmbeddingByPath 依然保留，作为 ExtractEmbedding 的便捷包装
func (e *CLIPSearchEngine) ExtractEmbeddingByPath(imagePath string) ([]float32, error) {
	f, err := os.Open(imagePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return e.ExtractEmbedding(f)
}

// ExtractEmbedding 接收 io.Reader 流（线程安全） 生成图片的特征向量
func (e *CLIPSearchEngine) ExtractEmbedding(r io.Reader) ([]float32, error) {
	// 从池中获取 worker 资源
	w := <-e.workers
	defer func() { e.workers <- w }()

	// 调用流式预处理方法
	inputData, err := PreprocessImageStream(r)
	if err != nil {
		return nil, err
	}

	// ✅ 零拷贝写入 tensor (确保 inputData 长度与 Tensor 匹配)
	copy(w.inputTensor.GetData(), inputData)

	// 执行推理
	if err := w.session.Run(); err != nil {
		return nil, fmt.Errorf("ONNX run error: %w", err)
	}

	// 获取输出并深拷贝
	output := w.outputTensor.GetData()
	emb := make([]float32, len(output))
	copy(emb, output)

	return l2Normalize(emb), nil
}

// ====================== Batch 推理 ======================

func (e *CLIPSearchEngine) ExtractEmbeddingBatch(paths []string) ([][]float32, error) {
	if len(paths) == 0 {
		return nil, errors.New("empty input")
	}

	w := <-e.workers
	defer func() { e.workers <- w }()

	batch := len(paths)
	if batch > int(w.inputShape[0]) {
		return nil, errors.New("batch too large")
	}

	input := w.inputTensor.GetData()
	perImageSize := 3 * 224 * 224

	for i, p := range paths {
		data, err := preprocessImage(p)
		if err != nil {
			return nil, err
		}
		copy(input[i*perImageSize:(i+1)*perImageSize], data)
	}

	if err := w.session.Run(); err != nil {
		return nil, err
	}

	output := w.outputTensor.GetData()
	dim := e.index.Dim

	results := make([][]float32, batch)

	for i := 0; i < batch; i++ {
		start := i * dim
		end := start + dim

		emb := make([]float32, dim)
		copy(emb, output[start:end])
		results[i] = l2Normalize(emb)
	}

	return results, nil
}

// ====================== 搜索 ======================
func (e *CLIPSearchEngine) SearchTopK(queryEmb []float32, k int) []Match {
	results := make([]Match, len(e.index.Embeddings))

	for i, emb := range e.index.Embeddings {
		results[i] = Match{
			Name:       e.index.Names[i],
			Index:      i,
			Similarity: dotProduct(queryEmb, emb),
		}
	}

	sortMatches(results)

	if k > len(results) {
		k = len(results)
	}
	return results[:k]
}

// SearchTopKByFile 接收图片路径，直接返回 Top-K 结果（线程安全）
func (e *CLIPSearchEngine) SearchTopKByFile(path string, k int) ([]Match, error) {
	// 直接调用 ExtractEmbeddingByPath 获取特征向量
	emb, err := e.ExtractEmbeddingByPath(path)

	// fmt.Printf("Extracted embedding for %s: %v\n", path, emb)
	if err != nil {
		return nil, err
	}
	return e.SearchTopK(emb, k), nil
}

// SearchTopKByReader 接收图片流，直接返回 Top-K 结果（线程安全）
func (e *CLIPSearchEngine) SearchTopKByReader(r io.Reader, k int) ([]Match, error) {
	// 直接调用 ExtractEmbedding 获取特征向量
	emb, err := e.ExtractEmbedding(r)

	// fmt.Printf("Extracted embedding: %v\n", emb)
	if err != nil {
		return nil, err
	}
	return e.SearchTopK(emb, k), nil
}

// SearchScopeByFile 接收图片路径，直接返回相似度高于指定阈值的结果（线程安全）
func (e *CLIPSearchEngine) SearchScopeByFile(path string, scope float32) ([]Match, error) {
	// 直接调用 ExtractEmbeddingByPath 获取特征向量
	emb, err := e.ExtractEmbeddingByPath(path)

	// fmt.Printf("Extracted embedding for %s: %v\n", path, emb)
	if err != nil {
		return nil, err
	}
	return e.SearchScope(emb, scope), nil
}

// SearchScopeByReader 接收图片流，直接返回相似度高于指定阈值的结果（线程安全）
func (e *CLIPSearchEngine) SearchScopeByReader(r io.Reader, scope float32) ([]Match, error) {
	// 直接调用 ExtractEmbedding 获取特征向量
	emb, err := e.ExtractEmbedding(r)

	// fmt.Printf("Extracted embedding: %v\n", emb)
	if err != nil {
		return nil, err
	}
	return e.SearchScope(emb, scope), nil
}

// SearchScope 在索引库中搜索所有相似度高于指定阈值的匹配项。
//
// 参数:
//
//	queryEmb: 查询向量（通常由 ExtractEmbedding 生成），应为 L2 归一化后的向量。
//	score:    相似度阈值（通常在 0 到 1 之间）。只有相似度 >= score 的结果才会被返回。
//
// 返回值:
//
//	返回一个 Match 切片，按相似度从高到低（降序）排列。如果未找到符合条件的匹配项，返回 nil 或空切片。
func (e *CLIPSearchEngine) SearchScope(queryEmb []float32, score float32) []Match {
	// 1. 初始化一个空切片，用于存放符合条件的结果
	var results []Match

	// 2. 遍历索引
	for i, emb := range e.index.Embeddings {
		similarity := dotProduct(queryEmb, emb)

		// 3. 仅当相似度大于或等于传入的阈值 score 时才记录
		if similarity >= score {
			results = append(results, Match{
				Name:       e.index.Names[i],
				Index:      i,
				Similarity: similarity,
			})
		}
	}

	// 4. 对符合条件的结果进行排序（从高到低）
	sortMatches(results)

	return results
}
