package clipsdk

import (
	"errors"
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
	ort.SetSharedLibraryPath(onnxLib)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, err
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
func (e *CLIPSearchEngine) ExtractEmbedding(imagePath string) ([]float32, error) {
	w := <-e.workers
	defer func() { e.workers <- w }()

	inputData, err := preprocessImage(imagePath)
	if err != nil {
		return nil, err
	}

	// ✅ 零拷贝写入 tensor
	copy(w.inputTensor.GetData(), inputData)

	// 推理
	if err := w.session.Run(); err != nil {
		return nil, err
	}

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

// ====================== 检索 ======================

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

func (e *CLIPSearchEngine) SearchTopKByFile(path string, k int) ([]Match, error) {
	emb, err := e.ExtractEmbedding(path)
	if err != nil {
		return nil, err
	}
	return e.SearchTopK(emb, k), nil
}
