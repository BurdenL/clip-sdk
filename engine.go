package clipsdk

import (
	ort "github.com/yalue/onnxruntime_go"
)

type CLIPSearchEngine struct {
	session     *ort.AdvancedSession
	index       *ImageIndex
	inputShape  ort.Shape
	outputShape ort.Shape
}

func NewCLIPSearchEngine(onnxPath, indexBin, indexTxt, onnxLib string) (*CLIPSearchEngine, error) {
	ort.SetSharedLibraryPath(onnxLib)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, err
	}

	idx, err := LoadIndex(indexBin, indexTxt)
	if err != nil {
		return nil, err
	}

	inputShape := ort.NewShape(1, 3, 224, 224)
	outputShape := ort.NewShape(1, int64(idx.Dim))

	inputTensor, _ := ort.NewEmptyTensor[float32](inputShape)
	outputTensor, _ := ort.NewEmptyTensor[float32](outputShape)

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

	return &CLIPSearchEngine{
		session:     session,
		index:       idx,
		inputShape:  inputShape,
		outputShape: outputShape,
	}, nil
}

func (e *CLIPSearchEngine) Close() {
	if e.session != nil {
		e.session.Destroy()
	}
	ort.DestroyEnvironment()
}

func (e *CLIPSearchEngine) SearchTopKByFile(imagePath string, k int) ([]Match, error) {
	embedding, err := e.ExtractEmbedding(imagePath)
	if err != nil {
		return nil, err
	}
	return e.SearchTopK(embedding, k), nil
}

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
