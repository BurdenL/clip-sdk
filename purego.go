package clipsdk

type PureGoSearchEngine struct {
	index *ImageIndex
}

// NewPureGoEngine 创建一个纯 Go 实现的搜索引擎实例，加载指定的索引数据（线程安全）
func NewPureGoEngine(bin, txt string) (*PureGoSearchEngine, error) {
	idx, err := LoadIndex(bin, txt)
	if err != nil {
		return nil, err
	}
	return &PureGoSearchEngine{index: idx}, nil
}

// SearchTopK 接收查询特征向量，返回 Top-K 结果（线程安全）
func (e *PureGoSearchEngine) SearchTopK(query []float32, k int) []Match {
	results := make([]Match, len(e.index.Embeddings))
	for i, emb := range e.index.Embeddings {
		results[i] = Match{
			Name:       e.index.Names[i],
			Index:      i,
			Similarity: dotProduct(query, emb),
		}
	}
	sortMatches(results)
	if k > len(results) {
		k = len(results)
	}
	return results[:k]
}

// Index 返回当前加载的索引数据（线程安全）
func (e *PureGoSearchEngine) Index() *ImageIndex {
	return e.index
}
