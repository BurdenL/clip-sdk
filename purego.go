package clipsdk

type PureGoSearchEngine struct {
	index *ImageIndex
}

func NewPureGoEngine(bin, txt string) (*PureGoSearchEngine, error) {
	idx, err := LoadIndex(bin, txt)
	if err != nil {
		return nil, err
	}
	return &PureGoSearchEngine{index: idx}, nil
}

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
