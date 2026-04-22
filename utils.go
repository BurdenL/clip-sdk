package clipsdk

import "sort"

func sortMatches(m []Match) {
	sort.Slice(m, func(i, j int) bool {
		return m[i].Similarity > m[j].Similarity
	})
}
