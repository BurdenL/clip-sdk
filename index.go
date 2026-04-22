package clipsdk

import (
	"bufio"
	"encoding/binary"
	"os"
	"strings"
)

type ImageIndex struct {
	Names      []string
	Embeddings [][]float32
	Dim        int
}

type Match struct {
	Name       string
	Index      int
	Similarity float32
}

func LoadIndex(binPath, txtPath string) (*ImageIndex, error) {
	bf, err := os.Open(binPath)
	if err != nil {
		return nil, err
	}
	defer bf.Close()

	var count, dim uint32
	binary.Read(bf, binary.LittleEndian, &count)
	binary.Read(bf, binary.LittleEndian, &dim)

	embeddings := make([][]float32, count)
	for i := uint32(0); i < count; i++ {
		emb := make([]float32, dim)
		binary.Read(bf, binary.LittleEndian, emb)
		embeddings[i] = emb
	}

	tf, _ := os.Open(txtPath)
	defer tf.Close()

	var names []string
	scanner := bufio.NewScanner(tf)
	for scanner.Scan() {
		names = append(names, strings.TrimSpace(scanner.Text()))
	}

	return &ImageIndex{
		Names:      names,
		Embeddings: embeddings,
		Dim:        int(dim),
	}, nil
}
