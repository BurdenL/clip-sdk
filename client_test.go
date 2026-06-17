package clipsdk

import (
	"math"
	"testing"
)

func TestValidateEmbeddingFormat(t *testing.T) {
	dim := 512
	emb := make([]float32, dim)
	for i := 0; i < dim; i++ {
		emb[i] = 1.0 / float32(math.Sqrt(float64(dim)))
	}

	if err := ValidateEmbeddingFormat(emb, dim); err != nil {
		t.Fatalf("expected valid embedding, got error: %v", err)
	}

	if err := ValidateEmbeddingFormat(emb, dim+1); err == nil {
		t.Fatal("expected dim mismatch error, got nil")
	}

	nonNormalized := make([]float32, dim)
	for i := 0; i < dim; i++ {
		nonNormalized[i] = 1.0
	}
	if err := ValidateEmbeddingFormat(nonNormalized, dim); err == nil {
		t.Fatal("expected non-normalized embedding error, got nil")
	}

	zeroEmb := make([]float32, dim)
	if err := ValidateEmbeddingFormat(zeroEmb, dim); err == nil {
		t.Fatal("expected zero embedding error, got nil")
	}
}
