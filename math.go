package clipsdk

import "math"

// dotProduct 计算两个向量的点积
func dotProduct(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// l2Normalize 对向量进行 L2 归一化
func l2Normalize(v []float32) []float32 {
	var sum float32
	for _, x := range v {
		sum += x * x
	}
	norm := float32(math.Sqrt(float64(sum)))
	if norm == 0 {
		return v
	}
	out := make([]float32, len(v))
	for i := range v {
		out[i] = v[i] / norm
	}
	return out
}
