package clipsdk

import (
	"io"
)

type Config struct {
	ModelPath string
	IndexBin  string
	IndexTxt  string
	ORTLib    string
}

type Client struct {
	engine *CLIPSearchEngine
}

func NewClient(cfg Config) (*Client, error) {
	engine, err := NewCLIPSearchEngine(
		cfg.ModelPath,
		cfg.IndexBin,
		cfg.IndexTxt,
		cfg.ORTLib,
	)
	if err != nil {
		return nil, err
	}
	return &Client{engine: engine}, nil
}

// SearchImageByPath 接收图片路径，直接返回 Top-K 结果（线程安全）
func (c *Client) SearchImageByPath(path string, topK int) ([]Match, error) {
	return c.engine.SearchTopKByFile(path, topK)
}

// SearchImageByReader 接收图片流，直接返回 Top-K 结果（线程安全）
func (c *Client) SearchImageByReader(r io.Reader, topK int) ([]Match, error) {
	return c.engine.SearchTopKByReader(r, topK)
}

// SearchScopeByReader 接收图片流，直接返回相似度高于指定阈值的结果（线程安全）
func (c *Client) SearchScopeByReader(r io.Reader, scope float32) ([]Match, error) {
	return c.engine.SearchScopeByReader(r, scope)
}

// SearchScopeByFile 接收图片路径，直接返回相似度高于指定阈值的结果（线程安全）
func (c *Client) SearchScopeByFile(path string, scope float32) ([]Match, error) {
	return c.engine.SearchScopeByFile(path, scope)
}

func (c *Client) Close() {
	c.engine.Close()
}
