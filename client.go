package clipsdk

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

func (c *Client) SearchImage(path string, topK int) ([]Match, error) {
	return c.engine.SearchTopKByFile(path, topK)
}

func (c *Client) Close() {
	c.engine.Close()
}
