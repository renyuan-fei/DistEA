package common

// ServerConfig holds all runtime configuration for the DistEA cache server.
// Fields use Go CamelCase; yaml tags match the YAML key names exactly.
type ServerConfig struct {
	RpcPort       int    `yaml:"rpcport"`         // gRPC listen port
	Debug         bool   `yaml:"debug"`           // verbose debug logging
	Dataset       string `yaml:"dataset"`         // raw dataset path (legacy LeapGNN mode)
	CacheType     string `yaml:"cache_type"`      // cache implementation: "static" | "p3"
	CacheGroup    string `yaml:"cache_group"`     // comma-separated "host:port" or "host" entries
	Partition     int    `yaml:"partition"`       // number of partitions (derived from CacheGroup)
	Statistic     bool   `yaml:"statistic"`       // enable hit-ratio statistics (slows throughput)
	MultiFeatFile bool   `yaml:"multi_feat_file"` // one feat-file per partition vs single file
	PartitionType string `yaml:"partition_type"`  // partition algorithm name (legacy: "metis")

	PartitionDir string `yaml:"partition_dir"` // dir containing {rank}.npy partition files (empty = legacy mode)
	EmbDir       string `yaml:"emb_dir"`       // dir containing emb_{rank}.npy embedding files
	DistEARank   int    `yaml:"distearank"`    // explicit rank (-1 = auto from local IP)
	BindAddr     string `yaml:"bindaddr"`      // gRPC bind address ("" = auto-detect)
	ReadyFile    string `yaml:"readyfile"`     // file created once server is fully initialised
}
