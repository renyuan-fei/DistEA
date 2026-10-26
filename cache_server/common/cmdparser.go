package common

import (
	"flag"
	"fmt"
	"strings"

	log "github.com/sirupsen/logrus"
)

// Parser parses command-line flags and the YAML config file, then merges them
// into the global Config (CLI flags take precedence over YAML values).
func Parser() {
	configPath := flag.String("config", "./conf/static_cache.yaml", "Path to YAML config file")
	rpcPort := flag.Int("rpcport", 0, "gRPC listen port (overrides config file)")
	dataset := flag.String("dataset", "", "Dataset path (legacy LeapGNN mode)")
	cacheGroup := flag.String("cachegroup", "", "Comma-separated host:port entries for all ranks")
	cacheType := flag.String("cachetype", "", "Cache type: static | p3")
	partitionType := flag.String("partition_type", "", "Partition algorithm name (e.g. metis)")
	multiFeatFile := flag.Bool("multi_feat_file", false, "One feat-file per partition (legacy mode)")
	statistic := flag.Bool("statistic", false, "Enable hit-ratio statistics")

	distEARank := flag.Int("distea_rank", -1, "Explicit rank for this process (-1 = auto from IP)")
	partitionDir := flag.String("partition_dir", "", "Dir containing {rank}.npy partition files")
	embDir := flag.String("emb_dir", "", "Dir containing emb_{rank}.npy embedding files")
	bindAddr := flag.String("bind_addr", "", "gRPC bind address ('' = auto-detect, '0.0.0.0' = all)")
	readyFile := flag.String("ready_file", "", "File to create when server is ready")

	flag.Parse()

	yamlparser(*configPath)

	if *rpcPort != 0 {
		Config.RpcPort = *rpcPort
	}
	if *dataset != "" {
		Config.Dataset = *dataset
	}
	if *cacheGroup != "" {
		Config.CacheGroup = *cacheGroup
	}
	if *cacheType != "" {
		Config.CacheType = *cacheType
	}
	if *partitionType != "" {
		Config.PartitionType = *partitionType
	}
	// Boolean flags: CLI false is indistinguishable from "not set", so only
	// override when the flag is explicitly true.
	if *multiFeatFile {
		Config.MultiFeatFile = true
	}
	if *statistic {
		Config.Statistic = true
	}

	// -1 and "" are valid sentinels meaning "disabled", so always apply these fields.
	Config.DistEARank = *distEARank
	if *partitionDir != "" {
		Config.PartitionDir = *partitionDir
	}
	if *embDir != "" {
		Config.EmbDir = *embDir
	}
	if *bindAddr != "" {
		Config.BindAddr = *bindAddr
	}
	if *readyFile != "" {
		Config.ReadyFile = *readyFile
	}

	entries := strings.Split(Config.CacheGroup, ",")
	Config.Partition = len(entries)
	Config.CacheGroup = strings.ReplaceAll(Config.CacheGroup, " ", "")

	fmt.Println("Config.Dataset:       ", Config.Dataset)
	fmt.Println("Config.Partition:     ", Config.Partition)
	fmt.Println("Config.CacheType:     ", Config.CacheType)
	fmt.Println("Config.CacheGroup:    ", Config.CacheGroup)
	fmt.Println("Config.PartitionType: ", Config.PartitionType)
	fmt.Println("Config.MultiFeatFile: ", Config.MultiFeatFile)
	fmt.Println("Config.RpcPort:       ", Config.RpcPort)
	fmt.Println("Config.Statistic:     ", Config.Statistic)
	fmt.Println("Config.DistEARank:    ", Config.DistEARank)
	fmt.Println("Config.PartitionDir:  ", Config.PartitionDir)
	fmt.Println("Config.EmbDir:        ", Config.EmbDir)
	fmt.Println("Config.BindAddr:      ", Config.BindAddr)
	fmt.Println("Config.ReadyFile:     ", Config.ReadyFile)

	if Config.Statistic {
		log.Warn("Statistic mode enabled — training-time measurements will be inaccurate.")
	}
}
