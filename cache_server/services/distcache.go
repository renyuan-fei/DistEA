package services

import (
	"fmt"
	"net"
	"os"
	"strings"
	"time"

	"distea/common"

	npyio "github.com/sbinet/npyio"
	log "github.com/sirupsen/logrus"
)

// CacheManager is the interface every cache implementation must satisfy.
type CacheManager interface {
	Type() string
	Get([]int64) ([]byte, error)
	FastGet([]int64, []int64) ([]byte, error)
	Put(int64, []float32) error
	PeerGet([]int64) ([]float32, error) // called by peer servers; distinct from client Get for hit-ratio accounting
	FeatDim() int64
	CacheInfo() (int64, string, int64, int64, []float32, []float32) // (partIdx, curAddr, reqCount, localHitCount, localTimes, remoteTimes)
	MaxChunkSize() int64                                            // max byte chunk for stream transfer
	Reset()
}

// cacheInitFunc is the constructor signature for cache implementations.
type cacheInitFunc func(*DistCache) CacheManager

// DistCache holds server-wide routing state shared by all cache implementations.
type DistCache struct {
	Nid2Pid    map[int64]int64 // graph node ID → partition index
	PartIdx    int64           // this server's partition index
	CurAddr    string          // this server's address (IP or host:port)
	CacheType  string          // chosen cache implementation: "static" | "p3"
	CacheMgr   CacheManager    // the active cache implementation
	CacheGroup string          // comma-separated server addresses from config
	IPSlice    []string        // ordered list of server addresses
}

// registered cache constructors keyed by the CacheType config value.
var cacheRegistry = map[string]cacheInitFunc{
	"static": newStaticCache,
	"p3":     newP3Cache,
}

// initDistCache populates dc from the global config and loads partition maps.
func initDistCache(dc *DistCache) {
	ipSlice := strings.Split(dc.CacheGroup, ",")
	log.Infof("[distcache.go] cache group (%d entries): %v", len(ipSlice), ipSlice)
	dc.IPSlice = ipSlice

	dc.CurAddr = ""
	dc.PartIdx = -1

	if common.Config.DistEARank >= 0 {
		// DistEA mode: rank provided explicitly — skip IP detection.
		// Supports single-host multi-process where all ranks share the same IP.
		dc.PartIdx = int64(common.Config.DistEARank)
		if int(dc.PartIdx) < len(ipSlice) {
			dc.CurAddr = ipSlice[dc.PartIdx]
		}
		log.Infof("[distcache.go] DistEA explicit rank %d, address %q", dc.PartIdx, dc.CurAddr)
	} else {
		// Legacy mode: detect local IPv4 and match against CacheGroup entries.
		addrs, err := net.InterfaceAddrs()
		if err != nil {
			log.Fatalf("[distcache.go] cannot enumerate interface addresses: %v", err)
		}
		log.Info("[distcache.go] local network addrs:", addrs)

		for _, address := range addrs {
			if ipnet, ok := address.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
				if ipnet.IP.To4() != nil {
					dc.CurAddr = ipnet.IP.String()
					if strings.HasPrefix(dc.CurAddr, "12.") {
						break
					}
				}
			}
		}

		for idx, addr := range ipSlice {
			// Strip port suffix before comparing so "host:port" entries also match.
			host := addr
			if i := strings.LastIndex(addr, ":"); i >= 0 {
				host = addr[:i]
			}
			if host == dc.CurAddr {
				dc.PartIdx = int64(idx)
				break
			}
		}

		if dc.PartIdx == -1 {
			log.Errorf("[distcache.go] partition resolution failed — local IP %q not found in cache_group %q",
				dc.CurAddr, dc.CacheGroup)
			os.Exit(1)
		}
	}

	log.Infof("[distcache.go] serving graph partition %d", dc.PartIdx)

	start := time.Now()
	dc.Nid2Pid = make(map[int64]int64)
	for pid := 0; pid < len(ipSlice); pid++ {
		var nidPath string
		if common.Config.EmbDir != "" {
			// DistEA mode: {PartitionDir}/{rank}.npy only lists KG1 entities
			// (partition.py's own output) -- push_final_embeddings/
			// GrpcEmbeddingCache.push() writes local_kg1+local_kg2 concatenated
			// into emb_{rank}.npy, with the matching full ID list in
			// ids_{rank}.npy. Routing on the KG1-only file silently drops every
			// KG2 entity from Nid2Pid (map lookup for a missing key returns the
			// int64 zero value, misrouting KG2 queries to partition 0 instead of
			// erroring) -- read the authoritative ids_{rank}.npy instead.
			nidPath = fmt.Sprintf("%s/ids_%d.npy", common.Config.EmbDir, pid)
		} else if common.Config.PartitionDir != "" {
			// DistEA mode without a pushed embedding cache yet (e.g. routing-only
			// startup): fall back to the KG1-only partition file.
			nidPath = fmt.Sprintf("%s/%d.npy", common.Config.PartitionDir, pid)
		} else {
			// Legacy mode: dataset layout produced by LeapGNN partition scripts.
			nidPath = fmt.Sprintf("%s/dist_True/%d_%s/%d.npy",
				common.Config.Dataset, common.Config.Partition, common.Config.PartitionType, pid)
		}

		f, err := os.Open(nidPath)
		if err != nil {
			log.Fatalf("[distcache.go] open %q: %v", nidPath, err)
		}
		var gnid []int64
		if err := npyio.Read(f, &gnid); err != nil {
			f.Close()
			log.Fatalf("[distcache.go] read %q: %v", nidPath, err)
		}
		f.Close()

		for _, nid := range gnid {
			dc.Nid2Pid[nid] = int64(pid)
		}
	}
	log.Infof("[distcache.go] Nid2Pid built in %v (%d entries)", time.Since(start), len(dc.Nid2Pid))

	init, ok := cacheRegistry[dc.CacheType]
	if !ok {
		log.Fatalf("[distcache.go] unknown cache type %q (valid: static, p3)", dc.CacheType)
	}
	dc.CacheMgr = init(dc)
	log.Infof("[distcache.go] cache type %q ready", dc.CacheMgr.Type())
}
