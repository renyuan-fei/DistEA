package services

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"
	"unsafe"

	"distea/common"
	"distea/peerclient"
	"distea/rpc/cache"

	gonpy "github.com/kshedden/gonpy"
	npyio "github.com/sbinet/npyio"
	log "github.com/sirupsen/logrus"
)

// StaticCache loads all node embeddings into memory at startup and serves
// feature lookups locally or by forwarding to peer servers.
type StaticCache struct {
	sync.RWMutex
	data              map[int64][]float32 // node ID → feature slice
	cachedCount       int64               // number of cached nodes
	featureDim        int64               // feature dimension
	requestCount      int64               // total client requests (statistic mode)
	localHitCount     int64               // local-cache hits (statistic mode)
	localGatherTimes  []float32           // per-batch local gather latency (ms)
	remoteGatherTimes []float32           // per-batch remote gather latency (ms)
	chunkSize         int64               // max byte chunk for stream transfer
	retFeats          []float32           // pre-allocated return buffer (statistic mode)
	remoteRetFeats    []float32           // pre-allocated remote fetch buffer
}

// newStaticCache constructs and populates a StaticCache from partition .npy files.
func newStaticCache(dc *DistCache) CacheManager {
	sc := &StaticCache{data: make(map[int64][]float32)}

	var gnid []int64

	if common.Config.EmbDir != "" {
		// DistEA mode: node IDs from {EmbDir}/ids_{rank}.npy, embeddings from
		// {EmbDir}/emb_{rank}.npy. NOT {PartitionDir}/{rank}.npy -- that file
		// only lists KG1 entities (partition.py's own output), whereas
		// push_final_embeddings/GrpcEmbeddingCache.push() writes
		// local_kg1+local_kg2 concatenated into emb_{rank}.npy. Zipping the
		// KG1-only id list positionally against the longer KG1+KG2 embedding
		// array silently dropped every KG2 row from the cache (found via the
		// single-node grpc roundtrip test: KG2-entity queries came back as
		// zero vectors instead of their real embedding).
		if common.Config.PartitionDir == "" {
			log.Fatal("[static_cache.go] EmbDir set but PartitionDir empty — both required in DistEA mode")
		}
		nidPath := fmt.Sprintf("%s/ids_%d.npy", common.Config.EmbDir, dc.PartIdx)
		nf, err := os.Open(nidPath)
		if err != nil {
			log.Fatalf("[static_cache.go] open %q: %v", nidPath, err)
		}
		if err := npyio.Read(nf, &gnid); err != nil {
			nf.Close()
			log.Fatalf("[static_cache.go] read %q: %v", nidPath, err)
		}
		nf.Close()
		log.Infof("[static_cache.go] DistEA: loaded %d node IDs from %q", len(gnid), nidPath)

		embPath := fmt.Sprintf("%s/emb_%d.npy", common.Config.EmbDir, dc.PartIdx)

		hf, err := os.Open(embPath)
		if err != nil {
			log.Fatalf("[static_cache.go] open %q: %v", embPath, err)
		}
		hr, err := npyio.NewReader(hf)
		if err != nil {
			hf.Close()
			log.Fatalf("[static_cache.go] npyio.NewReader %q: %v", embPath, err)
		}
		shape := hr.Header.Descr.Shape // [N, feat_dim]
		hf.Close()
		log.Infof("[static_cache.go] DistEA embedding shape: %v", shape)

		if len(gnid) != shape[0] {
			log.Fatalf("[static_cache.go] id/embedding row count mismatch: "+
				"%d ids in %q vs %d rows in %q", len(gnid), nidPath, shape[0], embPath)
		}

		rf, err := gonpy.NewFileReader(embPath)
		if err != nil {
			log.Fatalf("[static_cache.go] gonpy.NewFileReader %q: %v", embPath, err)
		}
		features, err := rf.GetFloat32()
		if err != nil {
			log.Fatalf("[static_cache.go] GetFloat32 %q: %v", embPath, err)
		}

		start := time.Now()
		dim := int64(shape[1])
		for i, nid := range gnid {
			lo := int64(i) * dim
			hi := lo + dim
			sc.put(nid, features[lo:hi])
		}
		sc.featureDim = dim
		log.Infof("[static_cache.go] DistEA: cached %d embeddings (dim=%d) in %v",
			len(gnid), dim, time.Since(start))

	} else {
		// Legacy mode: dataset layout produced by LeapGNN partition scripts.
		nidPath := fmt.Sprintf("%s/dist_True/%d_%s/%d.npy",
			common.Config.Dataset, common.Config.Partition, common.Config.PartitionType, dc.PartIdx)
		nf, err := os.Open(nidPath)
		if err != nil {
			log.Fatalf("[static_cache.go] open %q: %v", nidPath, err)
		}
		if err := npyio.Read(nf, &gnid); err != nil {
			nf.Close()
			log.Fatalf("[static_cache.go] read %q: %v", nidPath, err)
		}
		nf.Close()
		log.Infof("[static_cache.go] legacy: %d node IDs for partition %q", len(gnid), dc.CurAddr)

		if common.Config.MultiFeatFile {
			featPath := fmt.Sprintf("%s/feat_%s/feat%d.npy",
				common.Config.Dataset, common.Config.PartitionType, dc.PartIdx)
			shape, features := readNpy(featPath)
			start := time.Now()
			dim := int64(shape[1])
			for i, nid := range gnid {
				lo := int64(i) * dim
				sc.put(nid, features[lo:lo+dim])
			}
			log.Infof("[static_cache.go] legacy multi-file: cached %d nodes (dim=%d) in %v",
				len(gnid), dim, time.Since(start))
			sc.featureDim = dim
		} else {
			featPath := fmt.Sprintf("%s/feat.npy", common.Config.Dataset)
			shape, features := readNpy(featPath)
			start := time.Now()
			dim := int64(shape[1])
			for _, nid := range gnid {
				lo := nid * dim
				sc.put(nid, features[lo:lo+dim])
			}
			log.Infof("[static_cache.go] legacy single-file: cached %d nodes (dim=%d) in %v",
				len(gnid), dim, time.Since(start))
			sc.featureDim = dim
		}
	}

	sc.chunkSize = getMaxNumDivisibleByXYAnd1024(sc.featureDim, 4)
	log.Infof("[static_cache.go] stream chunk size: %d bytes", sc.chunkSize)

	// Pre-allocate return buffers to avoid allocations on the hot path.
	const preallocNodes = 8_000_000
	sc.retFeats = make([]float32, preallocNodes*int(sc.featureDim))
	sc.remoteRetFeats = make([]float32, preallocNodes*int(sc.featureDim))
	log.Info("[static_cache.go] return buffers pre-allocated")

	return sc
}

// readNpy opens a .npy file and returns its shape and float32 data.
func readNpy(path string) (shape []int, data []float32) {
	hf, err := os.Open(path)
	if err != nil {
		log.Fatalf("[static_cache.go] open %q: %v", path, err)
	}
	hr, err := npyio.NewReader(hf)
	if err != nil {
		hf.Close()
		log.Fatalf("[static_cache.go] npyio.NewReader %q: %v", path, err)
	}
	shape = hr.Header.Descr.Shape
	hf.Close()

	rf, err := gonpy.NewFileReader(path)
	if err != nil {
		log.Fatalf("[static_cache.go] gonpy.NewFileReader %q: %v", path, err)
	}
	data, err = rf.GetFloat32()
	if err != nil {
		log.Fatalf("[static_cache.go] GetFloat32 %q: %v", path, err)
	}
	return
}

// put writes one node's feature slice into the cache under a write lock.
func (sc *StaticCache) put(nid int64, feat []float32) error {
	sc.Lock()
	defer sc.Unlock()
	sc.data[nid] = feat
	sc.cachedCount++
	return nil
}

// Put stores a feature slice for the given node ID.
func (sc *StaticCache) Put(nid int64, feat []float32) error {
	return sc.put(nid, feat)
}

// Get returns serialised float32 features for the requested node IDs.
// Features from remote partitions are fetched via gRPC peer calls.
func (sc *StaticCache) Get(ids []int64) ([]byte, error) {
	if !common.Config.Statistic {
		sc.RLock()
		defer sc.RUnlock()
		return sc.gatherFeatures(ids, make([]float32, len(ids)*int(sc.featureDim)))
	}

	sc.Lock()
	defer sc.Unlock()
	sc.requestCount += int64(len(ids))
	ret, err := sc.gatherFeaturesStatistic(ids, sc.retFeats)
	if err != nil {
		return nil, err
	}
	return encodeUnsafe(ret[:len(ids)*int(sc.featureDim)]), nil
}

// gatherFeatures collects local and remote features into dst without statistics.
func (sc *StaticCache) gatherFeatures(ids []int64, dst []float32) ([]byte, error) {
	// Partition request IDs by owning server.
	pid2ids := make(map[int64][]int64)
	gnid2ret := make(map[int64]int64)
	for retID, nid := range ids {
		pid := DCRuntime.Nid2Pid[nid]
		pid2ids[pid] = append(pid2ids[pid], nid)
		gnid2ret[nid] = int64(retID)
	}

	// Copy locally cached features.
	localPID := DCRuntime.PartIdx
	dim := sc.featureDim
	for _, nid := range pid2ids[localPID] {
		retID := gnid2ret[nid]
		copy(dst[retID*dim:(retID+1)*dim], sc.data[nid])
	}

	// Fetch remote features from peer servers in ring order.
	total := int64(len(DCRuntime.IPSlice))
	cur := localPID
	for range total - 1 {
		cur = (cur + 1) % total
		remoteAddr := DCRuntime.IPSlice[cur]
		ctx, cancel := context.WithTimeout(context.Background(), time.Hour)
		ret, err := peerclient.GrpcClients[remoteAddr].DCSubmit(ctx, &cache.DCRequest{
			Type: cache.OpType_get_features_by_peer_server,
			Ids:  pid2ids[cur],
		})
		cancel()
		if err != nil {
			return nil, fmt.Errorf("[static_cache.go] peer get from %q: %w", remoteAddr, err)
		}
		fetched := ret.GetRfeatures()
		for j, nid := range pid2ids[cur] {
			retID := gnid2ret[nid]
			srcLo := int64(j) * dim
			copy(dst[retID*dim:(retID+1)*dim], fetched[srcLo:srcLo+dim])
		}
	}
	return encodeUnsafe(dst), nil
}

// gatherFeaturesStatistic is the statistics-enabled variant that writes into the
// pre-allocated sc.retFeats buffer and records timing.
func (sc *StaticCache) gatherFeaturesStatistic(ids []int64, dst []float32) ([]float32, error) {
	pid2ids := make(map[int64][]int64)
	gnid2ret := make(map[int64]int64)
	for retID, nid := range ids {
		pid := DCRuntime.Nid2Pid[nid]
		pid2ids[pid] = append(pid2ids[pid], nid)
		gnid2ret[nid] = int64(retID)
	}

	localPID := DCRuntime.PartIdx
	dim := sc.featureDim

	stLocal := time.Now()
	sc.localHitCount += int64(len(pid2ids[localPID]))
	for _, nid := range pid2ids[localPID] {
		retID := gnid2ret[nid]
		copy(dst[retID*dim:(retID+1)*dim], sc.data[nid])
	}
	sc.localGatherTimes = append(sc.localGatherTimes, float32(time.Since(stLocal)/time.Millisecond))

	stRemote := time.Now()
	total := int64(len(DCRuntime.IPSlice))
	cur := localPID
	for range total - 1 {
		cur = (cur + 1) % total
		remoteAddr := DCRuntime.IPSlice[cur]
		ctx, cancel := context.WithTimeout(context.Background(), time.Hour)
		ret, err := peerclient.GrpcClients[remoteAddr].DCSubmit(ctx, &cache.DCRequest{
			Type: cache.OpType_get_features_by_peer_server,
			Ids:  pid2ids[cur],
		})
		cancel()
		if err != nil {
			return nil, fmt.Errorf("[static_cache.go] peer get from %q: %w", remoteAddr, err)
		}
		fetched := ret.GetRfeatures()
		for j, nid := range pid2ids[cur] {
			retID := gnid2ret[nid]
			srcLo := int64(j) * dim
			copy(dst[retID*dim:(retID+1)*dim], fetched[srcLo:srcLo+dim])
		}
	}
	sc.remoteGatherTimes = append(sc.remoteGatherTimes, float32(time.Since(stRemote)/time.Millisecond))
	return dst, nil
}

// FastGet is the stream-friendly variant — IDs arrive pre-sorted by partition
// (serids) and seplen carries the cumulative split points.
func (sc *StaticCache) FastGet(serids []int64, seplen []int64) ([]byte, error) {
	if !common.Config.Statistic {
		sc.RLock()
		defer sc.RUnlock()
		dst := make([]float32, len(serids)*int(sc.featureDim))
		return sc.fastGather(serids, seplen, dst)
	}

	sc.Lock()
	defer sc.Unlock()
	sc.requestCount += int64(len(serids))

	stRemote := time.Now()

	// Build pid→ids from the pre-sorted seplen layout.
	pid2ids := make(map[int64][]int64)
	for idx, id := range seplen {
		if idx == 0 {
			pid2ids[int64(idx)] = serids[:id]
		} else {
			pid2ids[int64(idx)] = serids[seplen[idx-1]:id]
		}
	}

	localPID := DCRuntime.PartIdx
	dim := sc.featureDim

	stLocal := time.Now()
	sc.localHitCount += int64(len(pid2ids[localPID]))
	startRow := int64(0)
	if localPID > 0 {
		startRow = seplen[localPID-1]
	}
	retRow := startRow
	for _, nid := range pid2ids[localPID] {
		copy(sc.retFeats[retRow*dim:(retRow+1)*dim], sc.data[nid])
		retRow++
	}
	sc.localGatherTimes = append(sc.localGatherTimes, float32(time.Since(stLocal)/time.Millisecond))

	total := int64(len(DCRuntime.IPSlice))
	cur := localPID
	for range total - 1 {
		cur = (cur + 1) % total
		remoteAddr := DCRuntime.IPSlice[cur]
		ctx, cancel := context.WithTimeout(context.Background(), time.Hour)
		ret, err := peerclient.GrpcClients[remoteAddr].DCSubmit(ctx, &cache.DCRequest{
			Type: cache.OpType_get_features_by_peer_server,
			Ids:  pid2ids[cur],
		})
		cancel()
		if err != nil {
			return nil, fmt.Errorf("[static_cache.go] peer get from %q: %w", remoteAddr, err)
		}
		dstLo, dstHi := int64(0)*dim, seplen[0]*dim
		if cur > 0 {
			dstLo = seplen[cur-1] * dim
			dstHi = seplen[cur] * dim
		}
		copy(sc.retFeats[dstLo:dstHi], ret.GetRfeatures())
	}
	sc.remoteGatherTimes = append(sc.remoteGatherTimes, float32(time.Since(stRemote)/time.Millisecond))
	return encodeUnsafe(sc.retFeats[:len(serids)*int(dim)]), nil
}

// fastGather is the non-statistic path for FastGet.
func (sc *StaticCache) fastGather(serids []int64, seplen []int64, dst []float32) ([]byte, error) {
	pid2ids := make(map[int64][]int64)
	gnid2ret := make(map[int64]int64)
	for retID, nid := range serids {
		pid := DCRuntime.Nid2Pid[nid]
		pid2ids[pid] = append(pid2ids[pid], nid)
		gnid2ret[nid] = int64(retID)
	}
	localPID := DCRuntime.PartIdx
	dim := sc.featureDim
	for _, nid := range pid2ids[localPID] {
		retID := gnid2ret[nid]
		copy(dst[retID*dim:(retID+1)*dim], sc.data[nid])
	}
	total := int64(len(DCRuntime.IPSlice))
	cur := localPID
	for range total - 1 {
		cur = (cur + 1) % total
		remoteAddr := DCRuntime.IPSlice[cur]
		ctx, cancel := context.WithTimeout(context.Background(), time.Hour)
		ret, err := peerclient.GrpcClients[remoteAddr].DCSubmit(ctx, &cache.DCRequest{
			Type: cache.OpType_get_features_by_peer_server,
			Ids:  pid2ids[cur],
		})
		cancel()
		if err != nil {
			return nil, fmt.Errorf("[static_cache.go] peer get from %q: %w", remoteAddr, err)
		}
		fetched := ret.GetRfeatures()
		for j, nid := range pid2ids[cur] {
			retID := gnid2ret[nid]
			srcLo := int64(j) * dim
			copy(dst[retID*dim:(retID+1)*dim], fetched[srcLo:srcLo+dim])
		}
	}
	return encodeUnsafe(dst), nil
}

// PeerGet is called by remote peers requesting this server's locally cached features.
func (sc *StaticCache) PeerGet(ids []int64) ([]float32, error) {
	dim := sc.featureDim
	ret := make([]float32, int64(len(ids))*dim)
	for i, nid := range ids {
		feat, ok := sc.data[nid]
		if !ok {
			return nil, fmt.Errorf("[static_cache.go] node %d not in local cache", nid)
		}
		lo := int64(i) * dim
		copy(ret[lo:lo+dim], feat)
	}
	return ret, nil
}

func (sc *StaticCache) Type() string { return "static_cache" }

func (sc *StaticCache) FeatDim() int64 { return sc.featureDim }

func (sc *StaticCache) MaxChunkSize() int64 { return sc.chunkSize }

func (sc *StaticCache) Reset() {
	sc.requestCount = 0
	sc.localHitCount = 0
	sc.localGatherTimes = sc.localGatherTimes[:0]
	sc.remoteGatherTimes = sc.remoteGatherTimes[:0]
}

func (sc *StaticCache) CacheInfo() (int64, string, int64, int64, []float32, []float32) {
	return DCRuntime.PartIdx, DCRuntime.CurAddr,
		sc.requestCount, sc.localHitCount,
		sc.localGatherTimes, sc.remoteGatherTimes
}

// encodeUnsafe reinterprets a []float32 as []byte without copying.
// The caller must not modify fs while the returned slice is in use.
func encodeUnsafe(fs []float32) []byte {
	if len(fs) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&fs[0])), len(fs)*4)
}

// getMaxNumDivisibleByXYAnd1024 returns the largest integer ≤4 MiB
// that is evenly divisible by x, y, and 1024.
func getMaxNumDivisibleByXYAnd1024(x, y int64) int64 {
	const maxBytes = int64(4 * 1024 * 1024)
	for n := maxBytes; n > 0; n-- {
		if n%x == 0 && n%y == 0 && n%1024 == 0 {
			return n
		}
	}
	return -1
}
