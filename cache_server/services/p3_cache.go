package services

import (
	"fmt"
	"os"
	"sync"
	"time"

	"distea/common"

	npyio "github.com/sbinet/npyio"
	log "github.com/sirupsen/logrus"
)

// P3Cache implements the P3 (Pipeline-Parallel-Partition) cache strategy.
// Each server holds a vertical slice of the global feature matrix.
type P3Cache struct {
	sync.RWMutex
	data              map[int64][]float32 // node ID → partial feature slice
	cachedCount       int64               // number of cached nodes
	featureDim        int64               // full feature dimension (sum across all partitions)
	requestCount      int64               // total client requests (statistic mode)
	localHitCount     int64               // local-cache hits (statistic mode)
	localGatherTimes  []float32           // per-batch local gather latency (ms)
	remoteGatherTimes []float32           // per-batch remote gather latency (ms)
	chunkSize         int64               // max byte chunk for stream transfer
	offset            int64               // column offset of this server's slice
	featLen           int64               // length of this server's feature slice per node
}

// newP3Cache constructs and populates a P3Cache from partition feature files.
func newP3Cache(dc *DistCache) CacheManager {
	pc := &P3Cache{data: make(map[int64][]float32)}

	curPID := DCRuntime.PartIdx
	totalParts := int64(len(DCRuntime.IPSlice))

	if common.Config.MultiFeatFile {
		// One .npy file per partition: each server stores its own column slice.
		featPath := fmt.Sprintf("%s/feat_p3/p3_feat%d.npy", common.Config.Dataset, dc.PartIdx)
		shape, features := readNpy(featPath)
		fullDim := int64(shape[1])
		pc.featLen = fullDim / totalParts
		pc.offset = pc.featLen * curPID

		start := time.Now()
		for nid := int64(0); nid < int64(shape[0]); nid++ {
			lo := nid * fullDim
			hi := lo + fullDim
			pc.put(nid, features[lo:hi])
		}
		log.Infof("[p3_cache.go] multi-file: cached %d nodes in %v", shape[0], time.Since(start))

		// Sum feature dimensions across all partitions to get the full output dim.
		pc.featureDim = 0
		for i := 0; i < common.Config.Partition; i++ {
			pPath := fmt.Sprintf("%s/feat_p3/p3_feat%d.npy", common.Config.Dataset, i)
			pf, err := os.Open(pPath)
			if err != nil {
				log.Fatalf("[p3_cache.go] open %q: %v", pPath, err)
			}
			pr, err := npyio.NewReader(pf)
			if err != nil {
				pf.Close()
				log.Fatalf("[p3_cache.go] npyio.NewReader %q: %v", pPath, err)
			}
			ps := pr.Header.Descr.Shape
			pf.Close()
			pc.featureDim += int64(ps[1])
		}
	} else {
		// Single feat.npy: each server owns a contiguous column slice.
		featPath := fmt.Sprintf("%s/feat.npy", common.Config.Dataset)
		shape, features := readNpy(featPath)
		fullDim := int64(shape[1])

		if curPID == totalParts-1 {
			pc.featLen = fullDim/totalParts + fullDim%totalParts // last partition gets remainder
		} else {
			pc.featLen = fullDim / totalParts
		}
		pc.offset = (fullDim / totalParts) * curPID

		start := time.Now()
		for nid := int64(0); nid < int64(shape[0]); nid++ {
			lo := nid*fullDim + pc.offset
			pc.put(nid, features[lo:lo+pc.featLen])
		}
		log.Infof("[p3_cache.go] single-file: cached %d nodes in %v", shape[0], time.Since(start))
		pc.featureDim = fullDim
	}

	pc.chunkSize = getMaxNumDivisibleByXYAnd1024(pc.featLen, 4)
	log.Infof("[p3_cache.go] stream chunk size: %d bytes", pc.chunkSize)
	return pc
}

// put writes one node's partial feature slice under a write lock.
func (pc *P3Cache) put(nid int64, feat []float32) error {
	pc.Lock()
	defer pc.Unlock()
	pc.data[nid] = feat
	pc.cachedCount++
	return nil
}

// Put stores a partial feature slice for the given node ID.
func (pc *P3Cache) Put(nid int64, feat []float32) error {
	return pc.put(nid, feat)
}

// Get returns serialized float32 partial features for the requested node IDs.
// P3Cache serves only its own column slice; the caller assembles all slices.
func (pc *P3Cache) Get(ids []int64) ([]byte, error) {
	if !common.Config.Statistic {
		pc.RLock()
		defer pc.RUnlock()
		ret := make([]float32, int64(len(ids))*pc.featLen)
		for i, nid := range ids {
			lo := int64(i) * pc.featLen
			copy(ret[lo:lo+pc.featLen], pc.data[nid])
		}
		log.Infof("[p3_cache.go] gathered %d local features in this call", len(ids))
		return encodeUnsafe(ret), nil
	}

	pc.Lock()
	defer pc.Unlock()
	pc.requestCount += int64(len(ids))

	stLocal := time.Now()
	ret := make([]float32, int64(len(ids))*pc.featLen)
	for i, nid := range ids {
		lo := int64(i) * pc.featLen
		copy(ret[lo:lo+pc.featLen], pc.data[nid])
	}
	pc.localGatherTimes = append(pc.localGatherTimes, float32(time.Since(stLocal)/time.Millisecond))
	return encodeUnsafe(ret), nil
}

// FastGet is a no-op for P3Cache (layout does not support presorted IDs).
func (pc *P3Cache) FastGet(serids []int64, seplen []int64) ([]byte, error) {
	return nil, nil
}

// PeerGet returns this server's partial features for the given node IDs.
func (pc *P3Cache) PeerGet(ids []int64) ([]float32, error) {
	dim := pc.featureDim
	ret := make([]float32, int64(len(ids))*dim)
	st := time.Now()
	for i, nid := range ids {
		feat, ok := pc.data[nid]
		if !ok {
			return nil, fmt.Errorf("[p3_cache.go] node %d not in local cache", nid)
		}
		lo := int64(i) * dim
		copy(ret[lo:lo+dim], feat)
	}
	log.Infof("[p3_cache.go] PeerGet: gathered %d features in %v", len(ids), time.Since(st))
	return ret, nil
}

func (pc *P3Cache) Type() string { return "p3_cache" }

func (pc *P3Cache) FeatDim() int64 { return pc.featureDim }

func (pc *P3Cache) MaxChunkSize() int64 { return pc.chunkSize }

func (pc *P3Cache) Reset() {
	pc.requestCount = 0
	pc.localHitCount = 0
	pc.localGatherTimes = pc.localGatherTimes[:0]
	pc.remoteGatherTimes = pc.remoteGatherTimes[:0]
}

func (pc *P3Cache) CacheInfo() (int64, string, int64, int64, []float32, []float32) {
	return DCRuntime.PartIdx, DCRuntime.CurAddr,
		pc.requestCount, pc.localHitCount,
		pc.localGatherTimes, pc.remoteGatherTimes
}
