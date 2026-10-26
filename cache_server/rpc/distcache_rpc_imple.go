package rpc

import (
	"distea/common"
	"distea/rpc/cache"
	"distea/services"

	log "github.com/sirupsen/logrus"
)

// opGetFeaturesByClient handles a client request for node features, delegating
// to the local CacheManager which fetches remote shards as needed.
func opGetFeaturesByClient(request *cache.DCRequest) (*cache.DCReply, error) {
	var reply cache.DCReply
	features, err := services.DCRuntime.CacheMgr.Get(request.Ids)
	if err != nil {
		log.Fatalf("[distcache_rpc_imple.go] Get error: %v", err)
	}
	reply.Features = features
	return &reply, nil
}

// opReset resets hit-ratio statistics on the active cache implementation.
func opReset(_ *cache.DCRequest) (*cache.DCReply, error) {
	services.DCRuntime.CacheMgr.Reset()
	return &cache.DCReply{}, nil
}

// opGetStreamFeatures handles a streaming feature request.
// StaticCache uses FastGet (presorted IDs); P3Cache falls back to Get.
func opGetStreamFeatures(request *cache.DCRequest) (*cache.DCReply, error) {
	var (
		features []byte
		err      error
	)
	if services.DCRuntime.CacheMgr.Type() == "static_cache" {
		features, err = services.DCRuntime.CacheMgr.FastGet(request.Serids, request.Seplen)
	} else {
		features, err = services.DCRuntime.CacheMgr.Get(request.Ids)
	}
	if err != nil {
		log.Fatalf("[distcache_rpc_imple.go] FastGet/Get error: %v", err)
	}
	return &cache.DCReply{Features: features}, nil
}

// opPeerGet handles a peer-server request for locally cached features.
func opPeerGet(request *cache.DCRequest) (*cache.DCReply, error) {
	features, err := services.DCRuntime.CacheMgr.PeerGet(request.Ids)
	if err != nil {
		log.Fatalf("[distcache_rpc_imple.go] PeerGet error: %v", err)
	}
	return &cache.DCReply{Rfeatures: features}, nil
}

// opGetFeatDim returns the feature dimension of the active cache.
func opGetFeatDim(_ *cache.DCRequest) (*cache.DCReply, error) {
	return &cache.DCReply{Featdim: services.DCRuntime.CacheMgr.FeatDim()}, nil
}

// opGetCacheInfo returns per-partition statistics.
func opGetCacheInfo(_ *cache.DCRequest) (*cache.DCReply, error) {
	partIdx, curAddr, reqCount, localHits, localTimes, remoteTimes :=
		services.DCRuntime.CacheMgr.CacheInfo()

	var totalLocal, totalRemote float32
	for _, t := range localTimes {
		totalLocal += t
	}
	for _, t := range remoteTimes {
		totalRemote += t
	}

	return &cache.DCReply{
		Partidx:               partIdx,
		Curaddr:               curAddr,
		Requestnum:            reqCount,
		Localhitnum:           localHits,
		LocalFeatsGatherTime:  totalLocal,
		RemoteFeatsGatherTime: totalRemote,
	}, nil
}

// opGetStatistic returns whether statistic mode is enabled.
func opGetStatistic(_ *cache.DCRequest) (*cache.DCReply, error) {
	return &cache.DCReply{Statistic: common.Config.Statistic}, nil
}
