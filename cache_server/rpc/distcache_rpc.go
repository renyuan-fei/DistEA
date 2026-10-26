package rpc

import (
	context "context"
	"fmt"
	"os"
	"strconv"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"distea/rpc/cache"
	"distea/services"

	log "github.com/sirupsen/logrus"
	grpc "google.golang.org/grpc"
)

const maxChunkBytes = 4 * 1024 * 1024 // 4 MiB

const shmFileBase = "/dev/shm/repgnn_shm"

var (
	shmCounter int
	shmMu      sync.Mutex
)

type dcrpcserver struct {
	cache.UnimplementedOperatorServer
	cache.UnimplementedOperatorFeaturesServer
}

// DCSubmit dispatches a unary RPC to the appropriate handler.
func (s *dcrpcserver) DCSubmit(ctx context.Context, request *cache.DCRequest) (*cache.DCReply, error) {
	var reply *cache.DCReply
	switch request.Type {
	case cache.OpType_get_features_by_peer_server:
		reply, _ = opPeerGet(request)
	case cache.OpType_get_features_by_client:
		reply, _ = opGetFeaturesByClient(request)
	case cache.OpType_get_feature_dim:
		reply, _ = opGetFeatDim(request)
	case cache.OpType_get_cache_info:
		reply, _ = opGetCacheInfo(request)
	case cache.OpType_get_statistic:
		reply, _ = opGetStatistic(request)
	case cache.OpType_reset:
		reply, _ = opReset(request)
		shmCounter = 0
		log.Info("[distcache_rpc.go] cache reset")
	}
	return reply, nil
}

// chunkBytes splits a byte slice into chunks of at most chunkSize bytes.
func chunkBytes(b []byte, chunkSize int) [][]byte {
	var chunks [][]byte
	for len(b) > 0 {
		n := chunkSize
		if len(b) < n {
			n = len(b)
		}
		chunks = append(chunks, b[:n])
		b = b[n:]
	}
	return chunks
}

// writeShmFiles writes reply chunks into /dev/shm files for shared-memory IPC.
func writeShmFiles(replyChunks [][]byte, chunkSize int) error {
	shmMu.Lock()
	defer shmMu.Unlock()
	for _, chunk := range replyChunks {
		filename := shmFileBase + strconv.Itoa(shmCounter)
		// Wait until the consumer has removed the previous file.
		for {
			if _, err := os.Stat(filename); err != nil {
				break
			}
			time.Sleep(time.Second)
		}
		fd, err := syscall.Open(filename, syscall.O_CREAT|syscall.O_RDWR, 0600)
		if err != nil {
			return fmt.Errorf("open shm %q: %w", filename, err)
		}
		shmSize := int64(len(chunk))
		if err := syscall.Ftruncate(fd, shmSize); err != nil {
			syscall.Close(fd)
			return fmt.Errorf("ftruncate shm %q: %w", filename, err)
		}
		addr, err := syscall.Mmap(fd, 0, int(shmSize), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
		syscall.Close(fd)
		if err != nil {
			return fmt.Errorf("mmap shm %q: %w", filename, err)
		}
		blockPtr := unsafe.Pointer(uintptr(unsafe.Pointer(&addr[0])))
		copy((*[maxChunkBytes]byte)(blockPtr)[:len(chunk)], chunk)
		if err := syscall.Munmap(addr); err != nil {
			return fmt.Errorf("munmap shm %q: %w", filename, err)
		}
		shmCounter++
	}
	return nil
}

// DCSubmitFeatures dispatches a server-streaming RPC, writing results to shared memory.
func (s *dcrpcserver) DCSubmitFeatures(request *cache.DCRequest, stream cache.OperatorFeatures_DCSubmitFeaturesServer) error {
	var reply *cache.DCReply
	switch request.Type {
	case cache.OpType_get_stream_features_by_client:
		reply, _ = opGetStreamFeatures(request)
	}
	chunkSize := int(services.DCRuntime.CacheMgr.MaxChunkSize())
	chunks := chunkBytes(reply.Features, chunkSize)
	go writeShmFiles(chunks, chunkSize) //nolint:errcheck
	return nil
}

// Register registers the gRPC service implementations with the server.
func Register(s *grpc.Server) {
	log.Info("[distcache_rpc.go] registering gRPC service handlers")
	cache.RegisterOperatorServer(s, &dcrpcserver{})
	cache.RegisterOperatorFeaturesServer(s, &dcrpcserver{})
}
