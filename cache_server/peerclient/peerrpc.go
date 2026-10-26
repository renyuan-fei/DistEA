package peerclient

import (
	"net"
	"strconv"
	"strings"

	"distea/common"
	"distea/rpc/cache"

	log "github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// GrpcClients maps each peer server address (as it appears in CacheGroup) to
// a gRPC client stub. The local server's address is excluded.
var GrpcClients map[string]cache.OperatorClient

// Build creates gRPC client connections to all peer servers (everyone except self).
func Build() {
	ipsSlice := strings.Split(common.Config.CacheGroup, ",")

	// Determine this server's address so we can skip connecting to ourselves.
	selfAddr := resolveSelfAddr(ipsSlice)
	log.Infof("[peerrpc.go] self-address: %q (skipping)", selfAddr)

	GrpcClients = make(map[string]cache.OperatorClient)
	for _, addr := range ipsSlice {
		if addr == selfAddr {
			continue
		}

		// If the addr already contains a port (host:port), use as-is;
		// otherwise append the configured RpcPort (legacy IP-only entries).
		dialAddr := addr
		if !strings.Contains(addr, ":") {
			dialAddr = addr + ":" + strconv.Itoa(common.Config.RpcPort)
		}

		log.Infof("[peerrpc.go] connecting to peer gRPC server at %q", dialAddr)
		const recvLimit = 2000 * 1024 * 1024 // 2 GiB
		conn, err := grpc.NewClient(
			dialAddr,
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(recvLimit)),
		)
		if err != nil {
			log.Fatalf("[peerrpc.go] create client for %q: %v", dialAddr, err)
		}
		// Key by the raw addr so it matches DCRuntime.IPSlice entries.
		GrpcClients[addr] = cache.NewOperatorClient(conn)
		log.Infof("[peerrpc.go] registered peer client %q", addr)
	}

	for k, v := range GrpcClients {
		log.Debugf("[peerrpc.go] GrpcClients[%q] = %v", k, v)
	}
}

// resolveSelfAddr returns the CacheGroup entry that belongs to this process.
// In DistEA mode the rank is given explicitly; in legacy mode we match against
// locally detected IPv4 addresses.
func resolveSelfAddr(ipsSlice []string) string {
	if common.Config.DistEARank >= 0 {
		if common.Config.DistEARank < len(ipsSlice) {
			return ipsSlice[common.Config.DistEARank]
		}
		return ""
	}

	// Legacy: detect local IPv4 and match against CacheGroup entries.
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		log.Fatalf("[peerrpc.go] cannot enumerate interface addresses: %v", err)
	}
	for _, address := range addrs {
		ipnet, ok := address.(*net.IPNet)
		if !ok || ipnet.IP.IsLoopback() || ipnet.IP.To4() == nil {
			continue
		}
		localIP := ipnet.IP.String()
		for _, addr := range ipsSlice {
			host := addr
			if i := strings.LastIndex(addr, ":"); i >= 0 {
				host = addr[:i]
			}
			if host == localIP {
				return addr
			}
		}
	}
	return ""
}
