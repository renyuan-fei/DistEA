package rpc

import (
	"net"
	"strconv"
	"strings"

	"distea/common"

	log "github.com/sirupsen/logrus"
	"google.golang.org/grpc"
)

// Start launches the gRPC server in a background goroutine.
func Start() {
	go runGRPCServer()
}

func runGRPCServer() {
	var listenAddr string
	if common.Config.BindAddr != "" {
		// DistEA mode: explicit bind address (e.g. "0.0.0.0" or "127.0.0.1").
		listenAddr = common.Config.BindAddr + ":" + strconv.Itoa(common.Config.RpcPort)
	} else {
		// Legacy mode: auto-detect local IPv4.
		var localIP string
		addrs, err := net.InterfaceAddrs()
		if err != nil {
			log.Fatalf("[rpc.go] cannot enumerate interface addresses: %v", err)
		}
		for _, address := range addrs {
			if ipnet, ok := address.(*net.IPNet); ok && !ipnet.IP.IsLoopback() {
				if ipnet.IP.To4() != nil {
					localIP = ipnet.IP.String()
					if strings.HasPrefix(localIP, "12.") {
						break
					}
				}
			}
		}
		listenAddr = localIP + ":" + strconv.Itoa(common.Config.RpcPort)
	}

	lis, err := net.Listen("tcp", listenAddr)
	if err != nil {
		log.Fatalf("[rpc.go] listen %q: %v", listenAddr, err)
	}
	log.Infof("[rpc.go] gRPC server listening on %q", listenAddr)

	const msgLimit = 2000 * 1024 * 1024 // 2 GiB
	s := grpc.NewServer(
		grpc.MaxRecvMsgSize(msgLimit),
		grpc.MaxSendMsgSize(msgLimit),
	)
	Register(s)
	if err := s.Serve(lis); err != nil {
		log.Fatalf("[rpc.go] gRPC server error: %v", err)
	}
}
