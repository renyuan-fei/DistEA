package main

import (
	"fmt"
	"os"

	"distea/common"
	"distea/peerclient"
	"distea/rpc"
	"distea/services"

	log "github.com/sirupsen/logrus"
)

func init() {
	common.Parser()
	services.Start()
	rpc.Start()
	peerclient.Build()
}

// removeStaleReadyFile deletes the ready-signal file if it already exists,
// preventing a stale file from a previous run from being mistaken as a signal.
func removeStaleReadyFile(path string) error {
	if _, err := os.Stat(path); err == nil {
		if err := os.Remove(path); err != nil {
			return fmt.Errorf("remove stale ready file %q: %w", path, err)
		}
		fmt.Printf("[server.go] removed stale ready file: %s\n", path)
	}
	return nil
}

func main() {
	readyFile := common.Config.ReadyFile
	if readyFile == "" {
		readyFile = "server_done.txt"
	}

	if err := removeStaleReadyFile(readyFile); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	f, err := os.Create(readyFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "[server.go] create ready file %q: %v\n", readyFile, err)
		os.Exit(1)
	}
	f.Close()
	fmt.Printf("[server.go] ready file created: %s\n", readyFile)

	log.Info("[server.go] gRPC cache server ready")

	// Block forever; the server goroutines run in the background.
	<-make(chan struct{})
}
