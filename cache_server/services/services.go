package services

import (
	"os/exec"
	"runtime"

	"distea/common"

	log "github.com/sirupsen/logrus"
)

// DCRuntime is the global DistCache instance shared by all request handlers.
var DCRuntime *DistCache

// Start initialises the cache service: loads partition data and starts the
// chosen CacheManager implementation.
func Start() {
	DCRuntime = &DistCache{
		CacheType:  common.Config.CacheType,
		CacheGroup: common.Config.CacheGroup,
	}
	initDistCache(DCRuntime)
	log.Info("[services.go] cache service started")
}

// Command runs an arbitrary shell command and returns an empty string on
// success or "error" on failure.
func Command(arg ...string) string {
	name := "/bin/bash"
	flag := "-c"
	if runtime.GOOS == "windows" {
		name = "cmd"
		flag = "/C"
	}
	cmd := exec.Command(name, append([]string{flag}, arg...)...)
	if err := cmd.Run(); err != nil {
		log.Errorf("[services.go] command error: %v", err)
		return "error"
	}
	return ""
}
