package common

import (
	"fmt"

	"github.com/jinzhu/configor"
)

// Config is the global runtime configuration instance, populated by Parser().
var Config = ServerConfig{}

func yamlparser(configPath string) {
	fmt.Println("config file:", configPath)
	if err := configor.Load(&Config, configPath); err != nil {
		fmt.Printf("warning: failed to load config file %q: %v\n", configPath, err)
	}
}
