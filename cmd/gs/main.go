// Command gs is the single binary for the gs hybrid search library. Its
// search / schema / index capabilities are defined once (via xyz-go) and
// served over three interfaces:
//
//	gs search|schema|index ...        CLI subcommands
//	gs serve                           HTTP REST + /openapi.json + /mcp
//	gs mcp stdio|sse|http              MCP tool server
//	gs build|watch|version             local build / daemon / version
package main

import (
	"os"

	"github.com/ejfkdev/gs/cmd/gs/internal/cli"
)

func main() {
	os.Exit(cli.Run(os.Args[1:], os.Stdout, os.Stderr))
}
