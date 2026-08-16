// Command gs is the single-binary CLI (build / search / version) built
// on top of the gs hybrid search library.
//
// Usage:
//
//	gs build <skills|wiki> [flags]
//	gs search --index <dir> [flags] [<query>]
//	gs version | help
package main

import (
	"os"

	"github.com/ejfkdev/gs/internal/cli"
)

func main() {
	os.Exit(cli.Run(os.Args[1:], os.Stdout, os.Stderr))
}
