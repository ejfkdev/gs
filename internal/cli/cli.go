// Package cli implements the gs single-binary command-line interface
// (build / search / version subcommands) over the gs library.
package cli

import (
	"errors"
	"flag"
	"fmt"
	"io"
)

// Version: 与 cmd/gs 二进制对应的版本号
const Version = "0.2.1"

const rootUsage = `gs — hybrid (BM25 + BGE) full-text search for local knowledge bases.

Usage:
  gs <command> [arguments] [flags]

Commands:
  build     Build a hybrid (BM25 + BGE) index from a source tree
  search    Query a prebuilt index
  version   Print version information

Run "gs <command> -h" for command-specific help.
`

// Run: 执行 root 命令, 返回进程退出码 (0 成功, 1 运行错误, 2 用法错误)
func Run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprint(stderr, rootUsage)
		return 2
	}
	switch args[0] {
	case "-h", "--help", "help":
		fmt.Fprint(stdout, rootUsage)
		return 0
	case "version", "--version", "-V":
		fmt.Fprintf(stdout, "gs %s (github.com/ejfkdev/gs)\n", Version)
		return 0
	case "build":
		if err := runBuild(args[1:], stdout, stderr); err != nil {
			return printErr(stderr, "gs build", err)
		}
		return 0
	case "search":
		if err := runSearch(args[1:], stdout, stderr); err != nil {
			return printErr(stderr, "gs search", err)
		}
		return 0
	default:
		fmt.Fprintf(stderr, "gs: unknown command %q\n\n%s", args[0], rootUsage)
		return 2
	}
}

// printErr: 统一错误输出; -h 触发的 flag.ErrHelp 视为正常退出。
func printErr(stderr io.Writer, cmd string, err error) int {
	if errors.Is(err, flag.ErrHelp) {
		return 0
	}
	fmt.Fprintf(stderr, "%s: %v\n", cmd, err)
	return 1
}

// humanBytes: 人类可读字节数 (1024 进制)
func humanBytes(n int64) string {
	const unit = 1024.0
	units := []string{"B", "KB", "MB", "GB", "TB"}
	f := float64(n)
	i := 0
	for f >= unit && i < len(units)-1 {
		f /= unit
		i++
	}
	if i == 0 {
		return fmt.Sprintf("%d B", n)
	}
	return fmt.Sprintf("%.1f %s", f, units[i])
}
