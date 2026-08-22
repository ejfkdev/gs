// Package cli implements the gs single-binary command-line interface
// (build / search / version subcommands) over the gs library.
package cli

import (
	"errors"
	"flag"
	"fmt"
	"io"

	"github.com/ejfkdev/gs/cmd/gs/internal/xyzsvc"
	"github.com/ejfkdev/xyz-go"
)

// Version: 与 cmd/gs 二进制对应的版本号
const Version = "0.3.3"

// rootUsage 顶部带上程序名 + 版本号 + 描述 + 仓库地址, 便于一眼确认版本。
func rootUsage() string {
	return "gs v" + Version + ` — hybrid (BM25 + BGE) full-text search for local knowledge bases.
https://github.com/ejfkdev/gs

Usage:
  gs <command> [arguments] [flags]

Commands (one definition; CLI / HTTP / MCP):
  fastsearch  Fast BM25 search (text as query)
  index       Rebuild an index from a config YAML
  schema      Show an index's schema (field definitions)
  search      Search a prebuilt index

Modes:
  serve       HTTP service (REST + /openapi.json + /mcp)
  mcp         MCP tool server (stdio|sse|http)

Local commands:
  build       Build an index (skills/wiki or --config)
  watch       Watch source dirs and rebuild on change (atomic swap)

Environment:
  GS_INDEX    Default index directory (search/fastsearch)

Run "gs <command> -h" for command-specific help.
`
}

// Run: 执行 root 命令, 返回进程退出码 (0 成功, 1 运行错误, 2 用法错误)。
//
// search/schema/index/serve/mcp/help 由 xyz-go 统一派发 (一份定义, 三通道);
// build/watch/version 是 gs 特有的长驻/遗留子命令, 在这里预派发回本地实现。
func Run(args []string, stdout, stderr io.Writer) int {
	reg, err := xyzsvc.Registry()
	if err != nil {
		fmt.Fprintf(stderr, "gs: %v\n", err)
		return 2
	}
	if len(args) == 0 {
		fmt.Fprint(stdout, rootUsage())
		return 0
	}
	switch args[0] {
	case "-h", "--help", "help":
		fmt.Fprint(stdout, rootUsage())
		return 0
	case "version", "--version", "-V", "-v":
		fmt.Fprintf(stdout, "gs %s (github.com/ejfkdev/gs)\n", Version)
		return 0
	case "build":
		if err := runBuild(args[1:], stdout, stderr); err != nil {
			return printErr(stderr, "gs build", err)
		}
		return 0
	case "watch":
		if err := runWatch(args[1:], stdout, stderr); err != nil {
			return printErr(stderr, "gs watch", err)
		}
		return 0
	}
	// 界面语言固定中文 (与命令的 Summary/Description 一致); 用户可用
	// --xyz.lang=en 覆盖, 或改这里让 xyz 走 LANG 自动检测。
	return xyz.RunConfig(reg, args, xyz.Config{Lang: "zh-CN"})
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
