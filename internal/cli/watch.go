// watch.go - gs watch: 监控源目录变化, 变化后重建索引并用原子替换发布
//
// 并发安全: 每个循环把索引完整建到临时目录, 再 rename 换进正式目录。Load 会把
// 整个索引读进内存, 所以已经加载的搜索进程完全不受后续替换影响; 正在 Load 的
// 进程要么读到旧快照、要么读到新快照 (交换瞬间有极小窗口, 由 CLI 搜索的重试兜底)。
// 崩溃恢复: 重启时清掉残留的临时/备份目录并做一次全量重建, 无需状态日志。

package cli

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"reflect"
	"syscall"
	"time"

	"github.com/ejfkdev/gs"
)

func runWatch(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("gs watch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { fmt.Fprint(stderr, watchUsage) }

	var opts buildOptions
	var interval time.Duration
	fs.StringVar(&opts.config, "config", "", "index config YAML (required)")
	fs.StringVar(&opts.output, "output", "", "output index directory (required)")
	fs.StringVar(&opts.bgeWeights, "bge-weights", "", "BGE 权重源文件 (HF model.safetensors)")
	fs.StringVar(&opts.bgeVocab, "bge-vocab", "", "BGE 词表源文件 (HF vocab.txt)")
	fs.IntVar(&opts.maxEmbedRunes, "max-embed-runes", 0, "BGE 每字段编码截断的最大 rune 数 (0 = 默认 512)")
	fs.DurationVar(&interval, "interval", 5*time.Second, "poll interval")
	fs.BoolVar(&opts.verbose, "v", false, "verbose per-file logging")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if opts.config == "" {
		return errors.New("--config is required")
	}
	if opts.output == "" {
		return errors.New("--output is required")
	}
	if interval <= 0 {
		return errors.New("--interval must be > 0")
	}
	return watchLoop(&opts, interval, stderr)
}

const watchUsage = `Usage: gs watch --config <index.yaml> --output <dir> [flags]

监控 --config 里的源目录, 变化后重建索引并用原子替换发布 (搜索进程可无锁并发读)。

Flags:
  --config <file>          index config YAML (required)
  --output <dir>           output index directory (required)
  --bge-weights <file>     BGE 权重源文件 (HF model.safetensors)
  --bge-vocab <file>       BGE 词表源文件 (HF vocab.txt)
  --max-embed-runes <int>  BGE 每字段编码截断的最大 rune 数 (0 = 默认 512)
  --interval <dur>         poll interval (default 5s)
  -v                       verbose per-file logging
`

func watchLoop(o *buildOptions, interval time.Duration, stderr io.Writer) error {
	lock, err := acquireWatchLock(o.output)
	if err != nil {
		return err
	}
	defer releaseWatchLock(o.output, lock)

	cfg, err := LoadConfig(o.config)
	if err != nil {
		return err
	}
	schema, err := cfg.Schema.Schema()
	if err != nil {
		return fmt.Errorf("schema: %w", err)
	}
	o.applyMaxEmbedRunes(schema)

	// 启动即清理上次残留 + 全量建一次 (崩溃恢复)
	if err := cleanupStale(o.output); err != nil {
		return err
	}
	if err := o.rebuildAndSwap(cfg, schema, stderr, true); err != nil {
		return fmt.Errorf("initial build: %w", err)
	}
	sig, err := scanSignature(cfg)
	if err != nil {
		return err
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-sigCh:
			fmt.Fprintln(stderr, "[watch] interrupted, exiting")
			return nil
		case <-ticker.C:
			newSig, err := scanSignature(cfg)
			if err != nil {
				fmt.Fprintf(stderr, "[watch] scan error: %v (will retry)\n", err)
				continue
			}
			if reflect.DeepEqual(sig, newSig) {
				continue
			}
			fmt.Fprintln(stderr, "[watch] source changed, rebuilding...")
			if err := o.rebuildAndSwap(cfg, schema, stderr, false); err != nil {
				fmt.Fprintf(stderr, "[watch] rebuild failed: %v (will retry on next change)\n", err)
				continue
			}
			sig = newSig
		}
	}
}

// rebuildAndSwap: 建到临时目录 → 原子替换到 output
func (o *buildOptions) rebuildAndSwap(cfg *Config, schema *gs.Schema, stderr io.Writer, initial bool) error {
	tmp := o.output + ".tmp"
	if err := os.RemoveAll(tmp); err != nil {
		return err
	}
	_, err := o.buildConfigTo(cfg, schema, tmp, stderr)
	if err != nil {
		return err
	}
	if err := atomicSwapDir(tmp, o.output); err != nil {
		return err
	}
	_ = initial
	fmt.Fprintf(stderr, "[watch] published index → %s\n", o.output)
	return nil
}

// atomicSwapDir: 把 tmp 原子地换到 dest (先改名旧目录, 再改入新目录, 清旧)
func atomicSwapDir(tmp, dest string) error {
	bak := dest + ".bak"
	if err := os.RemoveAll(bak); err != nil {
		return err
	}
	if _, err := os.Stat(dest); err == nil {
		if err := os.Rename(dest, bak); err != nil {
			return err
		}
	}
	if err := os.Rename(tmp, dest); err != nil {
		_ = os.Rename(bak, dest) // 回滚
		return err
	}
	_ = os.RemoveAll(bak) // 清理旧快照 (失败也不影响已完成的替换)
	return nil
}

// cleanupStale: 清掉上次异常退出残留的 tmp/bak
func cleanupStale(dest string) error {
	for _, p := range []string{dest + ".tmp", dest + ".bak"} {
		if err := os.RemoveAll(p); err != nil {
			return err
		}
	}
	return nil
}

type fileSig struct {
	size  int64
	mtime int64
}

// scanSignature: 所有匹配文件的 (size, mtime) 快照, key = dir + "/" + rel
func scanSignature(cfg *Config) (map[string]fileSig, error) {
	sig := map[string]fileSig{}
	for _, src := range cfg.Sources {
		include := src.Include
		if include == "" {
			include = "**"
		}
		err := filepath.WalkDir(src.Dir, func(p string, d os.DirEntry, werr error) error {
			if werr != nil || d.IsDir() {
				return nil
			}
			rel, err := filepath.Rel(src.Dir, p)
			if err != nil {
				rel = p
			}
			if !matchGlob(include, filepath.ToSlash(rel)) {
				return nil
			}
			info, err := d.Info()
			if err != nil {
				return nil
			}
			sig[src.Dir+"/"+filepath.ToSlash(rel)] = fileSig{size: info.Size(), mtime: info.ModTime().UnixNano()}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return sig, nil
}

// ------------------------------------------------------------------ 单 watcher 锁

func acquireWatchLock(dest string) (*os.File, error) {
	lockPath := dest + ".watch.lock"
	f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		if os.IsExist(err) {
			return nil, fmt.Errorf("another watch may be running (lock %s). If it is stale, delete it and retry", lockPath)
		}
		return nil, err
	}
	_, _ = f.WriteString(fmt.Sprintf("%d\n", os.Getpid()))
	return f, nil
}

func releaseWatchLock(dest string, f *os.File) {
	if f != nil {
		_ = f.Close()
	}
	_ = os.Remove(dest + ".watch.lock")
}
