// watch.go - gs watch: 监控源目录变化, 变化后重建索引并用原子替换发布
//
// 并发安全: 每个循环把索引完整建到临时目录, 再 rename 换进正式目录。Load 会把
// 整个索引读进内存, 所以已加载的搜索进程不受后续替换影响; 正在 Load 的进程
// 要么读到旧快照、要么读到新快照 (交换瞬间有极小窗口, CLI 搜索用重试兜底)。
//
// 变化检测: fsnotify 监听为主 (即时触发), 轮询扫目录签名兜底 (防漏事件)。
// 崩溃恢复: 重启清残留 tmp/bak 并全量重建。单 watcher 用「含 pid + 心跳」的
// 锁文件互斥, pid 失效或心跳超时即抢占。

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
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/ejfkdev/gs"
	"github.com/fsnotify/fsnotify"
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
	fs.DurationVar(&interval, "interval", 5*time.Second, "poll / debounce interval")
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
fsnotify 即时触发 + 轮询兜底。

Flags:
  --config <file>          index config YAML (required)
  --output <dir>           output index directory (required)
  --bge-weights <file>     BGE 权重源文件 (HF model.safetensors)
  --bge-vocab <file>       BGE 词表源文件 (HF vocab.txt)
  --max-embed-runes <int>  BGE 每字段编码截断的最大 rune 数 (0 = 默认 512)
  --interval <dur>         fsnotify debounce / polling fallback interval (default 5s)
  -v                       verbose per-file logging
`

func watchLoop(o *buildOptions, interval time.Duration, stderr io.Writer) error {
	lock, err := acquireWatchLock(o.output)
	if err != nil {
		return err
	}
	defer releaseWatchLock(o.output, lock)

	cfg, err := gs.LoadIndexConfig(o.config)
	if err != nil {
		return err
	}
	if o.maxEmbedRunes > 0 {
		cfg.Schema.MaxEmbedRunes = o.maxEmbedRunes
	}

	if err := cleanupStale(o.output); err != nil {
		return err
	}
	if err := o.rebuildAndSwap(cfg, stderr); err != nil {
		return fmt.Errorf("initial build: %w", err)
	}
	sig, err := scanSignature(cfg)
	if err != nil {
		return err
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	trigger := make(chan struct{}, 1)
	send := func() {
		select {
		case trigger <- struct{}{}:
		default:
		}
	}

	// 轮询兜底: 每个 interval 扫一次签名 + 续心跳
	go func() {
		t := time.NewTicker(interval)
		defer t.Stop()
		for range t.C {
			writeLockHeartbeat(lock)
			send()
		}
	}()

	// fsnotify: 即时触发 (debounce 到 interval, 防止过于频繁)
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		fmt.Fprintf(stderr, "[watch] fsnotify unavailable (%v), falling back to polling\n", err)
	} else {
		defer watcher.Close()
		for _, src := range cfg.Sources {
			addRecursiveWatch(watcher, src.Dir)
		}
		go func() {
			d := time.NewTicker(interval)
			defer d.Stop()
			pending := false
			for {
				select {
				case ev, ok := <-watcher.Events:
					if !ok {
						return
					}
					// 新目录出现 → 加 watch
					if ev.Op&fsnotify.Create != 0 {
						if info, err := os.Stat(ev.Name); err == nil && info.IsDir() {
							_ = watcher.Add(ev.Name)
						}
					}
					pending = true
				case _, ok := <-watcher.Errors:
					if !ok {
						return
					}
				case <-d.C:
					if pending {
						pending = false
						send()
					}
				}
			}
		}()
	}

	for {
		select {
		case <-sigCh:
			fmt.Fprintln(stderr, "[watch] interrupted, exiting")
			return nil
		case <-trigger:
			ns, err := scanSignature(cfg)
			if err != nil {
				fmt.Fprintf(stderr, "[watch] scan error: %v (will retry)\n", err)
				continue
			}
			if reflect.DeepEqual(sig, ns) {
				continue
			}
			fmt.Fprintln(stderr, "[watch] source changed, rebuilding...")
			if err := o.rebuildAndSwap(cfg, stderr); err != nil {
				fmt.Fprintf(stderr, "[watch] rebuild failed: %v (will retry)\n", err)
				continue
			}
			sig = ns
		}
	}
}

// rebuildAndSwap: 建到临时目录 → 原子替换到 output
func (o *buildOptions) rebuildAndSwap(cfg *gs.IndexConfig, stderr io.Writer) error {
	tmp := o.output + ".tmp"
	if err := os.RemoveAll(tmp); err != nil {
		return err
	}
	opts := []gs.IndexBuildOption{
		gs.IndexWithProgress(o.progress(stderr)),
		gs.IndexWithEmbedCache(o.output + ".embcache"), // 增量缓存, 落在 output 旁, 跨重建/重启持久
	}
	if o.bgeWeights != "" {
		opts = append(opts, gs.IndexWithBGEPaths(o.bgeWeights, o.bgeVocab))
	}
	if _, err := cfg.Build(tmp, opts...); err != nil {
		return err
	}
	if err := atomicSwapDir(tmp, o.output); err != nil {
		return err
	}
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
	_ = os.RemoveAll(bak)
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
func scanSignature(cfg *gs.IndexConfig) (map[string]fileSig, error) {
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
			if !gs.MatchGlob(include, filepath.ToSlash(rel)) {
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

// addRecursiveWatch: 递归给 dir 及其所有子目录加 fsnotify watch
func addRecursiveWatch(w *fsnotify.Watcher, dir string) {
	_ = filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil || !d.IsDir() {
			return nil
		}
		_ = w.Add(p)
		return nil
	})
}

// ------------------------------------------------------------------ 单 watcher 锁

const watchLockStale = 30 * time.Second

func acquireWatchLock(dest string) (*os.File, error) {
	lockPath := dest + ".watch.lock"
	for i := 0; i < 2; i++ {
		f, err := os.OpenFile(lockPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
		if err == nil {
			writeLockHeartbeat(f)
			return f, nil
		}
		if !os.IsExist(err) {
			return nil, err
		}
		if lockStale(lockPath) {
			_ = os.Remove(lockPath)
			continue // 重新抢
		}
		return nil, fmt.Errorf("another watch is running (lock %s)", lockPath)
	}
	return nil, fmt.Errorf("could not acquire watch lock %s", lockPath)
}

func releaseWatchLock(dest string, f *os.File) {
	if f != nil {
		_ = f.Close()
	}
	_ = os.Remove(dest + ".watch.lock")
}

// writeLockHeartbeat: 写 pid + 心跳时间戳 (持有者周期性刷新)
func writeLockHeartbeat(f *os.File) {
	if f == nil {
		return
	}
	_ = f.Truncate(0)
	_, _ = f.Seek(0, 0)
	_, _ = fmt.Fprintf(f, "%d\n%d\n", os.Getpid(), time.Now().UnixNano())
	_ = f.Sync()
}

// lockStale: 锁是否失效 —— pid 已死, 或心跳超时 (Windows 无 pid 探测时靠心跳)
func lockStale(lockPath string) bool {
	b, err := os.ReadFile(lockPath)
	if err != nil {
		return true
	}
	lines := strings.SplitN(strings.TrimSpace(string(b)), "\n", 2)
	pid, _ := strconv.Atoi(strings.TrimSpace(lines[0]))
	if pid > 0 && !pidAlive(pid) {
		return true
	}
	if len(lines) == 2 {
		if t, err := strconv.ParseInt(strings.TrimSpace(lines[1]), 10, 64); err == nil {
			if time.Since(time.Unix(0, t)) > watchLockStale {
				return true
			}
		}
	}
	return false
}
