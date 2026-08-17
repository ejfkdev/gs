// live.go - 自动重载的 Engine 包装
//
// 长驻的搜索进程用 LiveEngine 持有索引: 它周期性检测索引目录变化 (items.bin
// 的 mtime), 变化时加载新 Engine 并原子换入。
//
// 并发约定: Search 全程持读锁, 重载持写锁换新引擎并关闭旧引擎。这样:
//   - 进行中的搜索不会被「换新 + 关旧」打断;
//   - 重载会等所有在途搜索结束后才换入 (即"更新期间短暂不可服务")。
// 注意: 不要长持某个 *Engine 指针 (没有公开 getter, 正是为了避免引用失效)。

package gs

import (
	"context"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// LiveEngine: 持有 *Engine 并在后台自动重载
type LiveEngine struct {
	mu        sync.RWMutex
	eng       *Engine
	dir       string
	modTime   time.Time
	interval  time.Duration
	stop      chan struct{}
	done      chan struct{}
	closeOnce sync.Once
}

// OpenLive: 加载 dir 并开启自动重载 (每 interval 检查一次; interval<=0 用 5s)
func OpenLive(dir string, interval time.Duration) (*LiveEngine, error) {
	if interval <= 0 {
		interval = 5 * time.Second
	}
	eng, err := Load(dir)
	if err != nil {
		return nil, err
	}
	l := &LiveEngine{
		eng:      eng,
		dir:      dir,
		interval: interval,
		stop:     make(chan struct{}),
		done:     make(chan struct{}),
	}
	if info, err := os.Stat(filepath.Join(dir, "items.bin")); err == nil {
		l.modTime = info.ModTime()
	}
	go l.loop()
	return l, nil
}

// Search: 线程安全地搜索。全程持读锁, 直到此次搜索完成才释放, 因此后台重载
// 不会把还在用的引擎关掉。
func (l *LiveEngine) Search(ctx context.Context, opts SearchOptions) ([]Hit, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.eng.Search(ctx, opts)
}

// Close: 停止重载并关闭当前引擎。可安全重入。
func (l *LiveEngine) Close() error {
	var err error
	l.closeOnce.Do(func() {
		close(l.stop)
		<-l.done
		l.mu.Lock()
		e := l.eng
		l.mu.Unlock()
		err = e.Close()
	})
	return err
}

func (l *LiveEngine) loop() {
	defer close(l.done)
	t := time.NewTicker(l.interval)
	defer t.Stop()
	for {
		select {
		case <-l.stop:
			return
		case <-t.C:
			info, err := os.Stat(filepath.Join(l.dir, "items.bin"))
			if err != nil || info.ModTime().Equal(l.modTime) {
				continue
			}
			eng, err := loadLiveWithRetry(l.dir)
			if err != nil {
				continue
			}
			// 写锁换入新引擎; 旧引擎在锁外关闭, 不阻塞在途搜索 (它们持有读锁)
			l.mu.Lock()
			old := l.eng
			l.eng = eng
			l.modTime = info.ModTime()
			l.mu.Unlock()
			_ = old.Close()
		}
	}
}

// loadLiveWithRetry: 目录在 gs watch 原子替换的瞬间会短暂缺失, 重试几次
func loadLiveWithRetry(dir string) (*Engine, error) {
	var last error
	for i := 0; i < 5; i++ {
		eng, err := Load(dir)
		if err == nil {
			return eng, nil
		}
		last = err
		if !os.IsNotExist(err) {
			return nil, err
		}
		time.Sleep(time.Duration(i+1) * 100 * time.Millisecond)
	}
	return nil, last
}
