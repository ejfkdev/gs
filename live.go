// live.go - 自动重载的 Engine 包装
//
// 长驻的搜索进程用 LiveEngine 持有索引: 它周期性检测索引目录变化 (items.bin
// 的 mtime), 变化时加载新 Engine 并原子换入, 换入期间 Search 会被 RWMutex
// 短暂串行化 (符合"更新期间可能短暂不可服务"的预期)。配合 gs watch 的原子
// 目录替换, 碰撞窗口极小, 加载时还会对"目录短暂缺失"做重试。

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
	mu       sync.RWMutex
	eng      *Engine
	dir      string
	modTime  time.Time
	interval time.Duration
	stop     chan struct{}
	done     chan struct{}
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

// Search: 线程安全地搜索 (重载瞬间可能短暂阻塞在写锁上)
func (l *LiveEngine) Search(ctx context.Context, opts SearchOptions) ([]Hit, error) {
	l.mu.RLock()
	e := l.eng
	l.mu.RUnlock()
	return e.Search(ctx, opts)
}

// Engine: 返回当前引擎 (只读; 可能被后台重载替换, 不应长期持有)
func (l *LiveEngine) Engine() *Engine {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.eng
}

// Close: 停止重载并关闭当前引擎
func (l *LiveEngine) Close() error {
	close(l.stop)
	<-l.done
	l.mu.Lock()
	e := l.eng
	l.mu.Unlock()
	return e.Close()
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
