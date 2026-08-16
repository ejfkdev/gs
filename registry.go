// registry.go - 已加载引擎的缓存与多实例管理
//
// Load 每次都会重新解析磁盘上的索引文件 (schema + items + embeddings +
// 重建 inverted index)。如果进程内需要反复打开同一个目录, 或同时持有
// 多个不同索引实例, 用 Registry 按目录缓存并做引用计数:
//
//	reg := gs.NewRegistry()
//	eng, _ := reg.Open("./indexes/wiki")   // 首次解析, refs=1
//	eng2, _ := reg.Open("./indexes/wiki")  // 复用同一个 *Engine, refs=2
//	reg.Release(eng)                        // refs=1
//	reg.Release(eng2)                       // refs=0 → 关闭引擎
//	reg.CloseAll()                          // 关闭所有缓存实例
//
// Registry 只做进程内缓存 (不落到磁盘), 不同目录的实例彼此独立。

package gs

import (
	"path/filepath"
	"sort"
	"sync"
)

type regEntry struct {
	eng  *Engine
	refs int
}

// Registry: 按目录缓存已加载引擎的引用计数注册表。
// 零值不可用, 请用 NewRegistry 构造。并发安全。
type Registry struct {
	mu      sync.Mutex
	engines map[string]*regEntry
}

// NewRegistry: 创建一个空的引擎注册表
func NewRegistry() *Registry {
	return &Registry{engines: make(map[string]*regEntry)}
}

// Open: 返回 dataDir 对应的引擎; 若该目录已打开则复用同一个 *Engine
// 并把引用计数 +1, 否则 Load 一次。返回的引擎在引用归零前不要手动 Close。
func (r *Registry) Open(dataDir string) (*Engine, error) {
	key, err := filepath.Abs(dataDir)
	if err == nil {
		key = filepath.Clean(key)
	} else {
		key = dataDir
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if e, ok := r.engines[key]; ok {
		e.refs++
		return e.eng, nil
	}
	eng, err := Load(key)
	if err != nil {
		return nil, err
	}
	r.engines[key] = &regEntry{eng: eng, refs: 1}
	return eng, nil
}

// Release: 释放一次对引擎的引用; 引用归零时关闭该引擎并移除缓存。
// 传非本 Registry 管理的引擎时是安全的 no-op。
func (r *Registry) Release(eng *Engine) {
	if eng == nil {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	for key, e := range r.engines {
		if e.eng == eng {
			e.refs--
			if e.refs <= 0 {
				_ = eng.Close()
				delete(r.engines, key)
			}
			return
		}
	}
}

// Remove: 立即关闭并移除 dataDir 对应的缓存引擎 (忽略引用计数,
// 用于"索引被重建后想强制换新"的场景)。
func (r *Registry) Remove(dataDir string) {
	key, err := filepath.Abs(dataDir)
	if err == nil {
		key = filepath.Clean(key)
	} else {
		key = dataDir
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if e, ok := r.engines[key]; ok {
		_ = e.eng.Close()
		delete(r.engines, key)
	}
}

// CloseAll: 关闭并清空所有缓存引擎
func (r *Registry) CloseAll() {
	r.mu.Lock()
	defer r.mu.Unlock()
	for key, e := range r.engines {
		_ = e.eng.Close()
		delete(r.engines, key)
	}
}

// Len: 当前缓存的引擎实例数
func (r *Registry) Len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.engines)
}

// Instances: 当前缓存实例的规范化目录列表 (字典序)
func (r *Registry) Instances() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]string, 0, len(r.engines))
	for k := range r.engines {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
