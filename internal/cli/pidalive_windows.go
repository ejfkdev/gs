//go:build windows

package cli

// pidAlive: Windows 下无法廉价地仅靠 pid 判断存活, 交给锁文件的心跳超时兜底。
// 这里恒返回 true 表示 "无法确认已死", 避免误抢占还活着的 watcher。
func pidAlive(_ int) bool { return true }