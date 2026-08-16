//go:build !windows

package cli

import "syscall"

// pidAlive: 进程是否存活 (posix signal 0 探测, 无副作用)
func pidAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	err := syscall.Kill(pid, 0)
	return err == nil || err == syscall.EPERM
}
