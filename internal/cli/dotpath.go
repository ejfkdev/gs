// dotpath.go - 点路径取值（零依赖，覆盖 JSON 树与 YAML 树）
//
// 语法: "title"、"a.b.c"、"list.0.name"、"list[0].name"、"a[0][1].x"

package cli

import (
	"strconv"
	"strings"
)

// getPath: 从 interface{} 树按点路径取值
func getPath(root interface{}, path string) (interface{}, bool) {
	if path == "" {
		return nil, false
	}
	cur := root
	for _, part := range strings.Split(path, ".") {
		if part == "" {
			return nil, false
		}
		rest := part
		for rest != "" {
			if strings.HasPrefix(rest, "[") {
				end := strings.IndexByte(rest, ']')
				if end < 0 {
					return nil, false
				}
				n, err := strconv.Atoi(rest[1:end])
				if err != nil {
					return nil, false
				}
				v, ok := arrayGet(cur, n)
				if !ok {
					return nil, false
				}
				cur = v
				rest = rest[end+1:]
			} else {
				end := strings.IndexByte(rest, '[')
				key := rest
				if end >= 0 {
					key = rest[:end]
				}
				v, ok := descend(cur, key)
				if !ok {
					return nil, false
				}
				cur = v
				if end < 0 {
					rest = ""
				} else {
					rest = rest[end:]
				}
			}
		}
	}
	return cur, true
}

// mapGet: 同时兼容 JSON 的 map[string]interface{} 与 yaml 的 map[interface{}]interface{}
func mapGet(m interface{}, key string) (interface{}, bool) {
	switch mm := m.(type) {
	case map[string]interface{}:
		v, ok := mm[key]
		return v, ok
	case map[interface{}]interface{}:
		v, ok := mm[key]
		return v, ok
	}
	return nil, false
}

// descend: 当前节点是数组时按下标, 否则按 map key 取值 (支持 "a.0" 这种裸下标)
func descend(cur interface{}, key string) (interface{}, bool) {
	if arr, ok := cur.([]interface{}); ok {
		if n, err := strconv.Atoi(key); err == nil {
			return arrayGet(arr, n)
		}
		return nil, false
	}
	return mapGet(cur, key)
}

// arrayGet: 数组按下标取值
func arrayGet(a interface{}, i int) (interface{}, bool) {
	if arr, ok := a.([]interface{}); ok && i >= 0 && i < len(arr) {
		return arr[i], true
	}
	return nil, false
}
