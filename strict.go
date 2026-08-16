// strict.go - 格式检测器 + 严格模式匹配
// 用于: IP、域名、邮箱、哈希、路径、URL 等固定格式的精确搜索
// 多字段版本: 接受一组 "field_name → text" 的映射, 在所有这些字段上找 strict token

package gs

import (
	"regexp"
	"strings"
)

// StrictToken: 一种格式 token + 原始字符串
type StrictToken struct {
	Type string // ip, domain, email, hash_md5, hash_sha1, hash_sha256, file_path, url
	Raw  string // 原始字符串
	Norm string // 归一化 (lowercase)
}

// strictPatterns: 各种格式的正则
var (
	ipRE         = regexp.MustCompile(`\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b`)
	domainRE     = regexp.MustCompile(`\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,24}\b`)
	emailRE      = regexp.MustCompile(`\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b`)
	hashMd5RE    = regexp.MustCompile(`\b[a-fA-F0-9]{32}\b`)
	hashSha1RE   = regexp.MustCompile(`\b[a-fA-F0-9]{40}\b`)
	hashSha256RE = regexp.MustCompile(`\b[a-fA-F0-9]{64}\b`)
	filePathRE   = regexp.MustCompile(`(?:/[a-zA-Z0-9_.-]+){2,}|\\[\w\-. \\]+`)
	urlRE        = regexp.MustCompile(`\bhttps?://[^\s<>"']+\b`)
)

// ExtractStrictTokens: 从 query 提取所有格式 token
func ExtractStrictTokens(q string) []StrictToken {
	var tokens []StrictToken
	seen := make(map[string]bool)

	// IP
	for _, m := range ipRE.FindAllString(q, -1) {
		key := "ip:" + m
		if !seen[key] && !strictIsInAnyMatch(m, tokens) {
			tokens = append(tokens, StrictToken{Type: "ip", Raw: m, Norm: m})
			seen[key] = true
		}
	}

	// Domain
	for _, m := range domainRE.FindAllString(q, -1) {
		key := "domain:" + strings.ToLower(m)
		if !seen[key] {
			if !strings.Contains(m, "@") {
				tokens = append(tokens, StrictToken{Type: "domain", Raw: m, Norm: strings.ToLower(m)})
				seen[key] = true
			}
		}
	}

	// Email
	for _, m := range emailRE.FindAllString(q, -1) {
		key := "email:" + strings.ToLower(m)
		if !seen[key] {
			tokens = append(tokens, StrictToken{Type: "email", Raw: m, Norm: strings.ToLower(m)})
			seen[key] = true
		}
	}

	// Hash (按长度从大到小匹配避免重复)
	for _, m := range hashSha256RE.FindAllString(q, -1) {
		key := "hash:" + strings.ToLower(m)
		if !seen[key] {
			tokens = append(tokens, StrictToken{Type: "hash_sha256", Raw: m, Norm: strings.ToLower(m)})
			seen[key] = true
		}
	}
	for _, m := range hashSha1RE.FindAllString(q, -1) {
		key := "hash:" + strings.ToLower(m)
		if !seen[key] {
			tokens = append(tokens, StrictToken{Type: "hash_sha1", Raw: m, Norm: strings.ToLower(m)})
			seen[key] = true
		}
	}
	for _, m := range hashMd5RE.FindAllString(q, -1) {
		key := "hash:" + strings.ToLower(m)
		if !seen[key] {
			tokens = append(tokens, StrictToken{Type: "hash_md5", Raw: m, Norm: strings.ToLower(m)})
			seen[key] = true
		}
	}

	// URL
	for _, m := range urlRE.FindAllString(q, -1) {
		key := "url:" + m
		if !seen[key] {
			tokens = append(tokens, StrictToken{Type: "url", Raw: m, Norm: m})
			seen[key] = true
		}
	}

	// File path
	for _, m := range filePathRE.FindAllString(q, -1) {
		key := "path:" + m
		if !seen[key] {
			tokens = append(tokens, StrictToken{Type: "file_path", Raw: m, Norm: m})
			seen[key] = true
		}
	}

	return tokens
}

// strictIsInAnyMatch: 检查 m 是否已经被其他 match 包含 (避免重复)
func strictIsInAnyMatch(m string, tokens []StrictToken) bool {
	for _, t := range tokens {
		if strings.Contains(t.Raw, m) {
			return true
		}
	}
	return false
}

// HasStrictToken: query 是否含格式 token
func HasStrictToken(q string) bool {
	return len(ExtractStrictTokens(q)) > 0
}

// StrictField: 一个字段在 strict match 时的文本
type StrictField struct {
	Name string
	Text string
	// Weight: 命中权重, 0 表示使用默认
	Weight float32
}

// computeStrictBoost: 对每个 item 计算 strict match boost
// 输入: query, items 字段文本, item 数
// 返回: 每个 item 一个分数, 命中的 strict 类型
func computeStrictBoost(q string, items [][]StrictField, n int) ([]float32, [][]string) {
	if n <= 0 {
		return nil, nil
	}
	tokens := ExtractStrictTokens(q)
	if len(tokens) == 0 {
		// 无 strict 格式 token → 返回全 0 但长度仍为 n 的 boosts。
		// Search 在 opts.Strict=true 时会强制启用 strict 分支; 若这里返回 nil,
		// strictN 归一化后仍为空, 后续 strictN[i] 会越界 panic。
		return make([]float32, n), nil
	}
	boosts := make([]float32, n)
	matchedTypes := make([][]string, n)

	for _, tok := range tokens {
		for i := 0; i < n; i++ {
			var score float32
			var types []string
			for _, f := range items[i] {
				if f.Text == "" {
					continue
				}
				w := f.Weight
				if w == 0 {
					w = 0.15 // 默认
				}
				// 命中: 字段文本含 strict token
				if strings.Contains(f.Text, tok.Raw) {
					score += w
					types = append(types, tok.Type)
				}
			}
			// hash/domain 严格加成: 跨字段合计一次 (lowercase)
			if tok.Type == "hash_sha256" || tok.Type == "hash_sha1" || tok.Type == "hash_md5" || tok.Type == "domain" {
				combined := ""
				for _, f := range items[i] {
					combined += strings.ToLower(f.Text) + " "
				}
				if strings.Contains(combined, tok.Norm) {
					score += 0.2
					types = append(types, tok.Type)
				}
			}
			if score > 0 {
				boosts[i] += score
				matchedTypes[i] = append(matchedTypes[i], types...)
			}
		}
	}

	// 去重
	uniqueTypes := make([][]string, n)
	for i, ts := range matchedTypes {
		seen := make(map[string]bool)
		for _, t := range ts {
			if !seen[t] {
				seen[t] = true
				uniqueTypes[i] = append(uniqueTypes[i], t)
			}
		}
	}
	return boosts, uniqueTypes
}
