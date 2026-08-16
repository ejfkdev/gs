// tokenize.go - 纯 Go、线程安全、无词典的分词器
//
// 索引和查询共用同一个 tokenizer, 保证两边切分一致。
// 规则:
//   - ASCII 字母/数字连续段 → 单个 token (统一转小写)
//   - CJK 连续段 → 相邻双字组合 (bigram); 孤立的单个汉字才单独成 token
//   - 标点、空白等其他字符作为分隔符
//
// 之前的实现依赖 gojieba (cgo 封装 cppjieba), 破坏了 "纯 Go / 零 cgo /
// 可交叉编译" 的约束, 且 init() 中的全局初始化对库使用者不友好。
// bigram 方案无词典、可并行、跨平台, 行为与 jieba CutForSearch 的
// 中文输出对齐 (jieba 搜索模式对多字词同样产出双字重叠组合)。

package gs

import (
	"unicode"
	"unicode/utf8"
)

// tokenize: 把文本切成搜索 token (拉丁部分小写)。调用方负责去重。
func tokenize(text string) []string {
	var out []string
	for i := 0; i < len(text); {
		r, size := utf8.DecodeRuneInString(text[i:])
		switch {
		case isCJK(r):
			// CJK 连续段 → bigram。与 jieba 搜索模式一致: 多字词只保留
			// 双字重叠组合, 单字 token 仅当汉字孤立成词时产生。
			// 全量 unigram 会让短文档 (title 型字段) 靠高频单字权重
			// 挤掉真正命中的长词文档。
			// 优化: 所有 isCJK 字符都是 3 字节 UTF-8, 直接从原文字节切片做
			// bigram (零拼接分配), 不再收集 []rune 或 string(rune)+string(rune)。
			start := i
			n := 0
			for i < len(text) {
				r2, sz := utf8.DecodeRuneInString(text[i:])
				if !isCJK(r2) {
					break
				}
				i += sz
				n++
			}
			if n == 1 {
				out = append(out, text[start:i])
			} else {
				for j := 0; j+1 < n; j++ {
					out = append(out, text[start+3*j:start+3*j+6])
				}
			}
		case isWordRune(r):
			// 拉丁字母/数字连续段
			start := i
			for i < len(text) {
				r2, sz := utf8.DecodeRuneInString(text[i:])
				if !isWordRune(r2) || isCJK(r2) {
					break
				}
				i += sz
			}
			out = append(out, lowerASCII(text[start:i]))
		default:
			i += size
		}
	}
	return out
}

// tokenizeDedup: tokenize + 去重 (保留顺序), 用于 query 侧。
func tokenizeDedup(text string) []string {
	toks := tokenize(text)
	seen := make(map[string]bool, len(toks))
	out := toks[:0]
	for _, t := range toks {
		if !seen[t] {
			seen[t] = true
			out = append(out, t)
		}
	}
	return out
}

// isCJK: CJK 统一表意文字 + 假名
func isCJK(r rune) bool {
	return (r >= 0x3400 && r <= 0x4DBF) || // CJK Ext-A
		(r >= 0x4E00 && r <= 0x9FFF) || // CJK 基本区
		(r >= 0xF900 && r <= 0xFAFF) || // 兼容表意文字
		(r >= 0x3040 && r <= 0x30FF) // 平假名/片假名
}

// isWordRune: 可构成 token 的字符 (字母/数字/下划线; CJK 在外层分支已处理,
// 这里排除 CJK 标点等非字母字符)
func isWordRune(r rune) bool {
	if r == '_' {
		return true
	}
	if r < 0x80 {
		return (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9')
	}
	return unicode.IsLetter(r) || unicode.IsDigit(r)
}

// lowerASCII: 仅在需要时分配, A-Z → a-z
func lowerASCII(s string) string {
	hasUpper := false
	for i := 0; i < len(s); i++ {
		if s[i] >= 'A' && s[i] <= 'Z' {
			hasUpper = true
			break
		}
	}
	if !hasUpper {
		return s
	}
	b := []byte(s)
	for i := range b {
		if b[i] >= 'A' && b[i] <= 'Z' {
			b[i] += 32
		}
	}
	return string(b)
}

// truncateRunes: 截断到前 maxRunes 个 rune (O(maxRunes), 不做全文 []rune 转换)。
// 用于 BGE 编码前把超长文本截到 100 rune, 避免对整篇长文档做 rune 转换。
func truncateRunes(s string, maxRunes int) string {
	if maxRunes <= 0 {
		return ""
	}
	i, n := 0, 0
	for n < maxRunes && i < len(s) {
		_, size := utf8.DecodeRuneInString(s[i:])
		i += size
		n++
	}
	return s[:i]
}
