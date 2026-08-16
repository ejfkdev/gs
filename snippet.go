// snippet.go - 从 doc 字段提取 query 命中的所有片段
// 多个命中位置 → 多个 snippet → 用 " ... " 连接 → 高亮 query token
// 多字段版本: 接受单字段文本, 返回该字段的 snippet

package gs

import (
	"sort"
	"strings"
	"unicode/utf8"
)

// SnippetRange: 一个命中文本片段
type SnippetRange struct {
	Start   int    // 起始 byte offset
	End     int    // 结束 byte offset
	Text    string // 实际文本
	Matched []int  // 命中的 query token 位置 (用于高亮)
}

// ExtractSnippets: 从文本中找所有 query token 出现位置, 提取前后文片段
// maxSnippets: 最多几个片段 (默认 3)
// contextBefore/After: 前后文长度 (runes)
// 返回: 拼接好的 snippet 字符串, 用 " ... " 分隔
func ExtractSnippets(text, query string, maxSnippets, contextBefore, contextAfter int) string {
	if text == "" {
		return ""
	}
	if query == "" {
		return truncateText(text, 300)
	}
	if maxSnippets <= 0 {
		maxSnippets = 3
	}
	if contextBefore <= 0 {
		contextBefore = 60
	}
	if contextAfter <= 0 {
		contextAfter = 100
	}

	// 1. 切 query tokens (去重)
	qTokens := extractQueryTokens(query)
	if len(qTokens) == 0 {
		return truncateText(text, 300)
	}

	// 2. 找所有 token 在 text 中的位置 (byte offset, 大小写不敏感)
	textLower := strings.ToLower(text)
	type matchPos struct {
		Pos int
		Len int
	}
	var matches []matchPos
	for _, tok := range qTokens {
		tokLower := strings.ToLower(tok)
		if tokLower == "" {
			continue
		}
		start := 0
		for {
			idx := strings.Index(textLower[start:], tokLower)
			if idx < 0 {
				break
			}
			idx += start
			end := idx + len(tokLower)
			leftOK := idx == 0 || !isWordByte(textLower[idx-1])
			rightOK := end >= len(textLower) || !isWordByte(textLower[end])
			if leftOK && rightOK {
				matches = append(matches, matchPos{Pos: idx, Len: len(tokLower)})
			}
			start = idx + 1
		}
	}
	if len(matches) == 0 {
		return truncateText(text, 300)
	}

	// 3. 按位置排序, 聚类
	sort.Slice(matches, func(i, j int) bool { return matches[i].Pos < matches[j].Pos })
	runes := []rune(text)

	const clusterGap = 40
	type cluster struct{ First, Last int } // rune offsets
	cur := cluster{byteToRuneIndex(text, matches[0].Pos), byteToRuneIndex(text, matches[0].Pos+matches[0].Len)}
	var clusters []cluster
	for _, m := range matches[1:] {
		s := byteToRuneIndex(text, m.Pos)
		e := byteToRuneIndex(text, m.Pos+m.Len)
		if s <= cur.Last+clusterGap {
			if e > cur.Last {
				cur.Last = e
			}
		} else {
			clusters = append(clusters, cur)
			cur = cluster{s, e}
		}
	}
	clusters = append(clusters, cur)

	if contextBefore > 40 {
		contextBefore = 40
	}
	if contextAfter > 60 {
		contextAfter = 60
	}

	type runeRange struct{ Start, End int }
	ranges := make([]runeRange, 0, len(clusters))
	for _, c := range clusters {
		rs := c.First - contextBefore
		if rs < 0 {
			rs = 0
		}
		re := c.Last + contextAfter
		if re > len(runes) {
			re = len(runes)
		}
		ranges = append(ranges, runeRange{rs, re})
	}

	// 强制不重叠
	const minGap = 5
	for i := 0; i+1 < len(ranges); i++ {
		if ranges[i].End > ranges[i+1].Start-minGap {
			ranges[i].End = ranges[i+1].Start - minGap
			if ranges[i].End < ranges[i].Start+1 {
				ranges[i].End = ranges[i].Start + 1
			}
		}
	}

	// 限制最多 maxSnippets 段
	if len(ranges) > maxSnippets {
		head := ranges[0]
		tail := ranges[len(ranges)-(maxSnippets-1):]
		ranges = append([]runeRange{head}, tail...)
		for i := 0; i+1 < len(ranges); i++ {
			if ranges[i].End > ranges[i+1].Start-minGap {
				ranges[i].End = ranges[i+1].Start - minGap
				if ranges[i].End < ranges[i].Start+1 {
					ranges[i].End = ranges[i].Start + 1
				}
			}
		}
	}

	// 4. 输出, 高亮 query token
	var sb strings.Builder
	for i, r := range ranges {
		if i > 0 {
			sb.WriteString(" ... ")
		}
		writeHighlighted(&sb, string(runes[r.Start:r.End]), qTokens)
	}
	return sb.String()
}

// writeHighlighted: 在文本中找 query token, 用 【TOKEN】 包裹
func writeHighlighted(sb *strings.Builder, text string, qTokens []string) {
	textLower := strings.ToLower(text)
	pos := 0
	for pos < len(text) {
		nextMatch, nextMatchLen := -1, 0
		for _, tok := range qTokens {
			tokLower := strings.ToLower(tok)
			if tokLower == "" {
				continue
			}
			idx := strings.Index(textLower[pos:], tokLower)
			if idx >= 0 {
				idx += pos
				// 词边界检查
				leftOK := idx == 0 || !isWordByte(textLower[idx-1])
				rightOK := idx+len(tokLower) >= len(textLower) || !isWordByte(textLower[idx+len(tokLower)])
				if leftOK && rightOK && (nextMatch < 0 || idx < nextMatch) {
					nextMatch, nextMatchLen = idx, len(tokLower)
				}
			}
		}
		if nextMatch < 0 {
			sb.WriteString(text[pos:])
			break
		}
		if nextMatch > pos {
			sb.WriteString(text[pos:nextMatch])
		}
		sb.WriteString("【")
		sb.WriteString(text[nextMatch : nextMatch+nextMatchLen])
		sb.WriteString("】")
		pos = nextMatch + nextMatchLen
	}
}

// extractQueryTokens: 从 query 提取 snippet 用的 token (小写, 去重,
// 跳过 CJK 单字 — 它们只服务于 BM25 召回, 不应逐字高亮)
func extractQueryTokens(query string) []string {
	toks := tokenize(query)
	seen := make(map[string]bool, len(toks))
	var out []string
	for _, t := range toks {
		t = strings.ToLower(t)
		if t == "" {
			continue
		}
		if rs := []rune(t); len(rs) == 1 && isCJK(rs[0]) {
			continue
		}
		if !seen[t] {
			seen[t] = true
			out = append(out, t)
		}
	}
	return out
}

func isWordByte(b byte) bool {
	return b == '_' ||
		(b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') ||
		(b >= '0' && b <= '9') ||
		b >= 0x80 // 非 ASCII 按"单词字符"处理 (CJK)
}

// byteToRuneIndex: byte offset → rune offset
func byteToRuneIndex(s string, byteOff int) int {
	if byteOff <= 0 {
		return 0
	}
	if byteOff >= len(s) {
		return utf8.RuneCountInString(s)
	}
	return utf8.RuneCountInString(s[:byteOff])
}

// truncateText: 返回前 N 字符
func truncateText(text string, n int) string {
	if utf8.RuneCountInString(text) <= n {
		return text
	}
	return string([]rune(text)[:n]) + "..."
}
