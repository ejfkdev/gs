// indexer_extract.go - 配置驱动的多格式提取 + 点路径取值
//
// 每个 source 支持一种 format:
//   - json        文件 → JSON 对象 → 点路径取字段
//   - yaml        文件 → YAML → 点路径取字段
//   - frontmatter YAML frontmatter + 正文 (正文用 "__body__")
//   - text        整文件当单个字段 ("__text__"/"__body__")
//   - csv         CSV 表 (首行表头, 每数据行一个 Item)
//   - ndjson      JSON Lines (每行一个 JSON 对象, 每行一个 Item)
//   - xlsx        首个 sheet (首行表头, 单元格按字符串)

package gs

import (
	"archive/zip"
	"bufio"
	"bytes"
	"encoding/csv"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"os"
	"path"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// 特殊取值路径
const (
	pathBody = "__body__" // frontmatter/text 的正文
	pathText = "__text__" // text 的整文件内容
)

// MatchGlob: 支持 ** 的 glob 匹配 (path 用 "/" 分隔)
func MatchGlob(pattern, name string) bool {
	ps := strings.Split(path.Clean(pattern), "/")
	ns := strings.Split(path.Clean(name), "/")
	return matchGlobSegs(ps, ns)
}

func matchGlobSegs(ps, ns []string) bool {
	if len(ps) == 0 {
		return len(ns) == 0
	}
	if ps[0] == "**" {
		for i := 0; i <= len(ns); i++ {
			if matchGlobSegs(ps[1:], ns[i:]) {
				return true
			}
		}
		return false
	}
	if len(ns) == 0 {
		return false
	}
	ok, err := path.Match(ps[0], ns[0])
	if err != nil || !ok {
		return false
	}
	return matchGlobSegs(ps[1:], ns[1:])
}

// ------------------------------------------------------------------ 点路径

// getPath: 从 interface{} 树按点路径取值。语法: "title"、"a.b.c"、"a[0].b"
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

func descend(cur interface{}, key string) (interface{}, bool) {
	if arr, ok := cur.([]interface{}); ok {
		if n, err := strconv.Atoi(key); err == nil {
			return arrayGet(arr, n)
		}
		return nil, false
	}
	return mapGet(cur, key)
}

func arrayGet(a interface{}, i int) (interface{}, bool) {
	if arr, ok := a.([]interface{}); ok && i >= 0 && i < len(arr) {
		return arr[i], true
	}
	return nil, false
}

// ------------------------------------------------------------------ 提取

// extractSource: 一个文件 → 若干 Item (json/yaml/frontmatter/text 通常 1 个,
// csv/ndjson/xlsx 每行 1 个)。
func extractSource(src SourceConfig, tagsFields map[string]bool, absPath, relPath string) ([]Item, error) {
	data, err := os.ReadFile(absPath)
	if err != nil {
		return nil, err
	}
	switch src.Format {
	case "json":
		it, err := extractJSON(data, relPath, src.Mapping, tagsFields)
		if err != nil {
			return nil, err
		}
		return []Item{it}, nil
	case "yaml", "yml":
		it, err := extractYAML(data, relPath, src.Mapping, tagsFields)
		if err != nil {
			return nil, err
		}
		return []Item{it}, nil
	case "frontmatter", "markdown":
		it, err := extractFrontmatter(data, relPath, src.Mapping, tagsFields)
		if err != nil {
			return nil, err
		}
		return []Item{it}, nil
	case "text":
		it, err := extractText(data, relPath, src.Mapping)
		if err != nil {
			return nil, err
		}
		return []Item{it}, nil
	case "csv":
		return extractCSV(data, relPath, src.Mapping, tagsFields)
	case "ndjson", "jsonl":
		return extractNDJSON(data, relPath, src.Mapping, tagsFields)
	case "xlsx":
		return extractXLSX(data, relPath, src.Mapping, tagsFields)
	default:
		return nil, fmt.Errorf("unsupported format %q (want json|yaml|frontmatter|text|csv|ndjson|jsonl|xlsx)", src.Format)
	}
}

func extractJSON(data []byte, relPath string, mapping map[string]string, tags map[string]bool) (Item, error) {
	var root interface{}
	if err := json.Unmarshal(data, &root); err != nil {
		return Item{}, err
	}
	return buildItem(root, "", relPath, mapping, tags)
}

func extractYAML(data []byte, relPath string, mapping map[string]string, tags map[string]bool) (Item, error) {
	var root interface{}
	if err := yaml.Unmarshal(data, &root); err != nil {
		return Item{}, err
	}
	return buildItem(root, "", relPath, mapping, tags)
}

// buildItem: 按 mapping 从树里取值, 拼成 Item
func buildItem(root interface{}, body, relPath string, mapping map[string]string, tags map[string]bool) (Item, error) {
	it := Item{ID: relPath, Path: relPath, Fields: make(map[string]string, len(mapping))}
	for field, p := range mapping {
		if p == pathBody || p == pathText {
			it.Fields[field] = body
			continue
		}
		v, ok := getPath(root, p)
		if !ok {
			it.Fields[field] = ""
			continue
		}
		if tags[field] {
			if t := valueToTags(v); t != nil {
				it.Tags = t
				continue // 交给 Builder 合并进 FieldTags 字段
			}
		}
		it.Fields[field] = stringifyValue(v)
	}
	return it, nil
}

func extractFrontmatter(data []byte, relPath string, mapping map[string]string, tags map[string]bool) (Item, error) {
	fm, body, ok := splitFrontmatterMap(data)
	if !ok {
		fm = map[string]interface{}{}
		body = string(data)
	}
	return buildItem(fm, body, relPath, mapping, tags)
}

// splitFrontmatterMap: 剥离开头的 "---...---" / "+++...+++" YAML 块
func splitFrontmatterMap(content []byte) (map[string]interface{}, string, bool) {
	trimmed := bytes.TrimLeft(content, " \t\r\n")
	if len(trimmed) == 0 {
		return nil, "", false
	}
	if bytes.HasPrefix(trimmed, []byte("---")) {
		rest := bytes.TrimLeft(trimmed[3:], " \t")
		if bytes.HasPrefix(rest, []byte("\r\n")) {
			rest = rest[2:]
		} else if bytes.HasPrefix(rest, []byte("\n")) {
			rest = rest[1:]
		}
		if end := bytes.Index(rest, []byte("\n---")); end >= 0 {
			var fm map[string]interface{}
			if err := yaml.Unmarshal(rest[:end], &fm); err != nil {
				return nil, string(content), false
			}
			return fm, string(bytes.TrimLeft(rest[end+4:], " \t\r\n")), true
		}
		return nil, string(content), false
	}
	if bytes.HasPrefix(trimmed, []byte("+++")) {
		rest := bytes.TrimLeft(trimmed[3:], " \t")
		if bytes.HasPrefix(rest, []byte("\r\n")) {
			rest = rest[2:]
		} else if bytes.HasPrefix(rest, []byte("\n")) {
			rest = rest[1:]
		}
		if end := bytes.Index(rest, []byte("\n+++")); end >= 0 {
			var fm map[string]interface{}
			if err := yaml.Unmarshal(rest[:end], &fm); err != nil {
				return nil, string(content), false
			}
			return fm, string(bytes.TrimLeft(rest[end+4:], " \t\r\n")), true
		}
	}
	return nil, string(content), false
}

func extractText(data []byte, relPath string, mapping map[string]string) (Item, error) {
	it := Item{ID: relPath, Path: relPath, Fields: make(map[string]string, len(mapping))}
	for field, p := range mapping {
		if p == pathText || p == pathBody {
			it.Fields[field] = string(data)
			continue
		}
		it.Fields[field] = ""
	}
	return it, nil
}

func extractCSV(data []byte, relPath string, mapping map[string]string, tags map[string]bool) ([]Item, error) {
	r := csv.NewReader(bytes.NewReader(data))
	r.FieldsPerRecord = -1
	rows, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("csv is empty")
	}
	header := rows[0]
	var out []Item
	for i := 1; i < len(rows); i++ {
		it := Item{ID: fmt.Sprintf("%s#%d", relPath, i), Path: relPath, Fields: make(map[string]string, len(mapping))}
		for field, col := range mapping {
			idx := headerIndex(header, col)
			var cell string
			if idx >= 0 && idx < len(rows[i]) {
				cell = rows[i][idx]
			}
			if tags[field] {
				it.Tags = valueToTags(cell)
				continue
			}
			it.Fields[field] = cell
		}
		out = append(out, it)
	}
	return out, nil
}

func headerIndex(header []string, name string) int {
	if i, err := strconv.Atoi(name); err == nil && i >= 0 {
		return i
	}
	for i, h := range header {
		if strings.TrimSpace(h) == name {
			return i
		}
	}
	return -1
}

func extractNDJSON(data []byte, relPath string, mapping map[string]string, tags map[string]bool) ([]Item, error) {
	var out []Item
	sc := bufio.NewScanner(bytes.NewReader(data))
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	lineNo := 0
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		lineNo++
		var root interface{}
		if err := json.Unmarshal(line, &root); err != nil {
			return nil, fmt.Errorf("line %d: %w", lineNo, err)
		}
		it, err := buildItem(root, "", fmt.Sprintf("%s#%d", relPath, lineNo), mapping, tags)
		if err != nil {
			return nil, err
		}
		it.Path = relPath
		out = append(out, it)
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

func extractXLSX(data []byte, relPath string, mapping map[string]string, tags map[string]bool) ([]Item, error) {
	rows, err := readXLSXRows(data)
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("xlsx has no rows")
	}
	header := rows[0]
	var out []Item
	for i := 1; i < len(rows); i++ {
		it := Item{ID: fmt.Sprintf("%s#%d", relPath, i), Path: relPath, Fields: make(map[string]string, len(mapping))}
		for field, col := range mapping {
			idx := headerIndex(header, col)
			var cell string
			if idx >= 0 && idx < len(rows[i]) {
				cell = rows[i][idx]
			}
			if tags[field] {
				it.Tags = valueToTags(cell)
				continue
			}
			it.Fields[field] = cell
		}
		out = append(out, it)
	}
	return out, nil
}

type xlsxSharedStrings struct {
	SI []struct {
		T []string `xml:"t"`
	} `xml:"si"`
}

type xlsxSheet struct {
	Rows []struct {
		Cells []struct {
			T  string `xml:"t,attr"`
			V  string `xml:"v"`
			IS struct {
				T string `xml:"t"`
			} `xml:"is"`
		} `xml:"c"`
	} `xml:"row"`
}

func readXLSXRows(data []byte) ([][]string, error) {
	zr, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		return nil, fmt.Errorf("open xlsx: %w", err)
	}
	shared := []string{}
	for _, f := range zr.File {
		if f.Name != "xl/sharedStrings.xml" {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return nil, err
		}
		b, _ := io.ReadAll(rc)
		rc.Close()
		var ss xlsxSharedStrings
		if xml.Unmarshal(b, &ss) == nil {
			for _, si := range ss.SI {
				shared = append(shared, strings.Join(si.T, ""))
			}
		}
	}
	for _, f := range zr.File {
		if f.Name != "xl/worksheets/sheet1.xml" {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return nil, err
		}
		b, _ := io.ReadAll(rc)
		rc.Close()
		var sheet xlsxSheet
		if err := xml.Unmarshal(b, &sheet); err != nil {
			return nil, err
		}
		var rows [][]string
		for _, r := range sheet.Rows {
			cells := make([]string, 0, len(r.Cells))
			for _, c := range r.Cells {
				switch c.T {
				case "s":
					if i, err := strconv.Atoi(c.V); err == nil && i >= 0 && i < len(shared) {
						cells = append(cells, shared[i])
					} else {
						cells = append(cells, "")
					}
				case "inlineStr":
					cells = append(cells, c.IS.T)
				default:
					cells = append(cells, c.V)
				}
			}
			rows = append(rows, cells)
		}
		return rows, nil
	}
	return nil, fmt.Errorf("xlsx: no sheet1.xml")
}

// ------------------------------------------------------------------ 值 → 字符串

func stringifyValue(v interface{}) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return t
	case bool:
		if t {
			return "true"
		}
		return "false"
	case float64:
		return strconv.FormatFloat(t, 'f', -1, 64)
	case int:
		return strconv.Itoa(t)
	case int64:
		return strconv.FormatInt(t, 10)
	case json.Number:
		return t.String()
	case []interface{}:
		parts := make([]string, 0, len(t))
		for _, e := range t {
			parts = append(parts, stringifyValue(e))
		}
		return strings.Join(parts, " ")
	default:
		b, err := json.Marshal(t)
		if err == nil {
			return string(b)
		}
		return fmt.Sprint(v)
	}
}

func valueToTags(v interface{}) []string {
	switch t := v.(type) {
	case []interface{}:
		var out []string
		for _, e := range t {
			if s := strings.TrimSpace(stringifyValue(e)); s != "" {
				out = append(out, s)
			}
		}
		return out
	case string:
		return strings.Fields(t)
	}
	return nil
}
