// config_extract.go - 配置驱动的多格式提取
//
// 每个 source 支持一种 format:
//   - json        文件 → JSON 对象 → 按点路径取字段
//   - yaml        文件 → YAML → 按点路径取字段
//   - frontmatter 文件 = YAML frontmatter + 正文 (skill 那种); 正文用 "__body__" 取
//   - text        整文件当单个字段 ("__text__"/"__body__")
//   - csv         CSV 表 (首行表头, 每数据行一个 Item)
//   - xlsx        首个 sheet (首行表头, 每数据行一个 Item, 单元格按字符串)
//
// 所有格式的文件路径用 --config 里的 dir + include (glob) 先匹配一遍。

package cli

import (
	"archive/zip"
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

	"github.com/ejfkdev/gs"
	"gopkg.in/yaml.v3"
)

// 特殊取值路径
const (
	pBody = "__body__" // frontmatter:text 的正文
	pText = "__text__" // text 的整文件内容
)

// extractSourceFile: 一个文件 → 若干 Item (json/yaml/frontmatter/text 通常 1 个,
// csv/xlsx 每个数据行 1 个)。tagsFields = type==tags 的字段名集合。
func extractSourceFile(src SourceConfig, tagsFields map[string]bool, absPath, relPath string) ([]gs.Item, error) {
	data, err := os.ReadFile(absPath)
	if err != nil {
		return nil, err
	}
	var items []gs.Item
	switch src.Format {
	case "json":
		it, err := extractJSON(data, relPath, src.Mapping, tagsFields)
		if err != nil {
			return nil, err
		}
		items = append(items, it)
	case "yaml", "yml":
		it, err := extractYAML(data, relPath, src.Mapping, tagsFields)
		if err != nil {
			return nil, err
		}
		items = append(items, it)
	case "frontmatter", "markdown":
		it, err := extractFrontmatter(data, relPath, src.Mapping, tagsFields)
		if err != nil {
			return nil, err
		}
		items = append(items, it)
	case "text":
		it, err := extractText(data, relPath, src.Mapping)
		if err != nil {
			return nil, err
		}
		items = append(items, it)
	case "csv":
		items, err = extractCSV(data, relPath, src.Mapping, tagsFields)
		if err != nil {
			return nil, err
		}
	case "xlsx":
		items, err = extractXLSX(data, relPath, src.Mapping, tagsFields)
		if err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("unsupported format %q (want json|yaml|frontmatter|text|csv|xlsx)", src.Format)
	}
	return items, nil
}

// ------------------------------------------------------------------ json / yaml

func extractJSON(data []byte, relPath string, mapping map[string]string, tags map[string]bool) (gs.Item, error) {
	var root interface{}
	if err := json.Unmarshal(data, &root); err != nil {
		return gs.Item{}, err
	}
	return buildItem(root, "", relPath, mapping, tags)
}

func extractYAML(data []byte, relPath string, mapping map[string]string, tags map[string]bool) (gs.Item, error) {
	var root interface{}
	if err := yaml.Unmarshal(data, &root); err != nil {
		return gs.Item{}, err
	}
	return buildItem(root, "", relPath, mapping, tags)
}

// buildItem: 按 mapping 从树里取值, 拼成 Item。body 供 frontmatter 用。
func buildItem(root interface{}, body, relPath string, mapping map[string]string, tags map[string]bool) (gs.Item, error) {
	it := gs.Item{ID: relPath, Path: relPath, Fields: make(map[string]string, len(mapping))}
	for field, p := range mapping {
		if p == pBody {
			it.Fields[field] = body
			continue
		}
		if p == pText {
			it.Fields[field] = body
			continue
		}
		v, ok := getPath(root, p)
		if !ok {
			// 取不到 → 空值, 不报错 (容错)
			it.Fields[field] = ""
			continue
		}
		if tags[field] {
			if t := valueToTags(v); t != nil {
				it.Tags = t
				continue // 交给 Builder 合并进 FieldTags 字段, 避免重复
			}
		}
		it.Fields[field] = stringifyValue(v)
	}
	return it, nil
}

// ------------------------------------------------------------------ frontmatter

func extractFrontmatter(data []byte, relPath string, mapping map[string]string, tags map[string]bool) (gs.Item, error) {
	fm, body, ok := splitFrontmatterMap(data)
	if !ok {
		fm = map[string]interface{}{}
		body = string(data)
	}
	return buildItem(fm, body, relPath, mapping, tags)
}

// splitFrontmatterMap: 剥离开头的 "---...---" / "+++...+++" YAML 块, frontmatter 以
// map 形式返回 (与 build.go 里定长版本的 splitFrontmatter 不同, 这里是通用版)。
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

// ------------------------------------------------------------------ text

func extractText(data []byte, relPath string, mapping map[string]string) (gs.Item, error) {
	it := gs.Item{ID: relPath, Path: relPath, Fields: make(map[string]string, len(mapping))}
	for field, p := range mapping {
		if p == pText || p == pBody {
			it.Fields[field] = string(data)
			continue
		}
		it.Fields[field] = ""
	}
	return it, nil
}

// ------------------------------------------------------------------ csv

func extractCSV(data []byte, relPath string, mapping map[string]string, tags map[string]bool) ([]gs.Item, error) {
	r := csv.NewReader(bytes.NewReader(data))
	r.FieldsPerRecord = -1 // 允许行内列数不一致
	rows, err := r.ReadAll()
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("csv is empty")
	}
	header := rows[0]
	var out []gs.Item
	for i := 1; i < len(rows); i++ {
		row := rows[i]
		it := gs.Item{ID: fmt.Sprintf("%s#%d", relPath, i), Path: relPath, Fields: make(map[string]string, len(mapping))}
		for field, col := range mapping {
			idx := headerIndex(header, col)
			var cell string
			if idx >= 0 && idx < len(row) {
				cell = row[idx]
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
	// 允许 "name" 或 "3" (0-based 列下标)
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

// ------------------------------------------------------------------ xlsx (minimal)

// xlsx 只读第一个 sheet, 首行当表头, 单元格统一按字符串取 (数值用其字面量)。
func extractXLSX(data []byte, relPath string, mapping map[string]string, tags map[string]bool) ([]gs.Item, error) {
	rows, err := readXLSXRows(data)
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("xlsx has no rows")
	}
	header := rows[0]
	var out []gs.Item
	for i := 1; i < len(rows); i++ {
		it := gs.Item{ID: fmt.Sprintf("%s#%d", relPath, i), Path: relPath, Fields: make(map[string]string, len(mapping))}
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
				case "s": // shared string 索引
					if i, err := strconv.Atoi(c.V); err == nil && i >= 0 && i < len(shared) {
						cells = append(cells, shared[i])
					} else {
						cells = append(cells, "")
					}
				case "inlineStr":
					cells = append(cells, c.IS.T)
				default: // 数字等: 用 <v> 字面量
					cells = append(cells, c.V)
				}
			}
			rows = append(rows, cells)
		}
		return rows, nil
	}
	return nil, fmt.Errorf("xlsx: no sheet1.xml")
}

// ------------------------------------------------------------------ value helpers

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

// valueToTags: 数组 → []string; 字符串 → 按空白切; 其他 → nil
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

// ------------------------------------------------------------------ glob

// matchGlob: 支持 ** 的 glob 匹配 (path 用 "/" 分隔)
func matchGlob(pattern, name string) bool {
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
