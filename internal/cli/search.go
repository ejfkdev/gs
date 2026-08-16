// search 子命令: 查询预构建索引。
//
// 三种模式:
//   - 常规搜索: gs search --index <dir> [flags] "<query>"
//   - schema 查看: gs search --index <dir> --list-schema
//   - 快速搜索: gs search --index <dir> --doc <file> (纯 BM25, 把文件当 query)
//
// query 来源优先级: positional > -q > stdin (管道输入)。
// 输出默认 JSON, -human 为人类可读格式。

package cli

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/ejfkdev/gs"
)

// searchOptions: search 子命令的全部 flag
type searchOptions struct {
	indexDir   string
	fields     string
	listSchema bool
	k          int
	strict     bool
	docPath    string
	qText      string
	human      bool
	fullDesc   bool
}

const searchUsage = `Usage: gs search --index <dir> [flags] [<query>]

Query a prebuilt index. Output is JSON by default; pass -human for a
friendly format. The query may be a positional argument, -q, or stdin.

Flags:
  --index <dir>      index directory (required)
  --fields <list>    restrict search to comma-separated field names
  --list-schema      print the schema and exit
  -k <int>           top-K results (default 10)
  --strict           force strict mode (exact boost for IP / domain / hash tokens)
  --doc <file>       fast BM25 search using the file's content as the query
  -q <text>          query text (alternative to the positional argument)
  -human             human-readable output (default: JSON)
  -full-desc         replace the snippet with the full description field

Examples:
  gs search --index ./indexes/wiki -q "nginx 配置" -k 5
  gs search --index ./indexes/wiki "deployment guide" --strict
  gs search --index ./indexes/wiki --fields name,description --list-schema
  echo "golang 并发" | gs search --index ./indexes/wiki
  gs search --index ./indexes/wiki --doc /path/to/long_text.txt
`

func runSearch(args []string, stdout, stderr io.Writer) error {
	opts, positional, err := parseSearchArgs(args)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			fmt.Fprint(stdout, searchUsage)
			return nil
		}
		fmt.Fprintf(stderr, "gs search: %v\n\n%s", err, searchUsage)
		return err
	}

	// query: positional > -q > stdin
	query := strings.TrimSpace(positional)
	if query == "" {
		query = strings.TrimSpace(opts.qText)
	}
	if query == "" && opts.docPath == "" && !opts.listSchema {
		data, rerr := io.ReadAll(os.Stdin)
		if rerr != nil {
			return fmt.Errorf("read stdin: %w", rerr)
		}
		query = strings.TrimSpace(string(data))
	}
	return runSearchMode(opts, query, stdout)
}

// parseSearchArgs: 解析 flag + 位置参数。
// stdlib flag 遇到第一个非 flag 就停, 所以这里把 flag 提前重排,
// 使 "gs search <query> --index x" 和 "gs search --index x <query>" 都可用。
func parseSearchArgs(args []string) (searchOptions, string, error) {
	for _, a := range args {
		if a == "-h" || a == "--help" {
			return searchOptions{}, "", flag.ErrHelp
		}
	}
	args = reorderArgs(args)

	fs := flag.NewFlagSet("gs search", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.Usage = func() {}

	var opts searchOptions
	fs.StringVar(&opts.indexDir, "index", "", "index directory (required)")
	fs.StringVar(&opts.fields, "fields", "", "restrict search to comma-separated field names")
	fs.BoolVar(&opts.listSchema, "list-schema", false, "print the schema and exit")
	fs.IntVar(&opts.k, "k", 10, "top-K results")
	fs.BoolVar(&opts.strict, "strict", false, "force strict mode (exact boost for IP / domain / hash tokens)")
	fs.StringVar(&opts.docPath, "doc", "", "fast BM25 search using the file's content as the query")
	fs.StringVar(&opts.qText, "q", "", "query text (alternative to positional argument)")
	fs.BoolVar(&opts.human, "human", false, "human-readable output (default JSON)")
	fs.BoolVar(&opts.fullDesc, "full-desc", false, "use the full description field in place of the snippet")

	if err := fs.Parse(args); err != nil {
		return opts, "", err
	}
	positional := ""
	if rest := fs.Args(); len(rest) > 0 {
		positional = strings.Join(rest, " ")
	}
	return opts, positional, nil
}

// valueFlags: 需要消费下一个参数的 flag (bool flag 不在内)
var valueFlags = map[string]bool{
	"--index": true, "--fields": true, "--k": true,
	"--doc": true, "--q": true,
	"-index": true, "-fields": true, "-k": true,
	"-doc": true, "-q": true,
}

// reorderArgs: 把 flag (及其值) 全部提前, 让 stdlib flag 能解析
// 位置参数在前、flag 在后的写法, 如 "gs search <query> --index x"。
func reorderArgs(args []string) []string {
	var flags, pos []string
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "--" {
			pos = append(pos, args[i+1:]...)
			break
		}
		if strings.HasPrefix(a, "-") && a != "-" {
			flags = append(flags, a)
			// 只有 "--flag value" 形式需要把后一个参数一并前移;
			// "--flag=value" 自带值, 不动下一个参数
			if !strings.Contains(a, "=") && valueFlags[a] && i+1 < len(args) {
				flags = append(flags, args[i+1])
				i++
			}
		} else {
			pos = append(pos, a)
		}
	}
	return append(flags, pos...)
}

// runSearchMode: 选择模式执行
func runSearchMode(opts searchOptions, query string, stdout io.Writer) error {
	if opts.indexDir == "" {
		return errors.New("--index is required")
	}
	info, err := os.Stat(opts.indexDir)
	if err != nil {
		return fmt.Errorf("--index: %w", err)
	}
	if !info.IsDir() {
		return fmt.Errorf("--index %q is not a directory", opts.indexDir)
	}

	eng, err := gs.Load(opts.indexDir)
	if err != nil {
		return fmt.Errorf("load index: %w", err)
	}
	defer eng.Close()

	// 模式 1: --list-schema
	if opts.listSchema {
		printSchema(stdout, eng.Schema(), opts.human)
		return nil
	}

	// 模式 2: --doc <file>
	if opts.docPath != "" {
		if query != "" {
			return errors.New("--doc and a query (positional / -q / stdin) are mutually exclusive")
		}
		return runFastSearch(stdout, eng, opts)
	}

	// 模式 3: 常规 Search
	if query == "" {
		return errors.New("query is required (positional argument, -q, or stdin)")
	}
	return runRegularSearch(stdout, eng, opts, query)
}

// runRegularSearch: hybrid Search 路径
func runRegularSearch(stdout io.Writer, eng *gs.Engine, opts searchOptions, query string) error {
	fields, err := resolveFieldFilter(eng.Schema(), opts.fields)
	if err != nil {
		return err
	}
	ctx := context.Background()
	hits, err := eng.Search(ctx, gs.SearchOptions{
		Query:  query,
		Fields: fields,
		TopK:   opts.k,
		Strict: opts.strict,
	})
	if err != nil {
		return fmt.Errorf("search: %w", err)
	}
	if opts.human {
		printHuman(stdout, query, hits, opts.fullDesc)
	} else {
		printJSON(stdout, hits, opts.fullDesc)
	}
	return nil
}

// runFastSearch: --doc <file> 路径 (纯 BM25)
func runFastSearch(stdout io.Writer, eng *gs.Engine, opts searchOptions) error {
	data, err := os.ReadFile(opts.docPath)
	if err != nil {
		return fmt.Errorf("read --doc: %w", err)
	}
	text := string(data)
	if strings.TrimSpace(text) == "" {
		return fmt.Errorf("--doc %q is empty", opts.docPath)
	}
	results := eng.FastSearch(text, opts.k)
	if opts.human {
		printFastHuman(stdout, opts.docPath, results)
	} else {
		printFastJSON(stdout, results)
	}
	return nil
}

// ------------------------------------------------------------------ field resolution

// resolveFieldFilter: 解析逗号分隔的 --fields, 按 schema 校验
func resolveFieldFilter(schema *gs.Schema, raw string) ([]string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	known := make(map[string]bool)
	if schema != nil {
		for _, f := range schema.Fields {
			known[f.Name] = true
		}
	}
	var out []string
	for _, p := range strings.Split(raw, ",") {
		name := strings.TrimSpace(p)
		if name == "" {
			continue
		}
		if !known[name] {
			return nil, fmt.Errorf("unknown field %q (available: %s)",
				name, strings.Join(fieldNames(schema), ", "))
		}
		out = append(out, name)
	}
	return out, nil
}

// fieldNames: schema 顺序的字段名 (after 用户看到确定性列表)
func fieldNames(schema *gs.Schema) []string {
	if schema == nil {
		return nil
	}
	names := make([]string, len(schema.Fields))
	for i, f := range schema.Fields {
		names[i] = f.Name
	}
	return names
}

// ------------------------------------------------------------------ JSON output

// hitJSON: 输出的命中结构 (Full snippet 用 --full-desc 覆盖)
type hitJSON struct {
	ID      string            `json:"id"`
	Score   float32           `json:"score"`
	Path    string            `json:"path"`
	Source  string            `json:"source"`
	Tags    []string          `json:"tags,omitempty"`
	Fields  map[string]string `json:"fields"`
	Snippet string            `json:"snippet"`
}

func toJSONHit(h gs.Hit, fullDesc bool) hitJSON {
	snippet := h.Snippet
	if fullDesc {
		if v, ok := h.Fields["description"]; ok && v != "" {
			snippet = v
		} else if v, ok := h.Fields["name"]; ok && v != "" {
			snippet = v
		}
	}
	return hitJSON{
		ID:      h.ID,
		Score:   h.Score,
		Path:    h.Path,
		Source:  h.Source,
		Tags:    h.Tags,
		Fields:  h.Fields,
		Snippet: snippet,
	}
}

func printJSON(w io.Writer, hits []gs.Hit, fullDesc bool) {
	out := make([]hitJSON, 0, len(hits))
	for _, h := range hits {
		out = append(out, toJSONHit(h, fullDesc))
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(out)
}

// ------------------------------------------------------------------ human output

func printHuman(w io.Writer, query string, hits []gs.Hit, fullDesc bool) {
	fmt.Fprintf(w, "=== Query: %q ===\n", query)
	if len(hits) == 0 {
		fmt.Fprintln(w, "  (no results)")
		return
	}
	for i, h := range hits {
		fmt.Fprintf(w, "  %d. [%s] %.3f  %s\n", i+1, h.Source, h.Score, displayName(h))
		fmt.Fprintf(w, "     path: %s\n", h.Path)
		for _, k := range sortedFieldKeys(h.Fields) {
			v := h.Fields[k]
			if v == "" {
				continue
			}
			fmt.Fprintf(w, "     %s: %s\n", k, truncate(v, 200))
		}
		snip := h.Snippet
		if fullDesc {
			if v, ok := h.Fields["description"]; ok && v != "" {
				snip = v
			} else if v, ok := h.Fields["name"]; ok && v != "" {
				snip = v
			}
		}
		if snip != "" {
			fmt.Fprintf(w, "     snippet: %s\n", truncate(snip, 400))
		}
	}
}

// sortedFieldKeys: 稳定输出 (map 遍历顺序不确定)
func sortedFieldKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	// insertion sort: 字段数一般 <10
	for i := 1; i < len(keys); i++ {
		for j := i; j > 0 && keys[j-1] > keys[j]; j-- {
			keys[j-1], keys[j] = keys[j], keys[j-1]
		}
	}
	return keys
}

func displayName(h gs.Hit) string {
	if v, ok := h.Fields["name"]; ok && v != "" {
		return v
	}
	if h.ID != "" {
		return h.ID
	}
	return h.Path
}

func truncate(s string, n int) string {
	rs := []rune(s)
	if len(rs) <= n {
		return s
	}
	return string(rs[:n]) + "…"
}

// ------------------------------------------------------------------ schema listing

func printSchema(w io.Writer, schema *gs.Schema, human bool) {
	if !human {
		_ = json.NewEncoder(w).Encode(schema)
		return
	}
	name := ""
	if schema != nil {
		name = schema.Name
	}
	fmt.Fprintf(w, "=== Schema: %s ===\n", name)
	if schema == nil {
		fmt.Fprintln(w, "  (no schema)")
		return
	}
	for _, f := range schema.Fields {
		fmt.Fprintf(w, "  %-15s %-9s srch=%-5v emb=%-5v disp=%-5v snip=%-5v strict=%-5v w=%.1f\n",
			f.Name, f.Type.String(), f.Searchable, f.Embeddable, f.Display, f.Snippet, f.Strict, f.FieldWeight)
	}
}

// ------------------------------------------------------------------ fast search output

type fastHitJSON struct {
	Idx     int     `json:"idx"`
	Path    string  `json:"path"`
	Source  string  `json:"source"`
	Name    string  `json:"name"`
	Desc    string  `json:"desc,omitempty"`
	Score   float32 `json:"score"`
	Snippet string  `json:"snippet"`
}

func printFastJSON(w io.Writer, results []gs.FastSearchResult) {
	out := make([]fastHitJSON, 0, len(results))
	for _, r := range results {
		out = append(out, fastHitJSON{
			Idx:     r.Idx,
			Path:    r.Path,
			Source:  r.Source,
			Name:    r.Name,
			Desc:    r.Desc,
			Score:   r.Score,
			Snippet: r.Snippet,
		})
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(out)
}

func printFastHuman(w io.Writer, docPath string, results []gs.FastSearchResult) {
	fmt.Fprintf(w, "=== Fast search (--doc %s) ===\n", docPath)
	if len(results) == 0 {
		fmt.Fprintln(w, "  (no results)")
		return
	}
	for i, r := range results {
		fmt.Fprintf(w, "  %d. [%s] %.3f  %s\n", i+1, r.Source, r.Score, orDefault(r.Name, r.Path))
		fmt.Fprintf(w, "     path: %s\n", r.Path)
		if r.Desc != "" {
			fmt.Fprintf(w, "     desc: %s\n", truncate(r.Desc, 300))
		}
		if r.Snippet != "" {
			fmt.Fprintf(w, "     snippet: %s\n", truncate(r.Snippet, 400))
		}
	}
}

func orDefault(s, fallback string) string {
	if s != "" {
		return s
	}
	return fallback
}
