// build 子命令: 把 skills/wiki 语料提取成 Item, 并构建混合索引。
//
// 两个 extractor:
//   - skills: 遍历 <source>/<user>/<skill>/SKILL.md, 解析 YAML frontmatter,
//     产出 (name, description, tags, category, full_content)
//   - wiki: 遍历 <source>/**/*.md (跳过 README/LICENSE/CHANGELOG/CONTRIBUTING),
//     产出 (name, category, content)
//
// markdown 正文全文索引, 不截断。

package cli

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/ejfkdev/gs"
	"gopkg.in/yaml.v3"
)

// indexKind: 选择 schema + extractor
type indexKind string

const (
	kindSkills indexKind = "skills"
	kindWiki   indexKind = "wiki"
)

func (k indexKind) Valid() bool { return k == kindSkills || k == kindWiki }

// skillsSchema / wikiSchema: CLI 内置的两种语料格式对应的字段定义。
// 库本身不内置 schema; 这些只服务于 gs build 的 <skills|wiki> 两种
// extractor。构建后 schema 会写成索引目录里的 schema.json, 检索时
// Load 自动读回, 不再依赖这里。
var (
	// skills 语料: name(5x), description(2x), tags(3x), category(1x), full_content(1x)
	skillsSchema = gs.Schema{
		Name: "skills",
		Fields: []gs.Field{
			{Name: "name", Type: gs.FieldText, Searchable: true, Embeddable: true, FieldWeight: 5.0, Display: true, Snippet: false, Strict: true},
			{Name: "description", Type: gs.FieldLongText, Searchable: true, Embeddable: true, FieldWeight: 2.0, Display: true, Snippet: true, Strict: true},
			{Name: "tags", Type: gs.FieldTags, Searchable: true, Embeddable: true, FieldWeight: 3.0, Display: true, Snippet: false, Strict: false},
			{Name: "category", Type: gs.FieldText, Searchable: true, Embeddable: true, FieldWeight: 1.0, Display: true, Snippet: false, Strict: false},
			{Name: "full_content", Type: gs.FieldLongText, Searchable: true, Embeddable: false, FieldWeight: 1.0, Display: false, Snippet: true, Strict: false},
		},
	}

	// wiki 语料: name(3x), category(1x), content(1x, 全文)
	wikiSchema = gs.Schema{
		Name: "wiki",
		Fields: []gs.Field{
			{Name: "name", Type: gs.FieldText, Searchable: true, Embeddable: true, FieldWeight: 3.0, Display: true, Snippet: false, Strict: true},
			{Name: "category", Type: gs.FieldText, Searchable: true, Embeddable: true, FieldWeight: 1.0, Display: true, Snippet: false, Strict: false},
			{Name: "content", Type: gs.FieldLongText, Searchable: true, Embeddable: true, FieldWeight: 1.0, Display: true, Snippet: true, Strict: true},
		},
	}
)

// buildOptions: build 子命令的全部 flag
type buildOptions struct {
	source        string
	output        string
	bgeWeights    string
	bgeVocab      string
	dryRun        bool
	verbose       bool
	maxEmbedRunes int
}

const buildUsage = `Usage: gs build <skills|wiki> [flags]

Build a hybrid (BM25 + BGE) index from a source tree.

Arguments:
  skills    Claude-style skills corpus: <source>/<user>/<skill>/SKILL.md
  wiki      wiki corpus: <source>/**/*.md (README/LICENSE/CHANGELOG skipped)

Flags:
  --source <dir>            source root directory (required)
  --output <dir>            output index directory (required)
  --bge-weights <file>      BGE 权重源文件 (HF model.safetensors)
  --bge-vocab <file>        BGE 词表源文件 (HF vocab.txt; 复制进索引目录为 vocab.txt)
  --max-embed-runes <int>   BGE 每字段编码截断的最大 rune 数 (0 = 默认 512)
  --dry-run                 walk + extract only; skip embedding and output
  -v                        verbose per-file logging

  --bge-weights / --bge-vocab 只需首次构建时指定一次 (把模型"种"进
  --output); 之后构建和搜索都从索引目录按固定名 model.safetensors/vocab.txt
  自动读取, 无需再传。权重唯一支持的格式是 HuggingFace 的 model.safetensors
  (pytorch_model.bin 是 pickle, 不支持)。详见 MODEL.md。

Examples:
  gs build wiki --source ./wiki --output ./indexes/wiki \
      --bge-weights ./model/model.safetensors --bge-vocab ./model/vocab.txt
  gs build skills --source ./skills --output ./indexes/skills \
      --bge-weights ./model/model.safetensors --bge-vocab ./model/vocab.txt
`

func runBuild(args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		return errors.New("missing argument: expected 'skills' or 'wiki' (run \"gs build -h\" for usage)")
	}
	switch args[0] {
	case "-h", "--help", "help":
		fmt.Fprint(stdout, buildUsage)
		return nil
	}
	kind := indexKind(args[0])
	if !kind.Valid() {
		fmt.Fprintf(stderr, "gs build: unknown kind %q (want 'skills' or 'wiki')\n\n%s", args[0], buildUsage)
		return fmt.Errorf("unknown index kind %q", args[0])
	}

	fs := flag.NewFlagSet("gs build", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.Usage = func() { fmt.Fprint(stderr, buildUsage) }

	var opts buildOptions
	fs.StringVar(&opts.source, "source", "", "source root directory (required)")
	fs.StringVar(&opts.output, "output", "", "output index directory (required)")
	fs.StringVar(&opts.bgeWeights, "bge-weights", "", "BGE 权重源文件 (HF model.safetensors)")
	fs.StringVar(&opts.bgeVocab, "bge-vocab", "", "BGE 词表源文件 (HF vocab.txt)")
	fs.IntVar(&opts.maxEmbedRunes, "max-embed-runes", 0, "BGE 每字段编码截断的最大 rune 数 (0 = 默认 512)")
	fs.BoolVar(&opts.dryRun, "dry-run", false, "walk + extract only; skip embedding and output")
	fs.BoolVar(&opts.verbose, "v", false, "verbose per-file logging")
	if err := fs.Parse(args[1:]); err != nil {
		return err
	}
	if err := opts.validate(); err != nil {
		return err
	}
	return opts.run(kind, stderr)
}

// validate: 校验 flag 组合
func (o *buildOptions) validate() error {
	if o.source == "" {
		return errors.New("--source is required")
	}
	if o.output == "" {
		return errors.New("--output is required")
	}
	if info, err := os.Stat(o.source); err != nil {
		return fmt.Errorf("--source: %w", err)
	} else if !info.IsDir() {
		return fmt.Errorf("--source %q is not a directory", o.source)
	}
	if !o.dryRun {
		if o.bgeWeights == "" || o.bgeVocab == "" {
			return errors.New("--bge-weights and --bge-vocab are required (or pass --dry-run)")
		}
		if _, err := os.Stat(o.bgeWeights); err != nil {
			return fmt.Errorf("--bge-weights: %w", err)
		}
		if _, err := os.Stat(o.bgeVocab); err != nil {
			return fmt.Errorf("--bge-vocab: %w", err)
		}
	}
	// 提前验证 output 可写, 而不是等 BGE 全算完才发现
	if err := os.MkdirAll(o.output, 0o755); err != nil {
		return fmt.Errorf("--output: %w", err)
	}
	probe := filepath.Join(o.output, ".write-test")
	if err := os.WriteFile(probe, nil, 0o644); err != nil {
		return fmt.Errorf("--output not writable: %w", err)
	}
	os.Remove(probe)
	return nil
}

// run: 单次 walk — dry-run 只统计; 否则在读文件的同时直接 Add 到 builder
func (o *buildOptions) run(kind indexKind, stderr io.Writer) error {
	var (
		schema  *gs.Schema
		extract func(absPath, relPath string) (gs.Item, error)
	)
	switch kind {
	case kindSkills:
		schema = &skillsSchema
		extract = extractSkill
	case kindWiki:
		schema = &wikiSchema
		extract = extractWiki
	}
	// --max-embed-runes 覆盖 schema 的截断长度 (拷贝一份, 不改包级共享变量)
	if o.maxEmbedRunes > 0 {
		s := *schema
		s.MaxEmbedRunes = o.maxEmbedRunes
		schema = &s
	}

	fmt.Fprintf(stderr, "[gs] kind=%s source=%s output=%s\n", kind, o.source, o.output)
	t0 := time.Now()

	opts := []gs.BuilderOption{gs.WithProgress(o.progress(stderr))}
	if !o.dryRun {
		opts = append(opts, gs.WithBGEPaths(o.bgeWeights, o.bgeVocab))
	}
	builder, err := gs.NewBuilder(schema, o.output, opts...)
	if err != nil {
		return fmt.Errorf("new builder: %w", err)
	}
	defer builder.Close()

	var (
		count   int
		skipped int
		errs    int
	)
	walkErr := filepath.WalkDir(o.source, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			fmt.Fprintf(stderr, "[gs] walk error at %s: %v (skipping)\n", p, walkErr)
			return nil
		}
		if d.IsDir() {
			return nil
		}
		if !shouldIndex(kind, p, d) {
			skipped++
			return nil
		}
		rel, rerr := filepath.Rel(o.source, p)
		if rerr != nil {
			rel = p
		}
		item, err := extract(p, rel)
		if err != nil {
			errs++
			fmt.Fprintf(stderr, "[gs] extract error %s: %v\n", rel, err)
			return nil
		}
		if o.verbose {
			fmt.Fprintf(stderr, "[gs] + %s\n", rel)
		}
		count++
		if o.dryRun {
			return nil
		}
		if err := builder.Add(item); err != nil {
			return fmt.Errorf("add %s: %w", rel, err)
		}
		return nil
	})
	if walkErr != nil {
		return fmt.Errorf("walk: %w", walkErr)
	}

	fmt.Fprintf(stderr, "[gs] scanned %d items, skipped %d, errors %d in %v\n",
		count, skipped, errs, time.Since(t0).Round(time.Millisecond))

	if o.dryRun {
		fmt.Fprintf(stderr, "[gs] --dry-run set: index not built\n")
		return nil
	}
	if count == 0 {
		return errors.New("no documents found under --source")
	}

	fmt.Fprintf(stderr, "[gs] building index for %d items (BGE encoding may take a while)...\n", count)
	t1 := time.Now()
	if err := builder.Build(); err != nil {
		return fmt.Errorf("build: %w", err)
	}
	fmt.Fprintf(stderr, "[gs] build done in %v → %s\n", time.Since(t1).Round(time.Millisecond), o.output)

	if sz, err := sizeOfDir(o.output); err == nil {
		fmt.Fprintf(stderr, "[gs] index payload: %s (%d files)\n", humanBytes(sz), countFiles(o.output))
	}
	return nil
}

// progress: 把 Builder 的进度回调转成 CLI 输出 (节流)
func (o *buildOptions) progress(stderr io.Writer) func(stage string, cur, total int) {
	last := struct {
		stage string
		cur   int
	}{}
	return func(stage string, cur, total int) {
		if stage != last.stage || cur-last.cur >= 500 || cur == total {
			fmt.Fprintf(stderr, "[gs] build %-15s %d/%d\n", stage, cur, total)
			last.stage = stage
			last.cur = cur
		}
	}
}

// ------------------------------------------------------------------ file filter

// shouldIndex: 廉价预过滤。
//   - skills: 只接受文件名为 SKILL.md 的文件
//   - wiki: 接受 .md, 跳过元数据文件 (README/LICENSE/CHANGELOG/CONTRIBUTING)
func shouldIndex(kind indexKind, absPath string, d fs.DirEntry) bool {
	name := d.Name()
	switch kind {
	case kindSkills:
		return name == "SKILL.md"
	case kindWiki:
		return strings.EqualFold(path.Ext(name), ".md") && !isMetaFile(name)
	}
	return false
}

// isMetaFile: README/LICENSE/CHANGELOG/CONTRIBUTING (任意大小写, 允许后缀)
func isMetaFile(name string) bool {
	upper := strings.ToUpper(strings.TrimSuffix(name, path.Ext(name)))
	switch {
	case upper == "README",
		upper == "LICENSE",
		upper == "CHANGELOG",
		upper == "CONTRIBUTING",
		strings.HasPrefix(upper, "LICENSE-"),
		strings.HasPrefix(upper, "CHANGELOG-"),
		strings.HasPrefix(upper, "CONTRIBUTING-"):
		return true
	}
	return false
}

// ------------------------------------------------------------------ extractors

// frontmatter: 共享字段, 其他字段静默容忍
type frontmatter struct {
	Name        string   `yaml:"name"`
	Description string   `yaml:"description"`
	Tags        []string `yaml:"tags"`
	Category    string   `yaml:"category"`
}

// splitFrontmatter: 摘掉开头的 "---...---" 或 "+++...+++" 块。
// 返回 (fm, body, ok)。没有可识别的 frontmatter 时 ok=false, 全文作为 body。
func splitFrontmatter(content []byte) (frontmatter, string, bool) {
	trimmed := bytes.TrimLeft(content, " \t\r\n")
	if len(trimmed) == 0 {
		return frontmatter{}, "", false
	}
	// YAML --- ... ---
	if bytes.HasPrefix(trimmed, []byte("---")) {
		rest := trimmed[3:]
		rest = bytes.TrimLeft(rest, " \t")
		if bytes.HasPrefix(rest, []byte("\r\n")) {
			rest = rest[2:]
		} else if bytes.HasPrefix(rest, []byte("\n")) {
			rest = rest[1:]
		}
		if end := bytes.Index(rest, []byte("\n---")); end >= 0 {
			fm := frontmatter{}
			body := bytes.TrimLeft(rest[end+4:], " \t\r\n")
			if err := yaml.Unmarshal(rest[:end], &fm); err != nil {
				// YAML 解析失败 → 全文当 body, 无 fm
				return frontmatter{}, string(content), false
			}
			return fm, string(body), true
		}
		// --- 无闭合 → 按普通正文处理
		return frontmatter{}, string(content), false
	}
	// TOML +++ ... +++ (字段风格与 YAML 近似, yaml 解析简单 TOML 足够)
	if bytes.HasPrefix(trimmed, []byte("+++")) {
		rest := trimmed[3:]
		rest = bytes.TrimLeft(rest, " \t")
		if bytes.HasPrefix(rest, []byte("\r\n")) {
			rest = rest[2:]
		} else if bytes.HasPrefix(rest, []byte("\n")) {
			rest = rest[1:]
		}
		if end := bytes.Index(rest, []byte("\n+++")); end >= 0 {
			fm := frontmatter{}
			body := bytes.TrimLeft(rest[end+4:], " \t\r\n")
			if err := yaml.Unmarshal(rest[:end], &fm); err == nil {
				return fm, string(body), true
			}
			return frontmatter{}, string(body), false
		}
	}
	return frontmatter{}, string(content), false
}

// extractSkill: SKILL.md → skills schema 的 Item
func extractSkill(absPath, relPath string) (gs.Item, error) {
	raw, err := os.ReadFile(absPath)
	if err != nil {
		return gs.Item{}, err
	}
	fm, body, _ := splitFrontmatter(raw)

	if fm.Name == "" {
		// dir/SKILL.md → 用父目录名当 name
		_, fm.Name = path.Split(path.Dir(relPath))
	}

	return gs.Item{
		ID:     relPath,
		Path:   relPath,
		Source: "skill",
		Tags:   fm.Tags, // 标签只放这里, Builder 会合并进 FieldTags 字段
		Fields: map[string]string{
			"name":         fm.Name,
			"description":  fm.Description,
			"category":     fm.Category,
			"full_content": body,
		},
	}, nil
}

// extractWiki: wiki .md → wiki schema 的 Item
func extractWiki(absPath, relPath string) (gs.Item, error) {
	raw, err := os.ReadFile(absPath)
	if err != nil {
		return gs.Item{}, err
	}
	_, body, _ := splitFrontmatter(raw)

	// name = 文件名去 .md; category = 源根以下的父目录
	base := path.Base(relPath)
	name := strings.TrimSuffix(base, path.Ext(base))
	dir := path.Dir(relPath)
	if dir == "." {
		dir = ""
	}

	return gs.Item{
		ID:     relPath,
		Path:   relPath,
		Source: "wiki",
		Fields: map[string]string{
			"name":     name,
			"category": dir,
			"content":  body,
		},
	}, nil
}

// ------------------------------------------------------------------ misc helpers

func sizeOfDir(p string) (int64, error) {
	var total int64
	err := filepath.WalkDir(p, func(_ string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		info, ierr := d.Info()
		if ierr == nil {
			total += info.Size()
		}
		return nil
	})
	return total, err
}

func countFiles(p string) int {
	n := 0
	_ = filepath.WalkDir(p, func(_ string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			n++
		}
		return nil
	})
	return n
}
