// config_build.go - 配置驱动索引 (gs build --config index.yaml)

package cli

import (
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"time"

	"github.com/ejfkdev/gs"
)

// runConfig: 按 --config 里的 schema + sources 建索引。
func (o *buildOptions) runConfig(stderr io.Writer) error {
	cfg, err := LoadConfig(o.config)
	if err != nil {
		return err
	}
	schema, err := cfg.Schema.Schema()
	if err != nil {
		return fmt.Errorf("schema: %w", err)
	}
	o.applyMaxEmbedRunes(schema)
	needEmb := schemaHasEmbeddable(schema)
	if err := o.validateConfigOutput(needEmb); err != nil {
		return err
	}
	if len(cfg.Sources) == 0 {
		return errors.New("config: sources is required")
	}
	_, err = o.buildConfigTo(cfg, schema, o.output, stderr)
	return err
}

// buildConfigTo: 遍历 sources 提取为 Item 并 Add, 然后把索引 Build 到 destDir
// (dryRun 只统计不落盘)。返回 item 数。
func (o *buildOptions) buildConfigTo(cfg *Config, schema *gs.Schema, destDir string, stderr io.Writer) (int, error) {
	tagsFields := map[string]bool{}
	for _, f := range schema.Fields {
		if f.Type == gs.FieldTags {
			tagsFields[f.Name] = true
		}
	}

	builderOpts := []gs.BuilderOption{gs.WithProgress(o.progress(stderr))}
	if !o.dryRun {
		builderOpts = append(builderOpts, gs.WithBGEPaths(o.bgeWeights, o.bgeVocab))
	}
	builder, err := gs.NewBuilder(schema, destDir, builderOpts...)
	if err != nil {
		return 0, fmt.Errorf("new builder: %w", err)
	}
	defer builder.Close()

	t0 := time.Now()
	var count, errs int

	for si, src := range cfg.Sources {
		if src.Dir == "" {
			return 0, fmt.Errorf("sources[%d].dir is required", si)
		}
		if src.Format == "" {
			return 0, fmt.Errorf("sources[%d].format is required", si)
		}
		include := src.Include
		if include == "" {
			include = "**"
		}
		onError := src.OnError
		if onError == "" {
			onError = "skip"
		}
		if onError != "skip" && onError != "fail" {
			return 0, fmt.Errorf("sources[%d].on_error must be 'skip' or 'fail'", si)
		}

		walkErr := filepath.WalkDir(src.Dir, func(p string, d fs.DirEntry, werr error) error {
			if werr != nil {
				fmt.Fprintf(stderr, "[gs] walk error at %s: %v (skipping)\n", p, werr)
				return nil
			}
			if d.IsDir() {
				return nil
			}
			rel, rerr := filepath.Rel(src.Dir, p)
			if rerr != nil {
				rel = p
			}
			if !matchGlob(include, filepath.ToSlash(rel)) {
				return nil
			}
			items, err := extractSourceFile(src, tagsFields, p, rel)
			if err != nil {
				errs++
				if onError == "fail" {
					return fmt.Errorf("extract %s: %w", rel, err)
				}
				fmt.Fprintf(stderr, "[gs] extract error %s: %v (skipped)\n", rel, err)
				return nil
			}
			for j := range items {
				items[j].Source = schema.Name
				if items[j].ID == "" {
					items[j].ID = rel
				}
				if items[j].Path == "" {
					items[j].Path = rel
				}
				if o.verbose {
					fmt.Fprintf(stderr, "[gs] + %s\n", rel)
				}
				count++
				if o.dryRun {
					continue
				}
				if err := builder.Add(items[j]); err != nil {
					return fmt.Errorf("add %s: %w", rel, err)
				}
			}
			return nil
		})
		if walkErr != nil {
			return 0, fmt.Errorf("walk %s: %w", src.Dir, walkErr)
		}
	}

	fmt.Fprintf(stderr, "[gs] scanned %d items, extract errors %d in %v\n",
		count, errs, time.Since(t0).Round(time.Millisecond))

	if o.dryRun {
		fmt.Fprintf(stderr, "[gs] --dry-run set: index not built\n")
		return count, nil
	}
	if count == 0 {
		return 0, errors.New("no documents matched any source")
	}

	fmt.Fprintf(stderr, "[gs] building index for %d items (BGE encoding may take a while)...\n", count)
	t1 := time.Now()
	if err := builder.Build(); err != nil {
		return 0, fmt.Errorf("build: %w", err)
	}
	fmt.Fprintf(stderr, "[gs] build done in %v → %s\n", time.Since(t1).Round(time.Millisecond), destDir)
	if sz, err := sizeOfDir(destDir); err == nil {
		fmt.Fprintf(stderr, "[gs] index payload: %s (%d files)\n", humanBytes(sz), countFiles(destDir))
	}
	return count, nil
}

// applyMaxEmbedRunes: --max-embed-runes 覆盖 schema
func (o *buildOptions) applyMaxEmbedRunes(schema *gs.Schema) {
	if o.maxEmbedRunes > 0 {
		s := *schema
		s.MaxEmbedRunes = o.maxEmbedRunes
		*schema = s
	}
}

func schemaHasEmbeddable(s *gs.Schema) bool {
	for _, f := range s.Fields {
		if f.Embeddable {
			return true
		}
	}
	return false
}

// validateConfigOutput: 校验 output 与 BGE 路径（只有 schema 含 embeddable 字段时才要求 BGE）
func (o *buildOptions) validateConfigOutput(needEmb bool) error {
	if o.output == "" {
		return errors.New("--output is required")
	}
	if err := os.MkdirAll(o.output, 0o755); err != nil {
		return fmt.Errorf("--output: %w", err)
	}
	probe := filepath.Join(o.output, ".write-test")
	if err := os.WriteFile(probe, nil, 0o644); err != nil {
		return fmt.Errorf("--output not writable: %w", err)
	}
	os.Remove(probe)
	if !o.dryRun && needEmb {
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
	return nil
}
