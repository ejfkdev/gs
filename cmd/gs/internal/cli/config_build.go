// config_build.go - 配置驱动索引 (gs build --config index.yaml)
//
// 这里只是库公开 API (gs.IndexConfig.Build / Scan) 的薄封装, 加上 CLI 的
// 校验、进度/错误打印、dry-run 与 payload 统计。

package cli

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/ejfkdev/gs"
)

// runConfig: 按 --config 建索引
func (o *buildOptions) runConfig(stderr io.Writer) error {
	cfg, err := gs.LoadIndexConfig(o.config)
	if err != nil {
		return err
	}
	if o.maxEmbedRunes > 0 {
		cfg.Schema.MaxEmbedRunes = o.maxEmbedRunes
	}
	schema, err := cfg.Schema.Schema()
	if err != nil {
		return fmt.Errorf("schema: %w", err)
	}
	needEmb := false
	for _, f := range schema.Fields {
		if f.Embeddable {
			needEmb = true
			break
		}
	}
	if err := o.validateConfigOutput(needEmb); err != nil {
		return err
	}
	if len(cfg.Sources) == 0 {
		return errors.New("config: sources is required")
	}

	opts := []gs.IndexBuildOption{
		gs.IndexWithProgress(o.progress(stderr)),
		gs.IndexWithOnError(func(path string, err error) {
			fmt.Fprintf(stderr, "[gs] extract error %s: %v (skipped)\n", path, err)
		}),
	}
	if !o.dryRun && o.bgeWeights != "" {
		opts = append(opts, gs.IndexWithBGEPaths(o.bgeWeights, o.bgeVocab))
	}
	if o.embCache != "" {
		opts = append(opts, gs.IndexWithEmbedCache(o.embCache))
	}

	t0 := time.Now()
	var stats gs.IndexBuildStats
	if o.dryRun {
		stats, err = cfg.Scan(opts...)
	} else {
		stats, err = cfg.Build(o.output, opts...)
	}
	if err != nil {
		return err
	}

	fmt.Fprintf(stderr, "[gs] scanned %d items, skipped %d in %v\n",
		stats.Items, stats.Skipped, time.Since(t0).Round(time.Millisecond))
	if o.dryRun {
		fmt.Fprintf(stderr, "[gs] --dry-run set: index not built\n")
		return nil
	}
	if sz, err := sizeOfDir(o.output); err == nil {
		fmt.Fprintf(stderr, "[gs] index payload: %s (%d files)\n", humanBytes(sz), countFiles(o.output))
	}
	return nil
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
