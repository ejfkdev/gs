// Command embedded_indexer 演示把 gs 嵌入一个长驻程序:
//   1) 声明式配置索引 (schema + 数据源 + 字段映射)
//   2) OpenLive 自动重载 (源目录变化后自动重新加载)
//   3) 持续搜索
//
// 运行: go run ./examples/embedded_indexer
// 纯 BM25 schema, 不依赖外部文件或 BGE 模型即可跑通。
//
// 配置也可以用 YAML 文件 + gs.LoadIndexConfig("index.yaml") 加载,
// 见 README「配置驱动索引」。这里用 Go 结构体直接构造, 演示程序化的方式。

package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	gs "github.com/ejfkdev/gs"
)

func main() {
	// 1. 准备一个临时源目录 + 几条示例 JSON 数据
	base, err := os.MkdirTemp("", "gs-embedded-*")
	check(err)
	defer os.RemoveAll(base)

	srcDir := filepath.Join(base, "data")
	check(os.MkdirAll(srcDir, 0o755))
	write(filepath.Join(srcDir, "nginx.json"), `{"title":"Nginx 反向代理","summary":"代理与静态缓存"}`)
	write(filepath.Join(srcDir, "redis.json"), `{"title":"Redis 缓存","summary":"内存数据库"}`)

	outDir := filepath.Join(base, "index")

	// 2. 声明式建索引 (schema + 数据源 + 每个字段怎么取值)
	cfg := &gs.IndexConfig{
		Schema: gs.SchemaConfig{
			Name:      "demo",
			NameField: "title",
			Fields: []gs.FieldConfig{
				{Name: "title", Type: "text", Searchable: true, Weight: 5.0, Display: true},
				{Name: "summary", Type: "longtext", Searchable: true, Weight: 1.0, Display: true, Snippet: true},
			},
		},
		Sources: []gs.SourceConfig{
			{
				Dir:     srcDir,
				Include: "*.json",
				Format:  "json",
				Mapping: map[string]string{"title": "title", "summary": "summary"},
			},
		},
	}
	stats, err := cfg.Build(outDir)
	check(err)
	fmt.Printf("built %d items\n\n", stats.Items)

	// 3. 长驻: OpenLive 每隔 interval 检测索引变化并自动重载
	live, err := gs.OpenLive(outDir, 500*time.Millisecond)
	check(err)
	defer live.Close()

	search(live, "nginx")

	// 4. 模拟新增一个文档: 新增源文件后重建索引 (生产里这一步由另一个进程的
	//    gs watch 自动完成; 这里手动 Build 只是为了示例自包含)。
	//    重建后 OpenLive 会检测到 items.bin 变化并自动重载。
	fmt.Println("--- 新增 golang.json 并重建之后 ---")
	write(filepath.Join(srcDir, "golang.json"), `{"title":"Go 并发","summary":"goroutine 与 channel"}`)
	_, err = cfg.Build(outDir)
	check(err)
	time.Sleep(2000 * time.Millisecond) // 等一次重载节拍

	search(live, "golang 并发")
}

func search(live *gs.LiveEngine, q string) {
	hits, err := live.Search(context.Background(), gs.SearchOptions{Query: q, TopK: 3})
	check(err)
	fmt.Printf("query=%q 命中 %d 条:\n", q, len(hits))
	for i, h := range hits {
		fmt.Printf("  %d. %s (%.3f)\n", i+1, h.ID, h.Score)
	}
	fmt.Println()
}

func write(path, content string) {
	check(os.WriteFile(path, []byte(content), 0o644))
}

func check(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "embedded_indexer: %v\n", err)
		os.Exit(1)
	}
}
