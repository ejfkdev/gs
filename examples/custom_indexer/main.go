// Command custom_indexer 演示如何把 gs 作为库使用:
// 定义自定义 schema → 构建索引 → 搜索。
//
// 运行: go run ./examples/custom_indexer
//
// 示例自带一个内存中的迷你语料, 不依赖外部文件即可跑通。
// 如果 ./bge/ 下有 BGE 模型 (model.safetensors + vocab.txt),
// 索引就是 hybrid (BM25 + BGE); 否则自动降级为纯 BM25, 依然
// 完整演示全部 API。

package main

import (
	"context"
	"fmt"
	"os"

	"github.com/ejfkdev/gs"
)

var sampleDocs = []doc{
	{path: "notes/golang.md", title: "Go 并发模型", tags: []string{"golang", "concurrency"}, body: "goroutine 与 channel 的协作方式, 以及 context 取消与超时控制。"},
	{path: "notes/nginx.md", title: "Nginx 配置", tags: []string{"web", "config"}, body: "nginx 反向代理与静态资源缓存的常用配置示例。"},
	{path: "notes/backup.md", title: "数据库备份", tags: []string{"database", "ops"}, body: "每日全量备份 + 增量 binlog 的恢复演练步骤。"},
}

type doc struct {
	path, title, body string
	tags              []string
}

func main() {
	// 1. 定义自定义 schema
	schema := &gs.Schema{
		Name: "demo",
		Fields: []gs.Field{
			{Name: "title", Type: gs.FieldText, Searchable: true, Embeddable: true, FieldWeight: 5.0, Display: true},
			{Name: "body", Type: gs.FieldLongText, Searchable: true, Embeddable: true, FieldWeight: 1.0, Display: true, Snippet: true},
			{Name: "tags", Type: gs.FieldTags, Searchable: true, Embeddable: true, FieldWeight: 3.0, Display: true},
		},
	}
	fmt.Println(schema.String())

	// 2. 构建索引 (没有 BGE 模型时降级为纯 BM25; 索引写到临时目录, 不污染仓库)
	dir, err := os.MkdirTemp("", "gs-demo-*")
	check(err)
	defer os.RemoveAll(dir)
	builderOpts := []gs.BuilderOption{
		gs.WithProgress(func(stage string, cur, total int) {
			if cur == total {
				fmt.Printf("  [%s] done (%d items)\n", stage, total)
			}
		}),
	}
	if haveModel() {
		builderOpts = append(builderOpts, gs.WithBGEPaths("bge/model.safetensors", "bge/vocab.txt"))
	} else {
		fmt.Println("(no ./bge model found — building a BM25-only index; set Embeddable=false for the same effect)")
		for i := range schema.Fields {
			schema.Fields[i].Embeddable = false
		}
	}

	b, err := gs.NewBuilder(schema, dir, builderOpts...)
	check(err)
	for _, d := range sampleDocs {
		check(b.Add(gs.Item{
			ID:     d.path,
			Path:   d.path,
			Source: "demo",
			Tags:   d.tags, // 会被自动合并进 "tags" (FieldTags) 字段
			Fields: map[string]string{
				"title": d.title,
				"body":  d.body,
			},
		}))
	}
	check(b.Build())
	check(b.Close())

	// 3. 搜索
	eng, err := gs.Load(dir)
	check(err)
	defer eng.Close()

	for _, q := range []string{"golang 并发", "nginx", "备份恢复"} {
		fmt.Printf("\nQuery: %q\n", q)
		hits, err := eng.Search(context.Background(), gs.SearchOptions{Query: q, TopK: 3})
		check(err)
		for i, h := range hits {
			fmt.Printf("  %d. %s (%.3f) %s\n", i+1, h.ID, h.Score, h.Path)
			fmt.Printf("     snippet: %s\n", h.Snippet)
		}
	}
}

func haveModel() bool {
	_, err1 := os.Stat("bge/model.safetensors")
	_, err2 := os.Stat("bge/vocab.txt")
	return err1 == nil && err2 == nil
}

func check(err error) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "demo: %v\n", err)
		os.Exit(1)
	}
}
