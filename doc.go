// Package gs provides a pure-Go library for building and searching
// hybrid (BM25 + BGE semantic) indexes over a corpus of documents.
//
// Import path: github.com/ejfkdev/gs
//
// Quick start:
//
//	// 1. Define a schema
//	schema := &gs.Schema{
//	    Name: "mycorpus",
//	    Fields: []gs.Field{
//	        {Name: "title", Type: gs.FieldText,     Searchable: true, Embeddable: true, FieldWeight: 5.0, Display: true},
//	        {Name: "body",  Type: gs.FieldLongText, Searchable: true, Embeddable: true, FieldWeight: 1.0, Display: true, Snippet: true},
//	        {Name: "tags",  Type: gs.FieldTags,     Searchable: true, Embeddable: true, FieldWeight: 3.0, Display: true},
//	    },
//	}
//
//	// 2. Build an index
//	b, _ := gs.NewBuilder(schema, "./indexes/mycorpus",
//	    gs.WithBGEPaths("./model/model.safetensors", "./model/vocab.txt"))
//	defer b.Close()
//	b.Add(gs.Item{
//	    Path:   "docs/intro.md",
//	    Source: "myapp",
//	    Fields: map[string]string{"title": "Getting started", "body": "...", "tags": "intro demo"},
//	})
//	b.Build()
//
//	// 3. Search
//	eng, _ := gs.Load("./indexes/mycorpus")
//	defer eng.Close()
//	hits, _ := eng.Search(ctx, gs.SearchOptions{Query: "getting started", TopK: 10})
//	for _, h := range hits {
//	    fmt.Printf("%s (%.3f) %s\n  %s\n", h.ID, h.Score, h.Path, h.Snippet)
//	}
//
// Load 只在这一次解析磁盘上的索引 (schema + items + embeddings + 重建
// inverted index), Search 全程走内存, 不重新读文件。所以应 Load 一次、
// 复用同一个 *Engine 反复 Search。进程内需要按目录缓存、或同时持有多个
// 不同索引实例时, 用 [Registry]:
//
//	reg := gs.NewRegistry()
//	wiki, _ := reg.Open("./indexes/wiki")    // 首次解析, 返回 *Engine
//	wiki2, _ := reg.Open("./indexes/wiki")   // 复用同一个实例 (refs+1)
//	skills, _ := reg.Open("./indexes/skills") // 另一个独立实例
//	reg.Release(wiki); reg.Release(wiki2)     // 引用归零时自动关闭
//	reg.CloseAll()
//
// Index files are little-endian and portable across platforms; the
// tokenizer is pure Go (no cgo), so cross-compilation works with
// CGO_ENABLED=0.
package gs
