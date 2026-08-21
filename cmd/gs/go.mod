module github.com/ejfkdev/gs/cmd/gs

go 1.26

require (
	github.com/ejfkdev/gs v0.3.0
	github.com/ejfkdev/xyz-go v0.2.2
	github.com/fsnotify/fsnotify v1.10.1
	gopkg.in/yaml.v3 v3.0.1
)

require (
	github.com/google/jsonschema-go v0.4.3 // indirect
	github.com/modelcontextprotocol/go-sdk v1.7.0 // indirect
	github.com/segmentio/asm v1.2.1 // indirect
	github.com/segmentio/encoding v0.5.4 // indirect
	github.com/yosida95/uritemplate/v3 v3.0.2 // indirect
	golang.org/x/oauth2 v0.36.0 // indirect
	golang.org/x/sync v0.22.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
	golang.org/x/time v0.15.0 // indirect
	gonum.org/v1/gonum v0.17.0 // indirect
)

// 嵌套 module: CLI 与库同仓库。本地/CI 构建直接指向父目录的库源码，
// 无需先给库打 tag；发布二进制时再由 release workflow 打包 cmd/gs。
replace github.com/ejfkdev/gs => ../..
