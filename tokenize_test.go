package gs

import (
	"reflect"
	"testing"
)

func TestTokenize(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"Nginx配置", []string{"nginx", "配置"}},
		{"PRODUCT-2024-001", []string{"product", "2024", "001"}},
		{"hello world", []string{"hello", "world"}},
		{"Nginx 配置管理", []string{"nginx", "配置", "置管", "管理"}},
		{"", nil},
		{"!!!", nil},
		{"测试abc测试", []string{"测试", "abc", "测试"}},
		{"单个字运行时: 洞 和 配置", []string{"单个", "个字", "字运", "运行", "行时", "洞", "和", "配置"}},
	}
	for _, c := range cases {
		got := tokenize(c.in)
		if !reflect.DeepEqual(got, c.want) {
			t.Errorf("tokenize(%q) = %v, want %v", c.in, got, c.want)
		}
	}
}

func TestTokenizeDedup(t *testing.T) {
	got := tokenizeDedup("配置配置a A")
	want := []string{"配置", "置配", "a"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("tokenizeDedup = %v, want %v", got, want)
	}
}
