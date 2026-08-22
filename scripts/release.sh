#!/usr/bin/env sh
# 发布 gs 新版本: 一条命令创建并推送「库 tag + CLI module tag」。
#
#   用法: ./scripts/release.sh v0.3.3
#
# - vX.Y.Z        : 库版本, 触发 .github/workflows/release.yml 发二进制
# - cmd/gs/vX.Y.Z : CLI module 版本, 供 `go install .../cmd/gs@latest` 解析
#
# 发版前请先把 cmd/gs/internal/cli/cli.go 的 Version 常量 bump 到同一版本。
set -euo pipefail

VERSION="${1:?用法: ./scripts/release.sh vX.Y.Z}"

case "$VERSION" in
  v*[0-9]) ;;
  *)
    echo "版本号须以 'v' 开头, 例如 v0.3.3" >&2
    exit 1
    ;;
esac

git tag -a "$VERSION" -m "gs $VERSION"
git tag -a "cmd/gs/$VERSION" -m "gs CLI $VERSION"

git push origin "$VERSION" "cmd/gs/$VERSION"
echo "已推送 $VERSION 与 cmd/gs/$VERSION"