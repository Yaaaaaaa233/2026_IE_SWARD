#!/usr/bin/env bash
# 协调者定期监督一键报告（ADR-0003，取代逐 PR 评审）。
# 用法：bash tools/supervise.sh [天数，默认 2]
# 在最新 main 检出状态下运行；只读不改。
set -u
cd "$(git rev-parse --show-toplevel)" || exit 1
DAYS="${1:-2}"

echo "===== 1. 当前检出 ====="
git fetch origin --quiet
echo "本地 HEAD: $(git rev-parse --short HEAD)  $(git branch --show-current)"
echo "origin/main: $(git rev-parse --short origin/main)"
if ! git merge-base --is-ancestor HEAD origin/main; then
  echo "!! 本地 HEAD 不在 origin/main 历史内（有分叉或未推送提交），请先核对"
fi

echo
echo "===== 2. 近 ${DAYS} 天 origin/main 提交 ====="
git log --since="${DAYS} days ago" --pretty='format:%h %ad %an %s' --date=format:'%m-%d %H:%M' origin/main
echo

echo
echo "===== 3. 治理检查（HEAD） ====="
if python3 tools/check_governance.py --scope head; then
  echo "[监督] 治理检查 PASS"
else
  echo "[监督] 治理检查 FAIL —— 按 roadmap R0 待办与 DATA_POLICY 定位责任人"
fi

echo
echo "===== 4. 单元测试 ====="
python3 -m unittest discover -s tests 2>&1 | tail -2

echo
echo "===== 5. 任务索引核对（tasks/*.json 是否都在 README 登记） ====="
MISSING=0
for f in docs/tasks/*.json; do
  id=$(basename "$f" .json)
  grep -q "$id" docs/tasks/README.md || { echo "未登记索引: $id"; MISSING=1; }
done
[ "$MISSING" = 0 ] && echo "全部已登记"

echo
echo "===== 6. 最近 worklog（新会话是否留了交接） ====="
ls -t docs/worklog/*.md 2>/dev/null | head -5

echo
echo "监督结论建议：检查项 3/4/5 全绿且近期提交均有对应任务与 worklog，则本轮通过；"
echo "任何 FAIL/未登记项，向对应责任人回溯并记录。"
