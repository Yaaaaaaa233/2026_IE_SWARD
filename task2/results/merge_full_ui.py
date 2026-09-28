"""Merge the real three-cutoff scores into the existing full Task-2 UI.

This leaves the original navigation, radar demonstration, ranking filters and
operations cards in place. Raw gpsno values are not embedded in the output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import secrets
from pathlib import Path


def read_csv(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 500 or len({row[key] for row in rows}) != 500:
        raise ValueError(f"{path}: expected 500 unique {key} rows")
    return {row[key]: row for row in rows}


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"UI structure changed: expected one occurrence of {old[:60]!r}")
    return source.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-ui", type=Path, required=True)
    parser.add_argument("--static-scores", type=Path, required=True)
    parser.add_argument("--three-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    html = args.original_ui.read_text(encoding="utf-8-sig")
    match = re.search(r"(?m)^window\.TASK2_DEMO_DATA = (\{.*\});$", html)
    if not match:
        raise ValueError("full UI has no embedded TASK2_DEMO_DATA")
    data = json.loads(match.group(1))
    static = read_csv(args.static_scores, "gpsno")
    dynamic = read_csv(args.three_scores, "gpsno")
    if set(static) != set(dynamic):
        raise ValueError("static and dynamic vehicle sets differ")
    vehicles = data["vehicles"]
    if len(vehicles) != 500:
        raise ValueError("full UI must contain 500 vehicles")

    ordered_ids = sorted(static, key=lambda g: (-float(static[g]["S_final"]), g))
    salt = secrets.token_hex(16)
    mapping = []
    for index, (vehicle, gpsno) in enumerate(zip(vehicles, ordered_ids), start=1):
        row = static[gpsno]
        if vehicle["safe_rank"] != index or vehicle["score"] != float(row["S_final"]) or vehicle["risk_pct"] != float(row["risk_pct"]):
            raise ValueError(f"original UI does not match frozen static rank {index}")
        dyn = dynamic[gpsno]
        scores = [float(dyn[f"score_{date}"]) for date in ("20260621", "20260711", "20260731")]
        if abs(scores[0] - vehicle["score"]) > 0.0005:
            raise ValueError(f"{gpsno}: anchor differs from original UI")
        display_id = "司机-" + hashlib.sha256(f"{salt}|{gpsno}".encode()).hexdigest()[:8].upper()
        vehicle["id"] = display_id
        vehicle["trend"] = [
            {"period": period, "score": value}
            for period, value in zip(("6/21", "7/11", "7/31"), scores)
        ]
        vehicle["score_latest"] = scores[2]
        vehicle["score_change"] = round(scores[2] - scores[0], 3)
        vehicle["g8_prob_20260801"] = float(dyn["g8_risk_prob_asof_20260801"])
        vehicle.pop("demo_change", None)
        vehicle.pop("illustrative_trend", None)
        mapping.append((gpsno, display_id))

    if len({name for _, name in mapping}) != 500:
        raise ValueError("display ID collision")
    for cutoff, field in enumerate(("rank_20260621", "rank_20260711", "rank_20260731")):
        ranked = sorted(vehicles, key=lambda v: (-v["trend"][cutoff]["score"], v["id"]))
        for position, vehicle in enumerate(ranked, start=1):
            vehicle[field] = position
    if any(v["rank_20260621"] != v["safe_rank"] for v in vehicles):
        # Equal-score cars can swap under a new display ID. Keep the frozen rank.
        for vehicle in vehicles:
            vehicle["rank_20260621"] = vehicle["safe_rank"]

    data["meta"]["dynamic_source"] = "final-v1-dyn-policy-v1; 20-day windows ending 2026-06-21/07-11/07-31"
    data["meta"]["g8_reference"] = "g8v6-corrected; risk probability as of 2026-08-01, not used for earlier scores"
    data["meta"]["illustrative_fields"] = ["dimensions", "demo_reasons"]
    data["meta"]["privacy"] = "gpsno omitted; salted short display IDs; source file stays controlled"
    means = [sum(v["trend"][i]["score"] for v in vehicles) / 500 for i in range(3)]
    improved = sum(v["score_change"] > 0 for v in vehicles)
    worsened = sum(v["score_change"] < 0 for v in vehicles)
    data["dynamic_summary"] = {
        "means": [round(mean, 3) for mean in means],
        "improved": improved,
        "worsened": worsened,
        "unchanged": 500 - improved - worsened,
    }

    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = html[:match.start(1)] + serialized + html[match.end(1):]
    html = replace_once(html, "<h3>近12周分数变化</h3><p>用于验证趋势展示是否容易理解。</p>",
                        "<h3>三个20天截点评分</h3><p>真实滚动回放：分别观察前20天的驾驶证据。</p>")
    html = replace_once(html, '<span class="source-tag amber">演示轨迹</span>',
                        '<span class="source-tag">真实动态评分</span>')
    html = replace_once(html, '<p class="chart-note">曲线终点使用冻结总分，其余周次为结构演示。正式版替换为60日滚动重放结果。</p>',
                        '<p id="trendSummary" class="chart-note">正在加载三截点结果。</p>')
    html = replace_once(html, '<span>总分与排名来自冻结结果</span>',
                        '<span>6/21总分与排名来自冻结结果；趋势为真实三截点回放</span>')
    html = replace_once(html, '<span class="badge">可分享演示</span>',
                        '<span class="badge">真实三截点 · 受控分享</span>')
    html = replace_once(html, '<p>先看结论，再定位需要解释和行动的车辆。</p>',
                        '<p>先看冻结主截点，再比较三次真实20天评分。</p>')
    html = replace_once(html, '<span class="muted">安全分</span><strong id="driverScore">—</strong>',
                        '<span class="muted">6/21安全分</span><strong id="driverScore">—</strong>')
    html = replace_once(html, '<span class="muted">安全排名</span><strong id="driverRank">—</strong>',
                        '<span class="muted">6/21安全排名</span><strong id="driverRank">—</strong>')
    html = replace_once(html, '<p>总分越高越安全；风险排名越靠前越需要关注。</p>',
                        '<p>按6/21冻结分排序，同时列出7/31安全分与名次；总分越高越安全。</p>')
    html = replace_once(html, '<th>安全分</th><th>风险分位</th>',
                        '<th>6/21安全分</th><th>7/31安全分</th><th>7/31排名</th><th>风险分位</th>')
    html = replace_once(html, '.table-panel { padding: 0; overflow: hidden; }',
                        '.table-panel { padding: 0; overflow-x: auto; }')
    html = replace_once(html, '["低暴露车辆", `${summary.evidence_counts.low_exposure_shrunk || 0} 辆`, "已收缩并单独标记"],',
                        '["低暴露车辆", `${summary.evidence_counts.low_exposure_shrunk || 0} 辆`, "已收缩并单独标记"],\n      ["7/31车队均分", score(data.dynamic_summary.means[2]), "第三个真实20天窗口"],\n      ["首末改善 / 恶化", `${data.dynamic_summary.improved} / ${data.dynamic_summary.worsened} 辆`, "按6/21与7/31分数比较"],')
    html = replace_once(html, '<div><dt>当前已证明</dt><dd>冻结规则覆盖 ${data.meta.record_count} 辆车，所有车辆均有总分与证据状态。</dd></div>',
                        '<div><dt>当前已核验</dt><dd>冻结规则和三截点回放均覆盖 ${data.meta.record_count} 辆车；原60日回放与交付台账逐车逐日完全一致。</dd></div>')
    html = replace_once(html, 'html += `<text class="trend-label" x="${width-right}" y="${top+2}" text-anchor="end">当前 ${score(vehicle.score)} 分</text>`;',
                        'html += `<text class="trend-label" x="${width-right}" y="${top+2}" text-anchor="end">7/31 ${score(vehicle.score_latest)} 分</text>`;')
    html = replace_once(html, 'if (i % 2 === 0 || i === series.length - 1) html += `<text class="trend-label" x="${x(i)}" y="${height-18}" text-anchor="middle">${i+1}</text>`;',
                        'html += `<text class="trend-label" x="${x(i)}" y="${height-18}" text-anchor="middle">${d.period}</text>`;\n      html += `<text class="trend-label" x="${x(i)}" y="${y(d.score)-12}" text-anchor="middle">${score(d.score)}</text>`;')
    html = replace_once(html, 'svg.innerHTML = html;\n  }\n\n  function renderDriver(vehicle) {',
                        'svg.innerHTML = html;\n    const sign = vehicle.score_change > 0 ? "+" : "";\n    $("#trendSummary").textContent = `首末变化 ${sign}${vehicle.score_change.toFixed(2)} 分；名次 ${vehicle.rank_20260621} → ${vehicle.rank_20260711} → ${vehicle.rank_20260731}。三个截点分别使用此前20天证据；8/1 g8风险概率 ${(vehicle.g8_prob_20260801*100).toFixed(2)}% 仅作参照。`;\n  }\n\n  function renderDriver(vehicle) {')
    html = replace_once(html, '<td class="score-cell">${score(vehicle.score)}</td><td>${pct(vehicle.risk_pct)}</td>',
                        '<td class="score-cell">${score(vehicle.score)}</td><td class="score-cell">${score(vehicle.score_latest)}</td><td>${vehicle.rank_20260731}</td><td>${pct(vehicle.risk_pct)}</td>')
    html = replace_once(html, '<p>展示分档用于UI阅读测试，不作为冻结处罚阈值。</p>',
                        '<p>6/21冻结分的展示分档；不作为正式处罚阈值。</p>')
    html = replace_once(html, '<p class="chart-note">证据可信度低表示数据不足，不代表驾驶行为更危险。正式维度分待冻结台账接口接入。</p>',
                        '<p class="chart-note">六维雷达仍是结构演示值，不参与三截点真实总分；证据可信度低表示依据不足，不代表驾驶行为更危险。</p>')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Merged {len(vehicles)} vehicles into {args.output}")
    print(f"Means: {data['dynamic_summary']['means']}; improved/worsened: {improved}/{worsened}")


if __name__ == "__main__":
    main()
