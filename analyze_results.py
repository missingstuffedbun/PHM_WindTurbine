"""实验结果分析：读取已完成实验的输出目录，生成 HTML 结果报表。

用法：
    python analyze_results.py --results_dir results/20260825104437_full_experiment
    python analyze_results.py --results_dir results/exp1 results/exp2 --tol 2

报表内容：
1. 各场景、各模型的指标总览；
2. 物理约束（PINN）相对同 backbone baseline 的增益；
3. 物理约束增益的边界分析：增益随监测稀疏度的变化、增益转正的临界稀疏度；
4. 负迁移 / 等效样本清单与归因。
"""

import argparse
import html
import os
import statistics

from utils.analysis import DEFAULT_TOL, build_report

COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]

CSS = """
:root{--bg:#f5f7fa;--card:#ffffff;--line:#e2e8f0;--text:#1f2937;--muted:#64748b;
--gain:#047857;--gain-bg:#d1fae5;--neg:#b91c1c;--neg-bg:#fee2e2;--neu:#475569;--neu-bg:#e5e7eb;}
*{box-sizing:border-box}
body{margin:0;padding:32px 24px 64px;background:var(--bg);color:var(--text);
font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",sans-serif;font-size:14px;line-height:1.65}
.wrap{max-width:1180px;margin:0 auto}
h1{font-size:26px;margin:0 0 6px}
h2{font-size:19px;margin:0 0 14px;padding-left:10px;border-left:4px solid #2563eb}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px 22px;margin-bottom:20px}
.sub{color:var(--muted);font-size:13px;margin-bottom:18px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:18px}
.card{flex:1 1 150px;background:#f8fafc;border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.card .k{color:var(--muted);font-size:12px}
.card .v{font-size:19px;font-weight:600;margin-top:2px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 4px}
th,td{border:1px solid var(--line);padding:7px 9px;text-align:right}
th{background:#f1f5f9;font-weight:600;text-align:center}
td:first-child,th:first-child{text-align:left}
tbody tr:nth-child(even){background:#fafbfc}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600}
.tag.gain{color:var(--gain);background:var(--gain-bg)}
.tag.neg{color:var(--neg);background:var(--neg-bg)}
.tag.neu{color:var(--neu);background:var(--neu-bg)}
ul.concl{margin:0;padding-left:20px}
ul.concl li{margin-bottom:6px}
.reasons{margin:6px 0 0 18px;padding:0;color:#374151}
.reasons li{margin-bottom:4px}
svg{max-width:100%;height:auto;display:block;margin:6px auto 0}
a{color:#2563eb;text-decoration:none}
a:hover{text-decoration:underline}
.note{color:var(--muted);font-size:12.5px;margin-top:8px}
"""


def esc(text):
    return html.escape(str(text))


def fmt(value, nd=4):
    if value is None or value != value:
        return "-"
    return f"{value:.{nd}f}"


def pct(value, nd=1):
    if value is None or value != value:
        return "-"
    return f"{value:+.{nd}f}%"


def verdict_of(gain, tol):
    if gain != gain:
        return "neu", "-"
    if gain > tol:
        return "gain", "增益"
    if gain < -tol:
        return "neg", "负迁移"
    return "neu", "等效"


def tag(gain, tol):
    cls, text = verdict_of(gain, tol)
    return f'<span class="tag {cls}">{esc(text)}</span>'


def line_chart_svg(series, tol, width=780, height=360):
    """稀疏度–增益折线图（含 ±tol 死区带与零线）。"""
    all_x = [x for pts in series.values() for x, _ in pts]
    all_y = [y for pts in series.values() for _, y in pts] + [0.0, tol, -tol]
    if not all_x:
        return ""

    x_min, x_max = min(all_x), max(all_x)
    if x_max - x_min < 1e-9:
        x_min, x_max = x_min - 0.1, x_max + 0.1
    y_min, y_max = min(all_y), max(all_y)
    pad = max(4.0, (y_max - y_min) * 0.15)
    y_min, y_max = y_min - pad, y_max + pad

    pad_l, pad_r, pad_t, pad_b = 62, 108, 18, 48
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def sx(x):
        return pad_l + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y):
        return pad_t + (1 - (y - y_min) / (y_max - y_min)) * plot_h

    parts = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
             f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>']

    # ±tol 死区带
    parts.append(
        f'<rect x="{pad_l}" y="{sy(tol):.1f}" width="{plot_w}" '
        f'height="{abs(sy(-tol) - sy(tol)):.1f}" fill="#94a3b8" opacity="0.14"/>'
    )

    # 网格与 y 轴刻度
    for i in range(6):
        val = y_min + (y_max - y_min) * i / 5
        y = sy(val)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
                     f'stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" font-size="11" fill="#64748b" '
                     f'text-anchor="end">{val:.1f}</text>')

    # 零线
    parts.append(f'<line x1="{pad_l}" y1="{sy(0):.1f}" x2="{pad_l + plot_w}" y2="{sy(0):.1f}" '
                 f'stroke="#334155" stroke-width="1.4" stroke-dasharray="6 4"/>')

    # x 轴刻度
    for x in sorted(set(all_x)):
        parts.append(f'<text x="{sx(x):.1f}" y="{pad_t + plot_h + 20}" font-size="11" '
                     f'fill="#64748b" text-anchor="middle">{x:g}</text>')
        parts.append(f'<line x1="{sx(x):.1f}" y1="{pad_t + plot_h}" x2="{sx(x):.1f}" '
                     f'y2="{pad_t + plot_h + 5}" stroke="#94a3b8"/>')

    # 坐标轴
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#94a3b8"/>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" '
                 f'y2="{pad_t + plot_h}" stroke="#94a3b8"/>')
    parts.append(f'<text x="{pad_l + plot_w / 2:.0f}" y="{height - 8}" font-size="12" '
                 f'fill="#334155" text-anchor="middle">observable_ratio（1.0 = 完整监测，越小越稀疏）</text>')
    parts.append(f'<text x="14" y="{pad_t + plot_h / 2:.0f}" font-size="12" fill="#334155" '
                 f'transform="rotate(-90 14 {pad_t + plot_h / 2:.0f})" text-anchor="middle">'
                 f'PINN 相对 baseline 的 RMSE 改善 (%)</text>')

    # 序列
    for idx, (name, pts) in enumerate(series.items()):
        color = COLORS[idx % len(COLORS)]
        label = "全局平均" if name == "__global__" else name
        pts = sorted(pts, key=lambda p: p[0])
        width_px = 2.6 if name == "__global__" else 1.6
        dash = "" if name == "__global__" else ' stroke-dasharray="5 3"'
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" '
                     f'stroke-width="{width_px}"{dash}/>')
        for x, y in pts:
            parts.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.4" fill="{color}"/>')
        lx, ly = sx(pts[-1][0]) + 8, sy(pts[-1][1]) + 4
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="11.5" fill="{color}">{esc(label)}</text>')

    parts.append("</svg>")
    return "".join(parts)


def section(title, body, subtitle=""):
    sub = f'<div class="sub">{subtitle}</div>' if subtitle else ""
    return f"<section><h2>{esc(title)}</h2>{sub}{body}</section>"


def render_html(report, out_path):
    tol = report["tol"]
    runs = report["runs"]
    gain_rows = report["gain_rows"]
    base_dir = os.path.dirname(os.path.abspath(out_path))

    def rel(path):
        try:
            return os.path.relpath(path, base_dir).replace("\\", "/")
        except ValueError:
            return path

    gains = [r["gain_rmse_pct"] for r in gain_rows]
    n_pos = sum(1 for g in gains if g > tol)
    n_neg = sum(1 for g in gains if g < -tol)

    # 头部
    cards = [
        ("结果根目录", f"{len(report['roots'])} 个"),
        ("Run 数", len(runs)),
        ("场景数", len(report["scenarios"])),
        ("Backbone 数", len({r["backbone"] for r in runs})),
        ("对比组数", len(gain_rows)),
        ("平均增益", pct(statistics.fmean(gains)) if gains else "-"),
    ]
    cards_html = "".join(
        f'<div class="card"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div></div>'
        for k, v in cards
    )
    roots_html = "<br>".join(esc(r) for r in report["roots"])

    head = section(
        "实验总览",
        f'<div class="cards">{cards_html}</div>'
        f'<div class="note">生成时间：{esc(report["generated_at"])}　|　判定死区：±{tol:g}%'
        f'　|　正增益 {n_pos} 组 / 负迁移 {n_neg} 组<br>{roots_html}</div>',
    )

    # 结论
    concl = "".join(f"<li>{esc(c)}</li>" for c in report["conclusions"])
    concl_sec = section("结论摘要", f'<ul class="concl">{concl}</ul>')

    # 配置
    if report["configs"]:
        rows = []
        for name, cfg in report["configs"].items():
            loss = cfg.get("loss") or {}
            tr = cfg.get("training") or {}
            rows.append([
                esc(name),
                esc(loss.get("physics_weight", "-")),
                esc(str(loss.get("adaptive_physics", "-"))),
                esc(loss.get("data_weight", "-")),
                esc(tr.get("epochs", "-")),
                esc(tr.get("batch_size", "-")),
                esc(tr.get("lr", "-")),
                esc(", ".join(cfg.get("backbones") or []) if isinstance(cfg.get("backbones"), list)
                    else cfg.get("backbones") or "-"),
            ])
        cfg_body = (
            "<table><thead><tr><th>实验目录</th><th>physics_weight</th><th>adaptive</th>"
            "<th>data_weight</th><th>epochs</th><th>batch_size</th><th>lr</th>"
            "<th>backbones</th></tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
            + "</tbody></table>"
        )
        cfg_sec = section("实验配置", cfg_body,
                          "物理约束权重与训练超参，用于解释增益差异的来源。")
    else:
        cfg_sec = ""

    # 场景与驱动可用性
    act_names = list((report["scenarios"][0]["activity"].keys() if report["scenarios"] else []))
    sc_rows = []
    for sc in report["scenarios"]:
        cells = []
        for name in act_names:
            a = sc["activity"].get(name, 1.0)
            cls = "gain" if a >= 0.9 else ("neu" if a >= 0.5 else "neg")
            cells.append(f'<span class="tag {cls}">{a:.0%}</span>')
        sc_rows.append([
            esc(sc["name"]), esc(sc["level"]), f"{sc['ratio']:g}",
            esc(", ".join(sc["blocked"]) or "无"), str(sc["n_runs"]), esc(sc["description"]),
        ] + cells)
    sc_head = "".join(f"<th>{esc(n)}</th>" for n in act_names)
    sc_sec = section(
        "场景与物理约束驱动可用性",
        "<table><thead><tr><th>场景</th><th>level</th><th>observable_ratio</th>"
        f"<th>blocked_signals</th><th>run 数</th><th>说明</th>{sc_head}</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in sc_rows)
        + "</tbody></table>"
        + '<div class="note">驱动可用性 = 该约束所用驱动通道在该场景下可用的比例：'
          'blocked 通道为 0；稀疏场景下的有效观测占比即 observable_ratio。'
          'level 决定缺失的组织方式：channel = 随机通道整条时间轴不可用；'
          'timestep = 随机时间点上所有候选通道同时缺失；segment = 连续时间段整段置空'
          '（连续缺失对时序相关性的破坏强于同比例的随机缺失）。</div>',
    )

    # 结果总览
    run_rows = []
    for r in sorted(runs, key=lambda x: (x["scenario"], x["backbone"], x["use_pinn"])):
        m = r["metrics"]
        run_rows.append([
            esc(r["scenario"]), esc(r["backbone"]),
            '<span class="tag gain">PINN</span>' if r["use_pinn"]
            else '<span class="tag neu">baseline</span>',
            fmt(m.get("overall_rmse")), fmt(m.get("overall_mae")), fmt(m.get("overall_r2")),
            fmt(m.get("TMBNS_rmse")), fmt(m.get("TMBEW_rmse")), fmt(m.get("TMBTOR_rmse")),
            f'<a href="{esc(rel(r["run_dir"]))}">{esc(r["run_name"])}</a>',
        ])
    runs_sec = section(
        "各场景 / 各模型结果总览",
        "<table><thead><tr><th>场景</th><th>backbone</th><th>类型</th><th>RMSE</th>"
        "<th>MAE</th><th>R²</th><th>TMBNS RMSE</th><th>TMBEW RMSE</th><th>TMBTOR RMSE</th>"
        "<th>run 目录</th></tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in run_rows)
        + "</tbody></table>"
        + '<div class="note">指标为标准化空间中的测试集结果；RMSE / MAE 越小越好，R² 越大越好。</div>',
    )

    # 增益总表
    if gain_rows:
        g_rows = []
        for r in sorted(gain_rows, key=lambda x: (x["ratio"], x["backbone"]), reverse=True):
            g_rows.append([
                esc(r["scenario"]), f"{r['ratio']:g}", esc(r["backbone"]),
                fmt(r["base_rmse"]), fmt(r["pinn_rmse"]),
                f"<b>{pct(r['gain_rmse_pct'])}</b>",
                fmt(r["base_r2"]), fmt(r["pinn_r2"]), pct(r["gain_r2"], 2),
                tag(r["gain_rmse_pct"], tol),
            ])
        gain_tbl = (
            "<table><thead><tr><th>场景</th><th>ratio</th><th>backbone</th>"
            "<th>baseline RMSE</th><th>PINN RMSE</th><th>RMSE 改善</th>"
            "<th>baseline R²</th><th>PINN R²</th><th>ΔR²</th><th>判定</th></tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in g_rows)
            + "</tbody></table>"
        )

        sc_tbl = (
            "<table><thead><tr><th>场景</th><th>ratio</th><th>平均增益</th><th>最小</th>"
            "<th>最大</th><th>正增益/总数</th><th>最优 backbone</th><th>最差 backbone</th>"
            "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{c}</td>" for c in [
                    esc(d["scenario"]), f"{d['ratio']:g}",
                    f"<b>{pct(d['mean_gain'])}</b>", pct(d["min_gain"]), pct(d["max_gain"]),
                    f"{d['n_pos']}/{d['n']}", esc(d["best"]), esc(d["worst"]),
                ]) + "</tr>" for d in report["by_scenario"])
            + "</tbody></table>"
        )

        bb_tbl = (
            "<table><thead><tr><th>backbone</th><th>平均增益</th><th>最小</th><th>最大</th>"
            "<th>正增益/总数</th><th>最优场景</th><th>最差场景</th></tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{c}</td>" for c in [
                    esc(d["backbone"]), f"<b>{pct(d['mean_gain'])}</b>",
                    pct(d["min_gain"]), pct(d["max_gain"]),
                    f"{d['n_pos']}/{d['n']}", esc(d["best"]), esc(d["worst"]),
                ]) + "</tr>" for d in report["by_backbone"])
            + "</tbody></table>"
        )
        gain_sec = section(
            "物理约束增益",
            gain_tbl
            + '<div class="sub" style="margin-top:18px">按场景汇总</div>' + sc_tbl
            + '<div class="sub" style="margin-top:18px">按 backbone 汇总</div>' + bb_tbl
            + '<div class="note">改善 = (baseline − PINN) / baseline × 100%，正值表示 PINN 更优；'
              f'|改善| ≤ {tol:g}% 视为等效。</div>',
            "同一 (场景, backbone) 下 PINN 与 baseline 的配对比较。",
        )

        # 分目标增益
        targets = sorted({t for r in gain_rows for t in r["targets"]})
        if targets:
            t_head = "".join(f"<th>{esc(t)} 改善</th>" for t in targets)
            t_rows = []
            for r in sorted(gain_rows, key=lambda x: (x["ratio"], x["backbone"]), reverse=True):
                cells = [esc(r["scenario"]), f"{r['ratio']:g}", esc(r["backbone"])]
                for t in targets:
                    g = r["targets"].get(t, {}).get("gain_pct", float("nan"))
                    cells.append(f"<b>{pct(g)}</b> {tag(g, tol)}")
                t_rows.append(cells)
            tgt_sec = section(
                "分目标增益",
                "<table><thead><tr><th>场景</th><th>ratio</th><th>backbone</th>"
                f"{t_head}</tr></thead><tbody>"
                + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in t_rows)
                + "</tbody></table>"
                + '<div class="note">物理约束对各目标的作用方向可能不一致：'
                  '弯矩类目标受惯性 / 推力约束影响，扭矩目标受 TMBTOR ∝ ω² 约束影响。</div>',
            )
        else:
            tgt_sec = ""
    else:
        gain_sec = section("物理约束增益", '<div class="note">未找到可配对的 baseline / PINN 实验。</div>')
        tgt_sec = ""

    # 边界分析
    boundary = report["boundary"]
    series = boundary.get("series") or {}
    chart = line_chart_svg(series, tol) if series else ""
    b_parts = [chart] if chart else []
    if boundary.get("note"):
        b_parts.append(f'<div class="note">{esc(boundary["note"])}</div>')

    cross_rows = []
    g = boundary.get("global") or {}
    for c in g.get("crossings", []):
        cross_rows.append([
            "全局平均", f"{c['ratio_from']:g} → {c['ratio_to']:g}",
            f"{c['gain_from']:+.1f}% → {c['gain_to']:+.1f}%",
            f"<b>{c['ratio_star']:.2f}</b>",
            "ratio &lt; r* 时 PINN 占优" if c["direction"] == "sparse_gain" else "ratio &gt; r* 时 PINN 占优",
        ])
    for bb in boundary.get("per_backbone", []):
        for c in bb["crossings"]:
            cross_rows.append([
                esc(bb["backbone"]), f"{c['ratio_from']:g} → {c['ratio_to']:g}",
                f"{c['gain_from']:+.1f}% → {c['gain_to']:+.1f}%",
                f"<b>{c['ratio_star']:.2f}</b>",
                "ratio &lt; r* 时 PINN 占优" if c["direction"] == "sparse_gain" else "ratio &gt; r* 时 PINN 占优",
            ])
    for lv in boundary.get("per_level", []):
        for c in lv["crossings"]:
            cross_rows.append([
                f"level = {esc(lv['level'])}", f"{c['ratio_from']:g} → {c['ratio_to']:g}",
                f"{c['gain_from']:+.1f}% → {c['gain_to']:+.1f}%",
                f"<b>{c['ratio_star']:.2f}</b>",
                "ratio &lt; r* 时 PINN 占优" if c["direction"] == "sparse_gain" else "ratio &gt; r* 时 PINN 占优",
            ])
    if cross_rows:
        b_parts.append(
            "<table><thead><tr><th>序列</th><th>区间 (observable_ratio)</th><th>增益变化</th>"
            "<th>临界稀疏度 r*</th><th>含义</th></tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in cross_rows)
            + "</tbody></table>"
        )
        b_parts.append(
            '<div class="note">r* 由相邻两个稀疏度采样点线性插值得到，'
            '其不确定度约为采样间隔的一半；采样点越密，边界越可靠。</div>'
        )
    elif not boundary.get("note"):
        b_parts.append(
            '<div class="note">在已测稀疏度区间内未出现增益符号翻转，'
            '因此没有可插值的临界点（详见结论摘要）。</div>'
        )

    ratios = sorted({r for pts in series.values() for r, _ in pts}, reverse=True)
    pts_rows = []
    for name, pts in series.items():
        label = "全局平均" if name == "__global__" else name
        mapping = {r: g for r, g in pts}
        pts_rows.append([esc(label)] + [pct(mapping.get(r, float("nan"))) for r in ratios])
    if pts_rows:
        r_head = "".join(f"<th>ratio = {r:g}</th>" for r in ratios)
        b_parts.append(
            "<table><thead><tr><th>序列</th>" + r_head + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in pts_rows)
            + "</tbody></table>"
        )

    boundary_sec = section(
        "物理约束增益的边界分析",
        "".join(b_parts),
        "增益随监测稀疏度（observable_ratio）的变化：r* 为增益由负转正的临界稀疏度，"
        "即只有稀疏到该程度以下，物理约束才开始提供净收益。",
    )

    # 负迁移归因
    if report["negative"]:
        neg_body = []
        for c in report["negative"]:
            reasons = "".join(f"<li>{esc(x)}</li>" for x in c["reasons"])
            neg_body.append(
                f'<div style="border:1px solid var(--line);border-radius:8px;padding:12px 14px;'
                f'margin-bottom:12px">'
                f'<div><b>{esc(c["scenario"])} / {esc(c["backbone"])}</b>　'
                f'{tag(c["gain_rmse_pct"], tol)}　'
                f'<span class="sub">增益 {pct(c["gain_rmse_pct"])}，'
                f'RMSE {fmt(c["base_rmse"])} → {fmt(c["pinn_rmse"])}，'
                f'baseline R² {fmt(c["base_r2"])}，驱动可用性均值 {c["avg_activity"]:.0%}</span></div>'
                f'<ul class="reasons">{reasons}</ul></div>'
            )
        neg_sec = section("负迁移归因", "".join(neg_body),
                          f"增益低于 −{tol:g}% 或与 baseline 等效的样本，及其可能的成因。")
    else:
        neg_sec = section("负迁移归因",
                          '<div class="note">未发现负迁移样本：所有配对实验中 PINN 均不劣于 baseline。</div>')

    # run 索引
    idx_rows = []
    for r in sorted(runs, key=lambda x: x["run_name"]):
        imgs = " ".join(
            f'<a href="{esc(rel(os.path.join(r["run_dir"], n)))}">{esc(n.replace(".png", ""))}</a>'
            for n in r["images"]
        ) or "-"
        idx_rows.append([
            f'<a href="{esc(rel(r["run_dir"]))}">{esc(r["run_name"])}</a>',
            esc(r["scenario"]), esc(r["backbone"]),
            "PINN" if r["use_pinn"] else "baseline",
            fmt(r["metrics"].get("overall_rmse")), imgs,
        ])
    idx_sec = section(
        "Run 索引",
        "<table><thead><tr><th>run</th><th>场景</th><th>backbone</th><th>类型</th>"
        "<th>RMSE</th><th>可视化</th></tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in idx_rows)
        + "</tbody></table>",
    )

    return (
        "<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>风机稀疏监测实验结果分析报表</title>"
        f"<style>{CSS}</style></head><body><div class=\"wrap\">"
        "<h1>风机稀疏监测实验结果分析报表</h1>"
        '<div class="sub" style="margin-bottom:22px">物理约束（PINN）在不同稀疏监测场景下的'
        '增益与边界分析</div>'
        + head + concl_sec + cfg_sec + sc_sec + runs_sec + gain_sec + tgt_sec
        + boundary_sec + neg_sec + idx_sec
        + "</div></body></html>"
    )


def main():
    parser = argparse.ArgumentParser(description="分析已完成的实验结果并生成 HTML 报表")
    parser.add_argument(
        "--results_dir", nargs="+", required=True,
        help="实验输出根目录（可多个），例如 results/20260825104437_full_experiment",
    )
    parser.add_argument(
        "--out", default=None,
        help="报表输出路径，默认写到第一个结果根目录下的 analysis_report.html",
    )
    parser.add_argument(
        "--tol", type=float, default=DEFAULT_TOL,
        help=f"增益判定死区（百分比），默认 {DEFAULT_TOL}",
    )
    args = parser.parse_args()

    roots = []
    for r in args.results_dir:
        path = os.path.abspath(r)
        if not os.path.isdir(path):
            raise SystemExit(f"结果目录不存在: {r}")
        roots.append(path)

    report = build_report(roots, tol=args.tol)
    if not report["runs"]:
        raise SystemExit("未在给定目录中找到任何 run 结果（metrics.yaml / results.npz）。")

    out_path = args.out or os.path.join(roots[0], "analysis_report.html")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(render_html(report, out_path))

    gains = [r["gain_rmse_pct"] for r in report["gain_rows"]]
    print(f"Runs: {len(report['runs'])} | 对比组: {len(report['gain_rows'])} | "
          f"负迁移: {len(report['negative'])}")
    if gains:
        print(f"平均增益: {statistics.fmean(gains):+.2f}% "
              f"(min {min(gains):+.2f}%, max {max(gains):+.2f}%)")
    print(f"报表已生成: {os.path.abspath(out_path)}")
    for line in report["conclusions"]:
        print(f"  - {line}")


if __name__ == "__main__":
    main()
