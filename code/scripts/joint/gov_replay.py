#!/usr/bin/env python3
"""gov_replay.py -- animate a governor run (thermal_ab.py ck_<ram|swap>_gov) against the baseline of the same scenario, for the video.

Run on a laptop (matplotlib + ffmpeg for MP4; without ffmpeg it writes a GIF). Inputs are the run folders thermal_ab.py writes:
  <run dir>/runs.jsonl                     t0, t1, first token, GPU job start/end, TTFT/TPOT, heat areas
  <run dir>/<config>_r<rep>/trace.csv      temperature of every zone, cpu6 and GPU frequency, every 2 s
  <run dir>/<gov config>_r<rep>/gov.csv    the governor's sensed state and applied setting, every 0.5 s (+ gov_events.jsonl)

  python3 scripts/joint/gov_replay.py --gov runs/thermal_X/ck_swap_gov_r1 --base runs/thermal_X/ck_ram_gdef_cdef_r1 --out gov.mp4
  python3 scripts/joint/gov_replay.py ... --still gov.png          # one picture of the finished run (report / poster)
  python3 scripts/joint/gov_replay.py ... --speed 20 --fps 20      # 20x real time
  python3 scripts/joint/gov_replay.py ... --ref "고정: KV swap만=runs/thermal_X/ck_swap_gdef_cdef_r1" --ref "고정: decode 1.5 GHz=runs/thermal_X/ck_swap_gdef_d1500_r1"
                                       # fixed configs of the same scenario as dashed lines and in the result card (governor vs static)
Panels: workload timeline (CPU job prefill / decode, GPU job), A720 frequency, GPU frequency, temperature of the hot zone (the zone with
the largest heat area in the baseline, as in thermal_ab.py) with the running heat area of both runs, and on the right the rule the governor
applies at that moment (state -> table cell -> setting -> evidence) with the list of switches so far.
"""
import argparse
import csv
import json
import math
import shutil
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib import animation                    # noqa: E402

import logging                                      # noqa: E402
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)    # the fallback font list is expected to miss some families
plt.rcParams.update({"font.family": ["Apple SD Gothic Neo", "Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"], "axes.unicode_minus": False,
                     "font.size": 13, "axes.edgecolor": "#bfbfbf", "axes.labelcolor": "#404040", "xtick.color": "#595959",
                     "ytick.color": "#595959", "savefig.facecolor": "white"})
BL, OR, GY, GN, YE, RD, LB = "#4472C4", "#ED7D31", "#A5A5A5", "#548235", "#FFC000", "#C00000", "#A5B8E0"
PHASE_KO = {"idle": "대기", "prefill": "CPU prefill", "decode": "CPU decode"}
# governor_lite.decide() rule names -> (Korean rule, Korean evidence) for the video
RULE_KO = {
    "prefill: keep max": ("prefill: 최대 주파수 유지", "prefill은 연산 병목. A720 1.5 GHz면 TTFT +50.6%"),
    "CPU decode: A720 1.5 GHz": ("CPU decode: A720 1.5 GHz", "decode만 1.5 GHz: 열 면적 0.87배, TTFT 1.00배"),
    "CPU decode, KV swap: default": ("CPU decode + KV swap: 기본 유지", "KV swap만으로 열 0.90배, 1.5 GHz를 더해도 0.90배"),
    "co-run decode: A720 1.5 GHz": ("동시 실행 decode: A720 1.5 GHz", "E7: 열 0.81배, CPU 작업 0.97배, GPU 작업 0.79배"),
    "co-run decode: A720 1.5 GHz (nearest cell)": ("동시 실행 decode: A720 1.5 GHz (가장 가까운 칸)", "요청 전체 1.5 GHz: 열 0.71배, CPU 작업 1.07배"),
    "GPU only: leave": ("GPU 단독: 그대로 둠", "GPU 주파수를 낮추면 열이 오히려 증가(1.09~1.44배)"),
    "idle": ("대기", ""),
}


def ko(rule):
    return RULE_KO.get(rule, (rule or "-", ""))


def wrap(t, n=30):
    return "\n  ".join(textwrap.wrap(t, n)) if t else "-"


def load_run(rdir):
    rdir = Path(rdir)
    name = rdir.name
    config, rep = name.rsplit("_r", 1)
    meta = None
    for line in open(rdir.parent / "runs.jsonl"):
        r = json.loads(line)
        if r["config"] == config and int(r["rep"]) == int(rep):
            meta = r
    if meta is None:
        raise SystemExit(f"{name}: no entry in {rdir.parent / 'runs.jsonl'}")
    rows = list(csv.DictReader(open(rdir / "trace.csv")))
    zones = [k for k in rows[0] if k.startswith("thermal_zone")]
    tr = {"t": [float(r["t_rel_s"]) for r in rows], "cpu6": [int(r["cpu6_khz"]) / 1000 for r in rows],
          "gpu": [int(r["gpu_hz"]) / 1e6 for r in rows], "T": {z: [float(r[z]) for r in rows] for z in zones}}
    gov, events = None, []
    if (rdir / "gov.csv").exists():
        gov = list(csv.DictReader(open(rdir / "gov.csv")))
        if (rdir / "gov_events.jsonl").exists():
            events = [json.loads(l) for l in open(rdir / "gov_events.jsonl")]
    t0 = meta["t0"]
    rel = lambda k: (meta[k] - t0) if meta.get(k) else None      # noqa: E731
    bursts = [(b[0] - t0, b[1] - t0) for b in meta.get("gpu_bursts") or []]
    if not bursts and meta.get("t_gpu_start"):
        bursts = [(rel("t_gpu_start"), rel("t_gpu_end") or meta["t1"] - t0)]
    return {"name": config, "meta": meta, "trace": tr, "zones": zones, "gov": gov, "events": events,
            "t_first": rel("t_first"), "t_end": meta["t1"] - t0, "bursts": bursts}


def baseline(tr, z):
    pre = [T for t, T in zip(tr["t"], tr["T"][z]) if t < 0]
    return sum(pre) / len(pre) if pre else tr["T"][z][0]


def running_area(tr, z, base):
    out, acc = [], 0.0
    for i, t in enumerate(tr["t"]):
        if i and t > 0:
            acc += 0.5 * ((tr["T"][z][i] - base) + (tr["T"][z][i - 1] - base)) * (t - max(0.0, tr["t"][i - 1]))
        out.append(acc)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gov", required=True, help="run folder of the governor config, e.g. runs/thermal_X/ck_swap_gov_r1")
    ap.add_argument("--base", required=True, help="run folder of the baseline of the same scenario, e.g. runs/thermal_X/ck_ram_gdef_cdef_r1")
    ap.add_argument("--out", default="gov_replay.mp4")
    ap.add_argument("--still", default=None, help="also write the last frame as a picture")
    ap.add_argument("--speed", type=float, default=15.0, help="seconds of the run per second of video")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--tail", type=float, default=90.0, help="seconds of cool-down shown after the governor run ends")
    ap.add_argument("--ref", action="append", default=[], help="'label=run folder' of a fixed config of the same scenario (repeatable)")
    ap.add_argument("--zone", default=None, help="thermal zone to plot (default: the one with the largest heat area in the baseline)")
    ap.add_argument("--title", default="지배 자원 기반 주파수 관리 (governor-lite): 실행 중 상태를 보고 사전 계산 테이블을 적용")
    a = ap.parse_args()

    G, B = load_run(a.gov), load_run(a.base)
    REFS = []
    for spec in a.ref:
        label, path = spec.split("=", 1)
        R = load_run(path); R["label"] = label
        REFS.append(R)
    if not G["gov"]:
        raise SystemExit(f"{a.gov}: no gov.csv (is it a ck_*_gov run?)")
    zone = a.zone or max(B["zones"], key=lambda z: running_area(B["trace"], z, baseline(B["trace"], z))[-1])
    for R in [G, B] + REFS:
        R["base"] = baseline(R["trace"], zone)
        R["area"] = running_area(R["trace"], zone, R["base"])
    t_max = max([G["t_end"], B["t_end"]] + [R["t_end"] for R in REFS]) + a.tail
    gv = G["gov"]
    gt = [float(r["t_rel"]) for r in gv]
    num = lambda v: float(v) if v not in ("", None) else None       # noqa: E731

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    gsp = fig.add_gridspec(4, 2, width_ratios=[2.6, 1.25], height_ratios=[0.8, 1, 1, 1.5], left=0.06, right=0.98, top=0.9, bottom=0.07,
                           hspace=0.35, wspace=0.08)
    axW, axC, axG, axT = (fig.add_subplot(gsp[i, 0]) for i in range(4))
    axP = fig.add_subplot(gsp[:, 1]); axP.axis("off")
    fig.suptitle(a.title, fontsize=20, color="#1F2A5A", fontweight="bold", x=0.06, ha="left")
    clock = fig.text(0.98, 0.945, "", ha="right", fontsize=18, color="#404040")
    for ax in (axW, axC, axG, axT):
        ax.set_xlim(-10, t_max)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6")
    axT.set_xlabel("실행 시작 후 시간 (s)")

    # workload lanes (static, revealed by a moving curtain)
    def lanes(R, y, label):
        tf = R["t_first"] or R["t_end"]
        axW.barh(y, tf, left=0, height=0.32, color=GY)
        axW.barh(y, R["t_end"] - tf, left=tf, height=0.32, color=BL)
        for gs_, ge_ in R["bursts"]:
            axW.barh(y - 0.36, ge_ - gs_, left=gs_, height=0.22, color=OR)
        axW.text(-12, y - 0.15, label, ha="right", va="center", fontsize=12)
    lanes(G, 1.0, "governor"); lanes(B, 0.0, "기준선")
    axW.set_ylim(-0.7, 1.4); axW.set_yticks([]); axW.set_title("작업: 회색 = CPU prefill, 파랑 = CPU decode, 주황 = GPU 작업", loc="left", fontsize=13)
    curtain = axW.axvspan(0, t_max, color="white", zorder=5)

    lc_b, = axC.plot([], [], color=GY, lw=2, label="기준선 (기본 거버너)")
    lc_g, = axC.plot([], [], color=BL, lw=2.5, label="governor")
    ls_c, = axC.step([], [], color=BL, lw=1.2, ls="--", where="post", label="governor 고정값 (없으면 기본 거버너)")
    axC.set_ylim(0, 2600); axC.set_ylabel("A720 MHz"); axC.legend(loc="lower left", ncol=3, frameon=False, fontsize=11)
    lg_b, = axG.plot([], [], color=GY, lw=2)
    lg_g, = axG.plot([], [], color=OR, lw=2.5)
    ls_g, = axG.step([], [], color=OR, lw=1.2, ls="--", where="post")
    axG.set_ylim(0, 1150); axG.set_ylabel("GPU MHz")
    lt_b, = axT.plot([], [], color=GY, lw=2, label="기준선")
    lt_g, = axT.plot([], [], color=RD, lw=2.5, label="governor")
    REF_COL = ["#7030A0", "#2E9AA8", "#8C8C8C"]
    ref_lines = []
    for k, R in enumerate(REFS):
        c = REF_COL[k % len(REF_COL)]
        l1, = axC.plot([], [], color=c, lw=1.4, ls="--", label=R["label"])
        l2, = axT.plot([], [], color=c, lw=1.4, ls="--", label=R["label"])
        ref_lines.append((R, l1, l2))
    if REFS:
        axC.legend(loc="lower left", ncol=3 + len(REFS), frameon=False, fontsize=10)
    allT = B["trace"]["T"][zone] + G["trace"]["T"][zone] + [x for R in REFS for x in R["trace"]["T"][zone]]
    # legend in one row above the curves (head-room added for it): stacked at the upper left it covered the prefill plateau
    span = max(allT) - min(allT)
    axT.set_ylim(min(allT) - 2, max(allT) + 0.22 * span + 4); axT.set_ylabel(f"온도 (°C), {zone}")
    axT.legend(loc="upper left", ncol=2 + len(REFS), frameon=False, fontsize=11)
    area_txt = None                                    # placed in the right-hand panel below (it would cover the curves)
    cursors = [ax.axvline(0, color="#7f7f7f", lw=1, ls=":") for ax in (axW, axC, axG, axT)]

    panel_hdr = axP.text(0.0, 0.98, "현재 적용 중인 규칙", fontsize=17, fontweight="bold", color="#1F2A5A", va="top")
    panel = axP.text(0.0, 0.93, "", fontsize=13.5, va="top", linespacing=1.5)
    log_hdr = axP.text(0.0, 0.6, "전환 기록", fontsize=15, fontweight="bold", color="#1F2A5A", va="top")
    logt = axP.text(0.0, 0.56, "", fontsize=11, va="top", linespacing=1.35)
    area_txt = axP.text(0.0, 0.335, "", fontsize=12, va="top", linespacing=1.4, color="#404040")
    summary = axP.text(0.0, 0.0, "", fontsize=11.5, va="bottom", linespacing=1.4, color="#404040",
                       bbox=dict(boxstyle="round,pad=0.5", fc="#F3F6FB", ec="#d9d9d9"))
    _ = (panel_hdr, log_hdr)

    def upto(t_list, *ys, T):
        k = 0
        while k < len(t_list) and t_list[k] <= T:
            k += 1
        return [t_list[:k]] + [y[:k] for y in ys]

    def fmt_set(v, unit="MHz"):
        return "기본" if v in ("", None) else f"{int(float(v))} {unit}".strip()

    def draw(T):
        for R, lc, lg, lt in ((B, lc_b, lg_b, lt_b), (G, lc_g, lg_g, lt_g)):
            t, c, g, th = upto(R["trace"]["t"], R["trace"]["cpu6"], R["trace"]["gpu"], R["trace"]["T"][zone], T=T)
            lc.set_data(t, c); lg.set_data(t, g); lt.set_data(t, th)
        for R, l1, l2 in ref_lines:
            t, c, th = upto(R["trace"]["t"], R["trace"]["cpu6"], R["trace"]["T"][zone], T=T)
            l1.set_data(t, c); l2.set_data(t, th)
        t, ca, ga = upto(gt, [num(r["a720_set_mhz"]) or math.nan for r in gv], [num(r["gpu_set_mhz"]) or math.nan for r in gv], T=T)
        ls_c.set_data(t, ca); ls_g.set_data(t, ga)
        x0 = max(T, 0.0)
        if hasattr(curtain, "set_width"):              # matplotlib >= 3.9: axvspan returns a Rectangle
            curtain.set_x(x0); curtain.set_width(max(0.0, t_max - x0))
        else:                                          # older: a Polygon
            curtain.set_xy([[x0, 0], [x0, 1], [t_max, 1], [t_max, 0], [x0, 0]])
        for cl in cursors:
            cl.set_xdata([T, T])
        ab = upto(B["trace"]["t"], B["area"], T=T)[1]
        ag = upto(G["trace"]["t"], G["area"], T=T)[1]
        ab, ag = (ab[-1] if ab else 0.0), (ag[-1] if ag else 0.0)
        area_txt.set_text(f"누적 열 면적 (°C·s):  기준선 {ab:,.0f},  governor {ag:,.0f}" + (f"  ({ag / ab:.2f}배)" if ab > 50 else ""))
        k = 0
        while k + 1 < len(gt) and gt[k + 1] <= T:
            k += 1
        r = gv[k] if gt and gt[0] <= T else None
        if r is None or T > G["t_end"]:
            panel.set_text("요청 종료: 기본 거버너로 복원\n(식는 구간)" if T > G["t_end"] else "대기")
        else:
            state = PHASE_KO.get(r["cpu_phase"], r["cpu_phase"]) + (" + GPU 작업" if r["gpu_on"] == "1" else "")
            rule_ko, evid = ko(r["rule"])
            panel.set_text(f"감지한 상태\n  {state}\n  (KV: {'디스크 swap' if r['kv'] == 'swap' else 'RAM'})\n"
                           f"테이블 규칙\n  {wrap(rule_ko)}\n"
                           f"적용\n  A720 {fmt_set(r['a720_set_mhz'])},  GPU {fmt_set(r['gpu_set_mhz'])}\n"
                           f"근거 (측정값)\n  {wrap(evid)}")
        ev = [e for e in G["events"] if e["t_rel"] <= T]
        logt.set_text("\n".join(f"{e['t_rel']:4.0f}s  {PHASE_KO.get(e['cpu'], e['cpu'])}{' + GPU' if e['gpu'] else ''}"
                                f" → A720 {fmt_set(e['a720_mhz'], '')}/GPU {fmt_set(e['gpu_mhz'], '')}".rstrip() for e in ev[-8:]))
        clock.set_text(f"t = {T:5.0f} s   (재생 {a.speed:.0f}배속)")
        if T >= t_max - 1e-6:
            mb = B["meta"]
            lines = ["결과 (기준선 대비, 열 = 표시 구간)"]
            for lab, R in [("governor", G)] + [(R["label"], R) for R in REFS]:
                m = R["meta"]
                ar = upto(R["trace"]["t"], R["area"], T=T)[1]
                bits = [f"열 {ar[-1] / ab:.2f}" if ar and ab > 0 else ""]
                bits = [b for b in bits if b]
                if m.get("prefill_s") and mb.get("prefill_s"):
                    bits.append(f"TTFT {m['prefill_s'] / mb['prefill_s']:.2f}")
                if m.get("cpu_Bps_on") and mb.get("cpu_Bps_on"):         # speed: higher = faster, so invert to a time ratio
                    bits.append(f"CPU(GPU중) {mb['cpu_Bps_on'] / m['cpu_Bps_on']:.2f}")
                if m.get("cpu_Bps_off") and mb.get("cpu_Bps_off"):
                    bits.append(f"CPU(그외) {mb['cpu_Bps_off'] / m['cpu_Bps_off']:.2f}")
                elif m.get("tpot_ms") and mb.get("tpot_ms"):
                    bits.append(f"CPU TPOT {m['tpot_ms'] / mb['tpot_ms']:.2f}")
                if m.get("gpu_tpot_ms") and mb.get("gpu_tpot_ms"):
                    bits.append(f"GPU {m['gpu_tpot_ms'] / mb['gpu_tpot_ms']:.2f}")
                lines.append(f"{lab}\n  " + " · ".join(bits))
            lines.append("시간·열 배율: 1보다 작을수록 좋음")
            summary.set_text("\n".join(lines))
        else:
            summary.set_text("")
        return []

    n = int(t_max / a.speed * a.fps) + 1
    times = [min(t_max, i * a.speed / a.fps) for i in range(n)] + [t_max] * a.fps     # hold the last frame for 1 s
    anim = animation.FuncAnimation(fig, lambda i: draw(times[i]), frames=len(times), blit=False)
    out = Path(a.out)
    if out.suffix.lower() == ".mp4" and shutil.which("ffmpeg"):
        anim.save(out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=6000))
    else:
        out = out.with_suffix(".gif")
        anim.save(out, writer=animation.PillowWriter(fps=a.fps))
    print("video:", out, f"({len(times)} frames, {len(times) / a.fps:.0f} s)")
    if a.still:
        draw(t_max)
        fig.savefig(a.still, dpi=100)
        print("still:", a.still)


if __name__ == "__main__":
    main()
