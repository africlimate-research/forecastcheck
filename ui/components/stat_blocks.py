# Authors: The africlimate AI team
# SPDX-License-Identifier: MIT
"""
Reusable statistical insight blocks for the Spatial Distribution
and Models Comparison tabs.

Public card renderers:
  - render_descriptive_stats()   — mean, std, median, IQR, skewness …
  - render_paired_ttest()        — paired t-test OR Wilcoxon+sign test (A vs B)
  - render_onesample_ttest()     — one-sample t-test vs H₀: μ = 0
  - render_desc_and_ttest_row()  — descriptive + test cards side by side
  - render_section_header()      — shared section separator

test parameter accepted by render_paired_ttest / render_desc_and_ttest_row:
  "ttest"    — parametric paired t-test (assumes normality)
  "wilcoxon" — Wilcoxon signed-rank + sign test (no distributional assumption)
  "both"     — show both cards side by side
"""
from __future__ import annotations

import math

import numpy as np
import streamlit as st
from ui.app_config.settings import higher_is_better

try:
    from scipy import stats as _sc
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── Pure numeric helpers ───────────────────────────────────────────────────

def _clean(arr: np.ndarray) -> np.ndarray:
    v = np.asarray(arr, dtype=float).ravel()
    return v[np.isfinite(v)]


def _describe(arr: np.ndarray) -> dict | None:
    v = _clean(arr)
    if v.size < 2:
        return None
    mean = float(np.mean(v))
    std  = float(np.std(v, ddof=1))
    p25  = float(np.percentile(v, 25))
    p75  = float(np.percentile(v, 75))
    skew = float(np.mean(((v - mean) / std) ** 3)) if std > 0 else 0.0
    return {
        "n":      v.size,
        "mean":   mean,
        "std":    std,
        "median": float(np.median(v)),
        "min":    float(np.min(v)),
        "max":    float(np.max(v)),
        "p25":    p25,
        "p75":    p75,
        "iqr":    p75 - p25,
        "skew":   skew,
    }


def _paired_t(a: np.ndarray, b: np.ndarray, higher_better: bool = False) -> dict | None:
    """Paired t-test of A vs B, oriented so "A better" means A's metric is
    preferable — smaller by default, larger when ``higher_better`` (correlation).
    ``p_better`` is the one-sided p-value for H₁: A better than B, and
    ``n_a_better``/``n_b_better`` count pixels in that same orientation."""
    a_f = np.asarray(a, dtype=float).ravel()
    b_f = np.asarray(b, dtype=float).ravel()
    if a_f.size != b_f.size:
        n = min(a_f.size, b_f.size)
        a_f, b_f = a_f[:n], b_f[:n]
    mask = np.isfinite(a_f) & np.isfinite(b_f)
    a_v, b_v = a_f[mask], b_f[mask]
    if a_v.size < 2:
        return None
    diff   = a_v - b_v
    n      = diff.size
    mean_d = float(np.mean(diff))
    std_d  = float(np.std(diff, ddof=1))
    se     = std_d / math.sqrt(n)
    t_stat = mean_d / se if se > 0 else 0.0
    df     = n - 1

    if _HAS_SCIPY:
        # one-sided: H₁ A < B  →  p_less  = P(T ≤ t | H₀)
        # one-sided: H₁ A > B  →  p_greater = P(T ≥ t | H₀)
        p_less    = float(_sc.t.cdf(t_stat, df))
        p_greater = float(_sc.t.sf(t_stat, df))
        t_crit    = float(_sc.t.ppf(0.975, df))
    else:
        z         = t_stat / math.sqrt(2)
        p_less    = float(0.5 * math.erfc(-z))
        p_greater = float(0.5 * math.erfc(z))
        t_crit    = 1.96

    return {
        "t": t_stat, "df": df, "n": n,
        "mean_d":    mean_d,
        "ci_lo":     mean_d - t_crit * se,
        "ci_hi":     mean_d + t_crit * se,
        "p_better":  p_greater if higher_better else p_less,   # H₁: A better
        "n_a_better": int(np.sum(diff > 0) if higher_better else np.sum(diff < 0)),
        "n_b_better": int(np.sum(diff < 0) if higher_better else np.sum(diff > 0)),
    }


_MAX_WILCOXON_N = 200_000   # subsample threshold for Wilcoxon (keeps it fast)


def _wilcoxon(a: np.ndarray, b: np.ndarray, higher_better: bool = False) -> dict | None:
    """Wilcoxon signed-rank test + sign test for pixel-wise pairs.

    Oriented like :func:`_paired_t`: "A better" means A's metric is preferable
    (smaller by default, larger when ``higher_better``); both tests are
    one-sided for H₁: A better than B."""
    a_f = np.asarray(a, dtype=float).ravel()
    b_f = np.asarray(b, dtype=float).ravel()
    if a_f.size != b_f.size:
        n = min(a_f.size, b_f.size)
        a_f, b_f = a_f[:n], b_f[:n]
    mask = np.isfinite(a_f) & np.isfinite(b_f)
    a_v, b_v = a_f[mask], b_f[mask]
    diff = a_v - b_v
    n_total = diff.size

    n_a_better = int(np.sum(diff > 0) if higher_better else np.sum(diff < 0))
    n_b_better = int(np.sum(diff < 0) if higher_better else np.sum(diff > 0))
    n_tied     = n_total - n_a_better - n_b_better
    n_nontied  = n_a_better + n_b_better

    # ── Sign test (exact binomial, always on full data) ────────────────────
    if _HAS_SCIPY and n_nontied > 0:
        p_sign = float(_sc.binomtest(n_a_better, n=n_nontied, p=0.5,
                                     alternative="greater").pvalue)
    elif n_nontied > 0:
        # Normal approximation for binomial when scipy unavailable
        z = (n_a_better - n_nontied / 2) / math.sqrt(n_nontied / 4)
        p_sign = float(0.5 * math.erfc(-z / math.sqrt(2)))
    else:
        p_sign = 1.0

    # ── Wilcoxon signed-rank (subsample for large n) ───────────────────────
    nonzero_mask = diff != 0
    n_nonzero    = int(np.sum(nonzero_mask))
    w_stat: float | None = None
    p_wilcoxon:  float | None = None
    sampled = False

    if _HAS_SCIPY and n_nonzero >= 10:
        a_nz = a_v[nonzero_mask]
        b_nz = b_v[nonzero_mask]
        if n_nonzero > _MAX_WILCOXON_N:
            rng  = np.random.default_rng(0)
            idx  = rng.choice(n_nonzero, _MAX_WILCOXON_N, replace=False)
            a_nz, b_nz = a_nz[idx], b_nz[idx]
            sampled = True
        try:
            res        = _sc.wilcoxon(a_nz, b_nz,
                                      alternative="greater" if higher_better else "less")
            w_stat     = float(res.statistic)
            p_wilcoxon = float(res.pvalue)
        except Exception:
            pass

    return {
        "n":          n_total,
        "n_a_better": n_a_better,
        "n_b_better": n_b_better,
        "n_tied":     n_tied,
        "n_nontied":  n_nontied,
        "p_sign":     p_sign,
        "w_stat":     w_stat,
        "p_wilcoxon": p_wilcoxon,
        "sampled":    sampled,
        "n_wilcoxon": min(n_nonzero, _MAX_WILCOXON_N),
    }


def _onesample_t(arr: np.ndarray) -> dict | None:
    v = _clean(arr)
    if v.size < 2:
        return None
    mean_v = float(np.mean(v))
    std_v  = float(np.std(v, ddof=1))
    n      = v.size
    se     = std_v / math.sqrt(n)
    t_stat = mean_v / se if se > 0 else 0.0
    df     = n - 1

    if _HAS_SCIPY:
        p_val  = float(_sc.t.sf(abs(t_stat), df) * 2)
        t_crit = float(_sc.t.ppf(0.975, df))
    else:
        p_val  = float(math.erfc(abs(t_stat) / math.sqrt(2)))
        t_crit = 1.96

    return {
        "t": t_stat, "df": df, "p": p_val, "n": n,
        "mean": mean_v,
        "ci_lo": mean_v - t_crit * se,
        "ci_hi": mean_v + t_crit * se,
        "reject": p_val < 0.05,
    }


# ── HTML primitives ────────────────────────────────────────────────────────

_F = "font-family:Inter,sans-serif;"

def _card(title: str, accent: str, body: str) -> str:
    return (
        f"<div style='background:#FFFFFF;border:1px solid #E5E7EB;"
        f"border-top:3px solid {accent};border-radius:8px;overflow:hidden;"
        f"box-shadow:0 1px 4px rgba(0,0,0,0.06);'>"
        f"<div style='{_F}font-size:0.6rem;font-weight:700;"
        f"letter-spacing:0.12em;text-transform:uppercase;color:{accent};"
        f"padding:6px 14px 6px;border-bottom:1px solid #F3F4F6;'>"
        f"{title}</div>"
        f"{body}"
        f"</div>"
    )


def _label(text: str) -> str:
    return (
        f"<div style='{_F}font-size:9px;font-weight:700;letter-spacing:0.1em;"
        f"text-transform:uppercase;color:#9CA3AF;margin-bottom:3px;'>{text}</div>"
    )


def _p_badge(p: float) -> str:
    if p < 0.001:
        label, bg, color = "p &lt; 0.001", "rgba(10,153,56,0.12)", "#0a9938"
    elif p < 0.05:
        label, bg, color = f"p = {p:.3f}", "rgba(242,176,41,0.18)", "#92580a"
    else:
        label, bg, color = f"p = {p:.3f}", "rgba(186,199,201,0.25)", "#6B7280"
    return (
        f"<span style='background:{bg};color:{color};border-radius:99px;"
        f"padding:3px 12px;font-size:11px;font-weight:700;"
        f"{_F}white-space:nowrap;'>{label}</span>"
    )


_NO_DATA = f"<div style='color:#9CA3AF;font-size:12px;padding:16px 14px;{_F}'>No data</div>"


def _wilcoxon_html(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    name_a: str,
    name_b: str,
    unit:   str,
    accent: str,
    higher_better: bool = False,
) -> str:
    r = _wilcoxon(vals_a, vals_b, higher_better)
    if r is None:
        return _card("🔬 Wilcoxon + Sign Test", accent, _NO_DATA)

    col_a = "#0a9938"
    col_b = "#213b7d" if "clim" in name_b.lower() else "#f2b029"

    pct_a = r["n_a_better"] / r["n"] * 100
    pct_b = r["n_b_better"] / r["n"] * 100

    # Hypothesis — H₁ is always "A better", spelled in the metric's direction.
    h0_rel, h1_rel = ("≤", "&gt;") if higher_better else ("≥", "&lt;")
    hyp = (
        f"<div style='padding:6px 14px 6px;background:#F9FAFB;"
        f"border-bottom:1px solid #F3F4F6;'>"
        f"<div style='{_F}font-size:10px;color:#374151;'>"
        f"<span style='font-weight:700;color:#9CA3AF;'>H₀</span>&ensp;"
        f"median({name_a} − {name_b}) {h0_rel} 0</div>"
        f"<div style='{_F}font-size:10px;color:#374151;margin-top:3px;'>"
        f"<span style='font-weight:700;color:#9CA3AF;'>H₁</span>&ensp;"
        f"{name_a} {h1_rel} {name_b} — i.e. {name_a} better"
        f"<span style='color:#9CA3AF;font-size:9px;'>&ensp;(α = 0.05 · no normality assumed)</span></div>"
        f"</div>"
    )

    # Wilcoxon row
    def _stat(lbl: str, val: str, border_right: bool = True) -> str:
        br = "border-right:1px solid #F3F4F6;" if border_right else ""
        return (
            f"<div style='flex:1;text-align:center;{br}padding:7px 6px;'>"
            + _label(lbl)
            + f"<div style='{_F}font-size:13px;font-weight:800;color:#171d1a;'>{val}</div>"
            f"</div>"
        )

    sample_note = (
        f"<span style='color:#9CA3AF;font-size:9px;'>&ensp;(sample n={r['n_wilcoxon']:,})</span>"
        if r["sampled"] else ""
    )
    if r["w_stat"] is not None and r["p_wilcoxon"] is not None:
        wilcoxon_row = (
            f"<div style='border-bottom:1px solid #F3F4F6;'>"
            f"<div style='padding:4px 14px 2px;'>"
            + _label(f"Wilcoxon signed-rank{sample_note}")
            + f"</div>"
            f"<div style='display:flex;'>"
            + _stat("W statistic", f"{r['w_stat']:.0f}")
            + _stat("n pairs", f"{r['n_wilcoxon']:,}")
            + _stat("p-value (H₁: A better)", _p_badge(r["p_wilcoxon"]).replace(
                "border-radius:99px;", "border-radius:4px;font-size:10px;"
            ), border_right=False)
            + f"</div></div>"
        )
        w_sig  = r["p_wilcoxon"] < 0.05
    else:
        wilcoxon_row = (
            f"<div style='padding:6px 14px;border-bottom:1px solid #F3F4F6;"
            f"color:#9CA3AF;{_F}font-size:11px;'>Wilcoxon: scipy not available or n too small</div>"
        )
        w_sig = None

    # Sign test row
    s_sig = r["p_sign"] < 0.05
    sign_row = (
        f"<div style='border-bottom:1px solid #F3F4F6;'>"
        f"<div style='padding:4px 14px 2px;'>" + _label("Sign test (binomial)") + f"</div>"
        f"<div style='display:flex;'>"
        + _stat(f"{name_a} wins", f"{r['n_a_better']:,}")
        + _stat(f"{name_b} wins", f"{r['n_b_better']:,}")
        + _stat("p-value (H₁: A wins more often)", _p_badge(r["p_sign"]).replace(
            "border-radius:99px;", "border-radius:4px;font-size:10px;"
        ), border_right=False)
        + f"</div></div>"
    )

    # Pixel-win bar
    bar = (
        f"<div style='padding:7px 14px;border-bottom:1px solid #F3F4F6;'>"
        + _label("Pixel-wise wins")
        + f"<div style='display:flex;height:8px;border-radius:4px;overflow:hidden;margin:6px 0;'>"
        f"<div style='width:{max(pct_a, 0.3):.1f}%;background:{col_a};'></div>"
        f"<div style='width:{max(pct_b, 0.3):.1f}%;background:{col_b};'></div>"
        f"</div>"
        f"<div style='display:flex;justify-content:space-between;'>"
        f"<span style='{_F}font-size:10px;font-weight:700;color:{col_a};'>"
        f"▮ {name_a}&ensp;{pct_a:.1f}%</span>"
        f"<span style='{_F}font-size:10px;font-weight:700;color:{col_b};'>"
        f"{pct_b:.1f}%&ensp;▮ {name_b}</span>"
        f"</div></div>"
    )

    # Overall verdict: both tests agree → stronger conclusion
    if w_sig is not None:
        both_sig  = w_sig and s_sig
        any_sig   = w_sig or s_sig
        verdict   = ("Both significant" if both_sig
                     else "Partially significant" if any_sig
                     else "Not significant")
        vc = "#0a9938" if both_sig else ("#f2b029" if any_sig else "#6B7280")
        icon = "✓✓" if both_sig else ("✓" if any_sig else "✗")
        p_display = min(r["p_wilcoxon"], r["p_sign"])
    else:
        verdict, vc, icon = ("Significant" if s_sig else "Not significant",
                             "#0a9938" if s_sig else "#6B7280",
                             "✓" if s_sig else "✗")
        p_display = r["p_sign"]

    result = (
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"padding:10px 14px;'>"
        f"<span style='{_F}font-size:13px;font-weight:800;color:{vc};'>"
        f"{icon}&ensp;{verdict}</span>"
        + _p_badge(p_display)
        + "</div>"
    )

    return _card("🔬 Wilcoxon + Sign Test", accent, hyp + wilcoxon_row + sign_row + bar + result)


# ── Public API ─────────────────────────────────────────────────────────────

def render_section_header(title: str) -> None:
    st.markdown(
        f"<p style='font-family:Inter,sans-serif;font-size:11px;font-weight:700;"
        f"color:#4B5563;letter-spacing:0.1em;text-transform:uppercase;"
        f"margin:24px 0 12px;'>{title}</p>",
        unsafe_allow_html=True,
    )


def _descriptive_stats_html(
    vals:   np.ndarray,
    name:   str,
    unit:   str,
    accent: str,
    higher_better: bool = False,
) -> str:
    u = f" ({unit})" if unit else ""
    s = _describe(vals)
    if not s:
        return _card(f"📊 Descriptive Stats · {name}", accent, _NO_DATA)

    sign = "+" if s["mean"] > 0 else ""
    # vals is the A − B difference: color the mean by whether it favours A
    # (green) or B (gold), in the metric's own direction.
    a_favoured = s["mean"] > 0 if higher_better else s["mean"] < 0
    mean_color = "#374151" if s["mean"] == 0 else ("#0a9938" if a_favoured else "#f2b029")

    # Hero: mean (big) + N pixels side by side
    hero = (
        f"<div style='display:flex;padding:8px 14px 8px;border-bottom:1px solid #F3F4F6;'>"
        f"<div style='flex:1;'>"
        + _label(f"Mean{u}")
        + f"<div style='{_F}font-size:24px;font-weight:800;color:{mean_color};line-height:1;'>"
        f"{sign}{s['mean']:.4f}</div>"
        f"</div>"
        f"<div style='width:1px;background:#F3F4F6;margin:0 14px;'></div>"
        f"<div style='flex:1;'>"
        + _label("N Pixels")
        + f"<div style='{_F}font-size:24px;font-weight:800;color:#171d1a;line-height:1;'>"
        f"{s['n']:,}</div>"
        f"</div>"
        f"</div>"
    )

    # Secondary stats: 2-column grid
    def _cell(lbl: str, val: str) -> str:
        return (
            f"<div>"
            + _label(lbl)
            + f"<div style='{_F}font-size:12px;font-weight:700;color:#374151;'>{val}</div>"
            f"</div>"
        )

    grid = (
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px 24px;"
        f"padding:8px 14px 10px;'>"
        + _cell(f"Std{u}", f"{s['std']:.4f}")
        + _cell("IQR", f"{s['iqr']:.4f}")
        + _cell("Median", f"{s['median']:.4f}")
        + _cell("Skewness", f"{s['skew']:.3f}")
        + _cell("Min", f"{s['min']:.3f}")
        + _cell("Max", f"{s['max']:.3f}")
        + "</div>"
    )

    return _card(f"📊 Descriptive Stats · {name}", accent, hero + grid)


def _paired_ttest_html(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    name_a: str,
    name_b: str,
    unit:   str,
    accent: str,
    higher_better: bool = False,
) -> str:
    u = f" ({unit})" if unit else ""
    r = _paired_t(vals_a, vals_b, higher_better)
    if r is None:
        return _card("🧪 One-Sided Paired T-Test", accent, _NO_DATA)

    pct_a = r["n_a_better"] / r["n"] * 100
    pct_b = r["n_b_better"] / r["n"] * 100
    p_val = r["p_better"]
    sig   = p_val < 0.05

    col_a     = "#0a9938"
    col_b     = "#213b7d" if "clim" in name_b.lower() else "#f2b029"
    res_color = "#0a9938" if sig else "#6B7280"

    # Hypothesis block — H₁ is always "A better", spelled in the metric's direction.
    h0_rel, h1_rel = ("≤", "&gt;") if higher_better else ("≥", "&lt;")
    hyp = (
        f"<div style='padding:6px 14px 6px;background:#F9FAFB;"
        f"border-bottom:1px solid #F3F4F6;'>"
        f"<div style='{_F}font-size:10px;color:#374151;'>"
        f"<span style='font-weight:700;color:#9CA3AF;'>H₀</span>&ensp;"
        f"mean({name_a} − {name_b}) {h0_rel} 0</div>"
        f"<div style='{_F}font-size:10px;color:#374151;margin-top:3px;'>"
        f"<span style='font-weight:700;color:#9CA3AF;'>H₁</span>&ensp;"
        f"{name_a} {h1_rel} {name_b} — i.e. {name_a} better"
        f"<span style='color:#9CA3AF;font-size:9px;'>&ensp;(α = 0.05)</span></div>"
        f"</div>"
    )

    # Key stats row: t / df / n
    def _stat(lbl: str, val: str, border_right: bool = True) -> str:
        br = "border-right:1px solid #F3F4F6;" if border_right else ""
        return (
            f"<div style='flex:1;text-align:center;{br}padding:7px 6px;'>"
            + _label(lbl)
            + f"<div style='{_F}font-size:13px;font-weight:800;color:#171d1a;'>{val}</div>"
            f"</div>"
        )

    stats = (
        f"<div style='display:flex;border-bottom:1px solid #F3F4F6;'>"
        + _stat("t-stat", f"{r['t']:.2f}")
        + _stat("df", f"{r['df']:,}")
        + _stat("n pairs", f"{r['n']:,}", border_right=False)
        + "</div>"
    )

    # 95% CI
    ci = (
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"padding:5px 14px;border-bottom:1px solid #F3F4F6;'>"
        + _label(f"95% CI{u}")
        + f"<span style='{_F}font-size:11px;font-weight:700;color:#374151;'>"
        f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]</span>"
        f"</div>"
    )

    # Visual pixel-win bar
    bar = (
        f"<div style='padding:7px 14px;border-bottom:1px solid #F3F4F6;'>"
        + _label("Pixel-wise wins")
        + f"<div style='display:flex;height:8px;border-radius:4px;overflow:hidden;margin:6px 0;'>"
        f"<div style='width:{max(pct_a, 0.3):.1f}%;background:{col_a};'></div>"
        f"<div style='width:{max(pct_b, 0.3):.1f}%;background:{col_b};'></div>"
        f"</div>"
        f"<div style='display:flex;justify-content:space-between;'>"
        f"<span style='{_F}font-size:10px;font-weight:700;color:{col_a};'>"
        f"▮ {name_a}&ensp;{pct_a:.1f}%</span>"
        f"<span style='{_F}font-size:10px;font-weight:700;color:{col_b};'>"
        f"{pct_b:.1f}%&ensp;▮ {name_b}</span>"
        f"</div>"
        f"</div>"
    )

    # Result
    verdict = "Significant" if sig else "Not Significant"
    result = (
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"padding:10px 14px;'>"
        f"<span style='{_F}font-size:13px;font-weight:800;color:{res_color};'>"
        f"{'✓' if sig else '✗'}&ensp;{verdict}</span>"
        + _p_badge(p_val)
        + "</div>"
    )

    return _card("🧪 One-Sided Paired T-Test", accent, hyp + stats + ci + bar + result)


# ── Public API ─────────────────────────────────────────────────────────────

def render_descriptive_stats(
    vals:       np.ndarray,
    name:       str,
    metric_key: str,
    unit:       str = "",
    accent:     str = "#0a9938",
) -> None:
    st.markdown(
        _descriptive_stats_html(vals, name, unit, accent, higher_is_better(metric_key)),
        unsafe_allow_html=True,
    )


def render_paired_ttest(
    vals_a:     np.ndarray,
    vals_b:     np.ndarray,
    name_a:     str,
    name_b:     str,
    metric_key: str,
    unit:       str = "",
    accent:     str = "#213b7d",
    test:       str = "ttest",
) -> None:
    """Render statistical test card(s).

    test: "ttest" | "wilcoxon" | "both"
    """
    hb = higher_is_better(metric_key)
    if test == "both":
        html_t = _paired_ttest_html(vals_a, vals_b, name_a, name_b, unit, accent, hb)
        html_w = _wilcoxon_html(vals_a, vals_b, name_a, name_b, unit, accent, hb)
        # Native st.columns instead of a CSS flex div — the latter gets
        # wrapped by Streamlit's own container which collapses the flex,
        # leaving only one card visible.
        col_t, col_w = st.columns(2)
        with col_t:
            st.markdown(html_t, unsafe_allow_html=True)
        with col_w:
            st.markdown(html_w, unsafe_allow_html=True)
    elif test == "wilcoxon":
        st.markdown(
            _wilcoxon_html(vals_a, vals_b, name_a, name_b, unit, accent, hb),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            _paired_ttest_html(vals_a, vals_b, name_a, name_b, unit, accent, hb),
            unsafe_allow_html=True,
        )


def render_desc_and_ttest_row(
    vals:        np.ndarray,
    name_desc:   str,
    vals_a:      np.ndarray,
    vals_b:      np.ndarray,
    name_a:      str,
    name_b:      str,
    metric_key:  str,
    unit:        str = "",
    accent_desc: str = "#213b7d",
    accent_test: str = "#213b7d",
    test:        str = "ttest",
) -> None:
    """Render descriptive-stats and test card(s) side-by-side.

    test: "ttest" | "wilcoxon" | "both"
    """
    hb = higher_is_better(metric_key)
    html_desc = _descriptive_stats_html(vals, name_desc, unit, accent_desc, hb)

    # Native st.columns instead of a CSS flex div — the flex div gets
    # wrapped by Streamlit's own container which can collapse the layout
    # so only one card is visible. Columns guarantee side-by-side rendering.
    if test == "both":
        html_t = _paired_ttest_html(vals_a, vals_b, name_a, name_b, unit, accent_test, hb)
        html_w = _wilcoxon_html(vals_a, vals_b, name_a, name_b, unit, accent_test, hb)
        col_d, col_t, col_w = st.columns(3)
        with col_d:
            st.markdown(html_desc, unsafe_allow_html=True)
        with col_t:
            st.markdown(html_t, unsafe_allow_html=True)
        with col_w:
            st.markdown(html_w, unsafe_allow_html=True)
    elif test == "wilcoxon":
        html_w = _wilcoxon_html(vals_a, vals_b, name_a, name_b, unit, accent_test, hb)
        col_d, col_w = st.columns(2)
        with col_d:
            st.markdown(html_desc, unsafe_allow_html=True)
        with col_w:
            st.markdown(html_w, unsafe_allow_html=True)
    else:
        html_t = _paired_ttest_html(vals_a, vals_b, name_a, name_b, unit, accent_test, hb)
        col_d, col_t = st.columns(2)
        with col_d:
            st.markdown(html_desc, unsafe_allow_html=True)
        with col_t:
            st.markdown(html_t, unsafe_allow_html=True)


def render_onesample_ttest(
    vals:       np.ndarray,
    name:       str,
    metric_key: str,
    unit:       str = "",
    accent:     str = "#213b7d",
) -> None:
    """Render a one-sample t-test card: H₀: mean error = 0."""
    u = f" ({unit})" if unit else ""
    r = _onesample_t(vals)
    if r is None:
        st.markdown(_card("🧪 One-Sample T-Test", accent, _NO_DATA), unsafe_allow_html=True)
        return

    sig     = r["reject"]
    vc      = "#0a9938" if sig else "#6B7280"
    verdict = "Reject H₀" if sig else "Fail to reject H₀"
    mean_color = "#f2b029" if r["mean"] > 0 else ("#0a9938" if r["mean"] < 0 else "#374151")

    hyp = (
        f"<div style='padding:6px 14px;background:#F9FAFB;border-bottom:1px solid #F3F4F6;'>"
        f"<div style='{_F}font-size:10px;color:#374151;'>"
        f"<span style='font-weight:700;color:#9CA3AF;'>H₀</span>&ensp;"
        f"mean {name} error = 0</div>"
        f"<div style='{_F}font-size:9px;color:#9CA3AF;margin-top:2px;'>"
        f"n = {r['n']:,} pixels&ensp;·&ensp;α = 0.05</div>"
        f"</div>"
    )

    hero = (
        f"<div style='display:flex;padding:8px 14px 8px;border-bottom:1px solid #F3F4F6;'>"
        f"<div style='flex:1;'>"
        + _label(f"Mean error{u}")
        + f"<div style='{_F}font-size:24px;font-weight:800;color:{mean_color};line-height:1;'>"
        f"{r['mean']:+.4f}</div>"
        f"</div>"
        f"<div style='width:1px;background:#F3F4F6;margin:0 14px;'></div>"
        f"<div style='flex:1;'>"
        + _label("t-statistic")
        + f"<div style='{_F}font-size:24px;font-weight:800;color:#171d1a;line-height:1;'>"
        f"{r['t']:.2f}</div>"
        f"</div>"
        f"</div>"
    )

    ci = (
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"padding:5px 14px;border-bottom:1px solid #F3F4F6;'>"
        + _label(f"95% CI{u}")
        + f"<span style='{_F}font-size:11px;font-weight:700;color:#374151;'>"
        f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]</span>"
        f"</div>"
    )

    result = (
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"padding:10px 14px;'>"
        f"<span style='{_F}font-size:13px;font-weight:800;color:{vc};'>"
        f"{'✓' if sig else '✗'}&ensp;{verdict}</span>"
        + _p_badge(r["p"])
        + "</div>"
    )

    st.markdown(
        _card("🧪 One-Sample T-Test", accent, hyp + hero + ci + result),
        unsafe_allow_html=True,
    )
