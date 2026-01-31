import os
from typing import List, Dict, Any, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# —— 官方 FaceMesh 区域 ——
import mediapipe as mp
mp_fm = mp.solutions.face_mesh



# ---------------------------------------------------------------------
# 字体设置（中文显示）
# ---------------------------------------------------------------------

def _set_chinese_font():
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception:
        pass

_set_chinese_font()

# ---------------------------------------------------------------------
# 部位 → 候选列集合（用于选择 Top-3 部位）
# ---------------------------------------------------------------------
REGION_DEFS: Dict[str, Dict[str, Sequence[Sequence[str]]]] = {
    "mouth": {
        "label": "Mouth",
        "candidates": [
            ["mouthSmileLeft", "mouthSmileRight"],
            ["mouthFrownLeft", "mouthFrownRight"],
            ["mouthStretchLeft", "mouthStretchRight"],
            ["mouthUpperUpLeft", "mouthUpperUpRight"],
            ["mouthLowerDownLeft", "mouthLowerDownRight"],
            ["mouthDimpleLeft", "mouthDimpleRight"],
            ["jawOpen"],
            ["mouthPucker"], ["mouthFunnel"],
        ],
    },
    "eyebrows": {
        "label": "Eyebrows",
        "candidates": [["browOuterUpLeft", "browOuterUpRight"], ["browDownLeft", "browDownRight"], ["browInnerUp"]],
    },
    "eyes": {
        "label": "Eyes",
        "candidates": [["eyeWideLeft", "eyeWideRight"], ["eyeSquintLeft", "eyeSquintRight"], ["eyeBlinkLeft", "eyeBlinkRight"]],
    },
    "nose": {"label": "Nose", "candidates": [["noseSneerLeft", "noseSneerRight"]]},
    "cheeks": {"label": "Cheeks", "candidates": [["cheekPuff"], ["cheekSquintLeft", "cheekSquintRight"]]},
    "jaw": {"label": "Jaw", "candidates": [["jawOpen"], ["jawForward"]]},
}

CATEGORY_CN = {"neutral": "neutral", "surprise": "surprise", "happy": "happy", "sad": "sad", "angry": "angry", "disgust": "disgust", "fear": "fear", "contempt": "contempt", "unknown": "unknown"}

# ---------------------------------------------------------------------
# 官方连接集合 → 唯一点索引列表
# ---------------------------------------------------------------------

def _indices_from_connections(conns) -> List[int]:
    s: set = set()
    for a, b in conns:
        s.add(int(a)); s.add(int(b))
    return sorted(s)

OFFICIAL = {
    'lips_ALL': _indices_from_connections(mp_fm.FACEMESH_LIPS),
    'eye_L': _indices_from_connections(mp_fm.FACEMESH_LEFT_EYE),
    'eye_R': _indices_from_connections(mp_fm.FACEMESH_RIGHT_EYE),
    'brow_L': _indices_from_connections(mp_fm.FACEMESH_LEFT_EYEBROW),
    'brow_R': _indices_from_connections(mp_fm.FACEMESH_RIGHT_EYEBROW),
    'nose_ALL': _indices_from_connections(mp_fm.FACEMESH_NOSE),
    'oval_ALL': _indices_from_connections(mp_fm.FACEMESH_FACE_OVAL),
}
OFFICIAL['eye_ALL'] = sorted(set(OFFICIAL['eye_L']) | set(OFFICIAL['eye_R']))
OFFICIAL['brow_ALL'] = sorted(set(OFFICIAL['brow_L']) | set(OFFICIAL['brow_R']))

# 双颊（近似点集）
CHEEK_L = [50, 101, 118, 123, 147, 148, 149, 176, 205, 206, 207, 187, 198, 209, 217, 126]
CHEEK_R = [280, 330, 349, 352, 377, 376, 401, 367, 425, 411, 427, 418, 406, 335, 345, 266]

# ---------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------

def _numeric_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _range(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return 0.0
    try:
        return float(series.max() - series.min())
    except Exception:
        return 0.0


def _peak_idx(series: pd.Series) -> Optional[int]:
    if series is None or len(series) == 0:
        return None
    try:
        return int(series.values.argmax())
    except Exception:
        return None


def _score_candidate(std_ts: pd.DataFrame, usr_ts: pd.DataFrame, cols: Sequence[str]) -> Optional[float]:
    # 列必须同时存在
    for c in cols:
        if c not in std_ts.columns or c not in usr_ts.columns:
            return None
    # 单帧模式：按强度之和评分，保证图片输入可用
    if len(std_ts) < 2 or len(usr_ts) < 2:
        try:
            return float(sum(abs(float(std_ts[c].iloc[-1])) + abs(float(usr_ts[c].iloc[-1])) for c in cols))
        except Exception:
            return None
    # 多帧：按波动范围评分
    return float(sum(_range(std_ts[c]) + _range(usr_ts[c]) for c in cols))


def select_top_regions(std_ts: Optional[pd.DataFrame], usr_ts: Optional[pd.DataFrame], k: int = 3) -> List[Dict[str, Any]]:
    if std_ts is None or usr_ts is None or std_ts.empty or usr_ts.empty:
        return []
    std_ts = std_ts[_numeric_cols(std_ts)]
    usr_ts = usr_ts[_numeric_cols(usr_ts)]
    if std_ts.empty or usr_ts.empty:
        return []
    regions: List[Dict[str, Any]] = []
    for key, spec in REGION_DEFS.items():
        best_cols, best_score = None, -1.0
        for cols in spec["candidates"]:
            sc = _score_candidate(std_ts, usr_ts, cols)
            if sc is not None and sc > best_score:
                best_score, best_cols = sc, list(cols)
        if best_cols is not None and best_score > 0:
            regions.append({"key": key, "label": spec["label"], "columns": best_cols, "score": float(best_score)})
    regions.sort(key=lambda x: x["score"], reverse=True)
    return regions[:k]

# —— 侧别判断 ——

def _side_of_col(col: str) -> str:
    if col.endswith('Left'): return 'L'
    if col.endswith('Right'): return 'R'
    return 'ALL'

# —— 颜色与转换 ——
PALETTE_HEX = {
    'Std-L': '#1f77b4',  # blue
    'Std-R': '#ff7f0e',  # orange
    'Std-ALL': '#1f77b4',
    'Usr-L': '#2ca02c',  # green
    'Usr-R': '#d62728',  # red
    'Usr-ALL': '#d62728',
}

def _hex_to_bgr(h: str) -> Tuple[int, int, int]:
    h = h.lstrip('#')
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return (b, g, r)

# —— 区域索引：依据被选列的侧别，返回需要绘制的点集 ——

def _mid_x(landmarks: np.ndarray) -> float:
    return float((landmarks[33, 0] + landmarks[263, 0]) / 2.0)


def _split_by_mid_x(indices: List[int], landmarks: np.ndarray) -> Tuple[List[int], List[int]]:
    midx = _mid_x(landmarks)
    left = [i for i in indices if landmarks[i, 0] < midx]
    right = [i for i in indices if landmarks[i, 0] >= midx]
    return left, right


def _want_sides(cols: Sequence[str]) -> Tuple[bool, bool]:
    return (any(c.endswith('Left') for c in cols), any(c.endswith('Right') for c in cols))


def _region_indices(region_key: str, cols: Sequence[str], landmarks: np.ndarray) -> Dict[str, List[int]]:
    has_L, has_R = _want_sides(cols)
    if region_key == 'mouth':
        all_idx = OFFICIAL['lips_ALL']
        if has_L or has_R:
            L, R = _split_by_mid_x(all_idx, landmarks)
            out = {}
            if has_L: out['L'] = L
            if has_R: out['R'] = R
            return out or {'ALL': all_idx}
        return {'ALL': all_idx}
    if region_key == 'eyes':
        if has_L or has_R:
            out = {}
            if has_L: out['L'] = OFFICIAL['eye_L']
            if has_R: out['R'] = OFFICIAL['eye_R']
            return out
        return {'ALL': OFFICIAL['eye_ALL']}
    if region_key == 'eyebrows':
        if has_L or has_R:
            out = {}
            if has_L: out['L'] = OFFICIAL['brow_L']
            if has_R: out['R'] = OFFICIAL['brow_R']
            return out
        return {'ALL': OFFICIAL['brow_ALL']}
    if region_key == 'nose':
        all_idx = OFFICIAL['nose_ALL']
        if has_L or has_R:
            L, R = _split_by_mid_x(all_idx, landmarks)
            out = {}
            if has_L: out['L'] = L
            if has_R: out['R'] = R
            return out or {'ALL': all_idx}
        return {'ALL': all_idx}
    if region_key == 'jaw':
        return {'ALL': OFFICIAL['oval_ALL']}
    if region_key == 'cheeks':
        if has_L or has_R:
            out = {}
            if has_L: out['L'] = CHEEK_L
            if has_R: out['R'] = CHEEK_R
            return out
        return {'ALL': sorted(set(CHEEK_L) | set(CHEEK_R))}
    return {'ALL': []}

# ---------------------------------------------------------------------
# 中性帧选择与对比图
# ---------------------------------------------------------------------

def _neutral_index(ts: Optional[pd.DataFrame]) -> Optional[int]:
    if ts is None or ts.empty:
        return None
    try:
        arr = ts[_numeric_cols(ts)].fillna(0.0).to_numpy(dtype=float)
        energy = np.linalg.norm(arr, axis=1)
        return int(np.argmin(energy))
    except Exception:
        return None


def create_neutral_comparison(std_frames: List[str], usr_frames: List[str],
                              std_ts: Optional[pd.DataFrame], usr_ts: Optional[pd.DataFrame],
                              out_path: str) -> str:
    i_std = _neutral_index(std_ts); i_usr = _neutral_index(usr_ts)
    if i_std is None or i_usr is None: return ""
    if i_std >= len(std_frames) or i_usr >= len(usr_frames): return ""
    std_img = cv2.imread(std_frames[i_std]); usr_img = cv2.imread(usr_frames[i_usr])
    if std_img is None or usr_img is None: return ""
    h = max(std_img.shape[0], usr_img.shape[0])
    def _rh(img, h): r = h / img.shape[0]; return cv2.resize(img, (int(img.shape[1]*r), h))
    std_r = _rh(std_img, h); usr_r = _rh(usr_img, h)
    cv2.putText(std_r, "Neutral face comparison", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    out = np.hstack((std_r, usr_r)); cv2.imwrite(out_path, out); return out_path

# ---------------------------------------------------------------------
# 区域关键点可视化（与时序图同色）
# ---------------------------------------------------------------------

def _draw_points(img: np.ndarray, landmarks: Optional[np.ndarray], idx_list: Sequence[int], color_bgr: Tuple[int, int, int]):
    if img is None or landmarks is None: return
    h, w = img.shape[:2]
    for i in idx_list:
        if i < 0 or i >= landmarks.shape[0]:
            continue
        x, y = int(landmarks[i, 0] * w), int(landmarks[i, 1] * h)
        cv2.circle(img, (x, y), 2, color_bgr, -1)


def create_region_keypoints_image(std_img_path: str, usr_img_path: str,
                                  std_landmarks: Optional[np.ndarray], usr_landmarks: Optional[np.ndarray],
                                  region: Dict[str, Any], out_path: str,
                                  palette_hex: Dict[str, str]) -> str:
    std_img = cv2.imread(std_img_path); usr_img = cv2.imread(usr_img_path)
    if std_img is None or usr_img is None: return ""

    key = region['key']; cols = region['columns']
    std_sets = _region_indices(key, cols, std_landmarks) if std_landmarks is not None else {'ALL': []}
    usr_sets = _region_indices(key, cols, usr_landmarks) if usr_landmarks is not None else {'ALL': []}

    std_vis = std_img.copy(); usr_vis = usr_img.copy()
    for side, idxs in std_sets.items():
        color = _hex_to_bgr(palette_hex.get(f'Std-{side}', PALETTE_HEX['Std-ALL']))
        _draw_points(std_vis, std_landmarks, idxs, color)
    for side, idxs in usr_sets.items():
        color = _hex_to_bgr(palette_hex.get(f'Usr-{side}', PALETTE_HEX['Usr-ALL']))
        _draw_points(usr_vis, usr_landmarks, idxs, color)

    h = max(std_vis.shape[0], usr_vis.shape[0])
    def _rh(img, H): r = H / img.shape[0]; return cv2.resize(img, (int(img.shape[1]*r), H))
    out_img = np.hstack((_rh(std_vis, h), _rh(usr_vis, h)))
    title = f"{region['label']} — landmark overlay (AU‑peak keyframe)"
    cv2.putText(out_img, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(out_path, out_img); return out_path

# ---------------------------------------------------------------------
# 三部位时序图（纵轴自适应，返回同色调色盘）
# ---------------------------------------------------------------------

def _pretty_label(col: str, video_tag: str) -> str:
    if col.endswith("Left"): return f"{video_tag} {col[:-4]} L"
    if col.endswith("Right"): return f"{video_tag} {col[:-5]} R"
    return f"{video_tag} {col}"


def _make_palette_for_region(cols: Sequence[str]) -> Dict[str, str]:
    return dict(PALETTE_HEX)


def create_region_timeseries_plot(std_ts: pd.DataFrame, usr_ts: pd.DataFrame, region: Dict[str, Any], out_path: str) -> Tuple[str, Dict[str, str]]:
    cols = region["columns"]
    palette = _make_palette_for_region(cols)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    max_val = 0.0

    for c in cols:
        side = _side_of_col(c)
        color = palette.get(f'Std-{side}', PALETTE_HEX['Std-ALL'])
        y = std_ts[c].values
        ax.plot(std_ts.index, y, label=_pretty_label(c, "Std"), linestyle='-', marker='o', markersize=2, color=color)
        if len(y) > 0: max_val = max(max_val, float(np.nanmax(y)))

    for c in cols:
        side = _side_of_col(c)
        color = palette.get(f'Usr-{side}', PALETTE_HEX['Usr-ALL'])
        y = usr_ts[c].values
        ax.plot(usr_ts.index, y, label=_pretty_label(c, "Usr"), linestyle='--', marker='x', markersize=2, color=color)
        if len(y) > 0: max_val = max(max_val, float(np.nanmax(y)))

    ax.set_title(f"{region['label']} — time series comparison")
    ax.set_xlabel("Frame index"); ax.set_ylabel("Strength (arb. unit)")
    ax.set_ylim(0.0, max(1.0, max_val * 1.05)); ax.grid(True, linestyle=':'); ax.legend(ncol=2)
    plt.tight_layout(); plt.savefig(out_path); plt.close(fig)
    return out_path, palette

# ---------------------------------------------------------------------
# 文本报告
# ---------------------------------------------------------------------

def _fmt_pct(x: Optional[float]) -> str:
    if x is None: return "N/A"
    try: return f"{float(x):.2%}"
    except Exception: return "N/A"


def _fmt_float(x: Optional[float]) -> str:
    if x is None: return "N/A"
    try: return f"{float(x):.2f}"
    except Exception: return "N/A"


def summarize_region(std_ts: pd.DataFrame, usr_ts: pd.DataFrame, region: Dict[str, Any]) -> Dict[str, Any]:
    cols = region["columns"]
    summary = {"label": region["label"], "columns": cols, "items": []}
    for c in cols:
        std_rng = float(_range(std_ts[c])) if c in std_ts.columns else 0.0
        usr_rng = float(_range(usr_ts[c])) if c in usr_ts.columns else 0.0
        std_peak = _peak_idx(std_ts[c]) if c in std_ts.columns else None
        usr_peak = _peak_idx(usr_ts[c]) if c in usr_ts.columns else None
        diff_pk = (abs((std_peak or 0) - (usr_peak or 0)) if (std_peak is not None and usr_peak is not None) else None)
        summary["items"].append({"col": c, "std_range": std_rng, "usr_range": usr_rng, "std_peak": std_peak, "usr_peak": usr_peak, "peak_diff": diff_pk})
    return summary


def generate_report_for_expression(expr: str, ssim_static: Optional[float], procrustes_similarity: Optional[float], dtw_similarity: Optional[float], landmark_dist: Optional[float], region_summaries: List[Dict[str, Any]]) -> str:
    cn_expr = CATEGORY_CN.get(expr, expr)
    lines: List[str] = [f"【{cn_expr} Facial Expression Imitation Evaluation Report】\n"]
    lines.append("--- Core indicator score ---")
    if ssim_static is not None: lines.append(f"  - Static appearance similarity (SSIM): {_fmt_pct(ssim_static)}")
    if procrustes_similarity is not None: lines.append(f"  - Facial structure alignment (Procrustes): {_fmt_pct(procrustes_similarity)}")
    if dtw_similarity is not None: lines.append(f"  - Blendshape (DTW-based): {_fmt_pct(dtw_similarity)}\n")
    elif landmark_dist is not None:
        approx = 1.0 / (1.0 + float(landmark_dist)) if np.isfinite(landmark_dist) else 0.0
        lines.append(f"  - Landmark average alignment (approx.): {_fmt_pct(approx)}\n")
    lines.append("--- Key geometric feature analysis (3 regions) ---")
    if not region_summaries:
        lines.append("  (No valid time series found for plotting.)")
    else:
        for rs in region_summaries:
            lines.append(f"  - {rs['label']} (columns: {', '.join(rs['columns'])}):")
            for it in rs["items"]:
                nm = it["col"]; std_rng = _fmt_float(it["std_range"]); usr_rng = _fmt_float(it["usr_range"])
                std_pk = it["std_peak"] if it["std_peak"] is not None else "N/A"
                usr_pk = it["usr_peak"] if it["usr_peak"] is not None else "N/A"
                diff_pk = it["peak_diff"] if it["peak_diff"] is not None else "N/A"
                lines.append(f"    · {nm}: range std={std_rng}, usr={usr_rng}; peak frame std={std_pk}, usr={usr_pk}, Δpeak={diff_pk}")
            lines.append("")
    # 综合评分（无 DTW 时自动降级）
    if dtw_similarity is not None and procrustes_similarity is not None:
        final_score = 0.5 * float(dtw_similarity) + 0.3 * float(procrustes_similarity) + 0.2 * float(ssim_static or 0.0)
    elif procrustes_similarity is not None:
        final_score = 0.6 * float(procrustes_similarity) + 0.4 * float(ssim_static or 0.0)
    else:
        final_score = float(ssim_static or 0.0)
    lines.append("--- Comprehensive assessment and recommendations ---")
    if final_score >= 0.85:
        summary = "Excellent! Your imitation is very accurate, both in final appearance and in dynamic evolution."
    elif final_score >= 0.70:
        summary = "Good! Fine‑tune timing and amplitude in the three key regions to improve further."
    else:
        summary = "There's room for improvement. Focus on matching peak timing and amplitude in the key regions."
    lines.append(f"  {summary}")
    return "\n".join(lines)

# ---------------------------------------------------------------------
# 入口：批量生成
# ---------------------------------------------------------------------

def generate_all_reports(results: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    out: List[Dict[str, Any]] = []

    for item in results:
        expr: str = item['expr']
        base = os.path.join(output_dir, expr)

        std_ts: Optional[pd.DataFrame] = item.get('std_ts')
        usr_ts: Optional[pd.DataFrame] = item.get('usr_ts')
        std_frames: List[str] = item.get('std_frames', [])
        usr_frames: List[str] = item.get('usr_frames', [])

        # 1) 中性脸对比图
        neutral_compare_img = create_neutral_comparison(std_frames, usr_frames, std_ts, usr_ts, base + "_neutral.jpg")

        # 2) 三部位选择 + 时序（返回 palette）+ 对应关键点（用相同 palette）
        top_regions = select_top_regions(std_ts, usr_ts, k=3)
        region_pairs: List[Dict[str, str]] = []
        region_summaries: List[Dict[str, Any]] = []
        for idx, r in enumerate(top_regions, start=1):
            ts_path, palette_hex = create_region_timeseries_plot(std_ts, usr_ts, r, f"{base}_ts_{idx}.jpg")
            kp_path = create_region_keypoints_image(
                item['std_img'], item['usr_img'], item.get('std_kp'), item.get('usr_kp'), r, f"{base}_kp_{idx}.jpg",
                palette_hex=palette_hex,
            )
            region_pairs.append({'region': r['label'], 'timeseries': ts_path, 'keypoints_img': kp_path})
            region_summaries.append(summarize_region(std_ts, usr_ts, r))

        # 3) 文本报告
        report_text = generate_report_for_expression(
            expr=expr,
            ssim_static=item.get('ssim_static'),
            procrustes_similarity=item.get('procrustes_similarity'),
            dtw_similarity=item.get('dtw_similarity'),
            landmark_dist=item.get('landmark_dist'),
            region_summaries=region_summaries,
        )
        report_txt_path = base + "_report.txt"
        with open(report_txt_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        out.append({'expr': expr, 'neutral_compare_img': neutral_compare_img, 'region_pairs': region_pairs, 'report_txt': report_txt_path})

    return out
