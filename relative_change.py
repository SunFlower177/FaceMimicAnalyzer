import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional



# ---------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------

def _ensure_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return df[cols]


def _au_mean_from_segment(seg: Dict) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:

    # 1) 直接复用已存在的 au_mean
    if 'au_mean' in seg and seg['au_mean'] is not None:
        au = seg['au_mean']
        if isinstance(au, np.ndarray):
            cols = seg.get('au_cols', None)
            # 尝试从 timeseries 恢复列名
            if cols is None and isinstance(seg.get('timeseries_blendshapes'), pd.DataFrame):
                cols = list(_ensure_numeric_columns(seg['timeseries_blendshapes']).columns)
            return au.astype(float), cols
        elif isinstance(au, dict):
            # 需要一个列顺序
            if isinstance(seg.get('timeseries_blendshapes'), pd.DataFrame):
                cols = list(_ensure_numeric_columns(seg['timeseries_blendshapes']).columns)
            else:
                cols = sorted(list(au.keys()))  # 退化：按键名排序
            vec = np.array([float(au.get(c, 0.0)) for c in cols], dtype=float)
            return vec, cols

    # 2) 从 timeseries_blendshapes 计算
    ts = seg.get('timeseries_blendshapes', None)
    if isinstance(ts, pd.DataFrame) and not ts.empty:
        ts_num = _ensure_numeric_columns(ts)
        if ts_num.shape[1] > 0:
            cols = list(ts_num.columns)
            vec = ts_num.mean(axis=0).to_numpy(dtype=float)
            return vec, cols

    return None, None


def _align_by_columns(vec_a: np.ndarray, cols_a: List[str],
                      vec_b: np.ndarray, cols_b: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """将 (vec_a, cols_a) 与 (vec_b, cols_b) 对齐到统一列顺序，返回 (a_aligned, b_aligned, cols_all)。"""
    set_all = list(dict.fromkeys([*cols_a, *cols_b]))  # 保持顺序去重
    idx_a = {c: i for i, c in enumerate(cols_a)}
    idx_b = {c: i for i, c in enumerate(cols_b)}
    a_new = np.array([vec_a[idx_a[c]] if c in idx_a else 0.0 for c in set_all], dtype=float)
    b_new = np.array([vec_b[idx_b[c]] if c in idx_b else 0.0 for c in set_all], dtype=float)
    return a_new, b_new, set_all


def _geo_from_segment(seg: Dict) -> Dict[str, float]:
    """返回几何特征字典（若无则补空字典）。"""
    geo = seg.get('expression_params_mean')
    if not isinstance(geo, dict):
        # 回退：从关键帧特征中取
        kf = seg.get('key_frame_features') or {}
        geo = kf.get('geometric_features', {}) if isinstance(kf, dict) else {}
    # 确保为 float
    return {str(k): float(v) for k, v in geo.items()} if isinstance(geo, dict) else {}


# ---------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------

def compute_relative_features(
    segments: List[Dict],
    baseline_mode: str = 'first'  # 'first' | 'min_energy'
) -> List[Dict]:
    """为每个分段计算相对 AU 与相对几何特征（相对某个基线分段）。"""
    if not segments:
        return []

    # 计算/补齐每个分段的 AU 均值与几何特征
    au_list: List[Tuple[Optional[np.ndarray], Optional[List[str]]]] = []
    geo_list: List[Dict[str, float]] = []
    for seg in segments:
        au_vec, au_cols = _au_mean_from_segment(seg)
        seg['au_mean'] = au_vec  # 统一保存为 ndarray（若存在）
        if au_cols is not None:
            seg['au_cols'] = list(au_cols)
        au_list.append((au_vec, au_cols))

        geo = _geo_from_segment(seg)
        seg['expression_params_mean'] = dict(geo)  # 统一保存
        geo_list.append(geo)

    # 选择基线索引
    baseline_idx = 0
    if baseline_mode == 'min_energy':
        # 选择 ||au_mean|| 最小者；若 AU 缺失，则忽略该段
        energies = [np.linalg.norm(v) if (v is not None and np.isfinite(v).all()) else np.inf
                    for (v, _) in au_list]
        if any(np.isfinite(energies)):
            baseline_idx = int(np.nanargmin(energies))

    baseline_seg = segments[baseline_idx]
    base_au, base_cols = au_list[baseline_idx]
    base_geo = geo_list[baseline_idx]

    # 逐段计算相对量
    for i, seg in enumerate(segments):
        # --- 相对 AU 向量 ---
        cur_au, cur_cols = au_list[i]
        rel_au = None
        if cur_au is not None and base_au is not None and cur_cols is not None and base_cols is not None:
            a, b, cols_all = _align_by_columns(cur_au, cur_cols, base_au, base_cols)
            rel_au = a - b
            # 同时更新标准化后的列名顺序，方便下游（可选）
            seg['au_cols'] = cols_all
        seg['rel_au_mean'] = rel_au

        # --- 相对几何特征（字典按键对齐） ---
        cur_geo = geo_list[i]
        keys = set(cur_geo.keys()) | set(base_geo.keys())
        rel_geo = {k: float(cur_geo.get(k, 0.0) - base_geo.get(k, 0.0)) for k in keys}
        seg['rel_expression_params_mean'] = rel_geo

    return segments


if __name__ == '__main__':
    print('This module is intended to be imported and used within the pipeline.')
