import os
from typing import List, Optional, Sequence
import numpy as np
import pandas as pd



def _safe_index(idx: int, n: int) -> Optional[int]:
    if n <= 0:
        return None
    if idx < 0 or idx >= n:
        return None
    return idx


def select_keyframe_middle(frames: List[str]) -> Optional[str]:
    """选取中位数帧作为关键帧。"""
    if not frames:
        return None
    return frames[len(frames) // 2]


def select_keyframe_au_peak(frames: List[str], au_values: np.ndarray) -> Optional[str]:

    if not frames:
        return None
    if au_values is None:
        return None
    au = np.asarray(au_values)
    if au.ndim != 2 or au.shape[0] != len(frames) or au.size == 0:
        return None
    sum_vals = au.sum(axis=1)
    peak_idx = int(np.argmax(sum_vals))
    peak_idx = _safe_index(peak_idx, len(frames))
    return frames[peak_idx] if peak_idx is not None else None


def select_keyframe_by_au_df(
    frames: List[str],
    au_df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
) -> Optional[str]:

    if not frames or au_df is None or not isinstance(au_df, pd.DataFrame) or au_df.empty:
        return None

    df = au_df.copy()
    # 仅保留数值列
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if not num_cols:
        return None

    if columns is not None:
        cols = [c for c in columns if c in num_cols]
        if not cols:
            cols = num_cols
    else:
        cols = num_cols

    # 保证长度匹配
    if len(df) != len(frames):
        # 尝试截断/对齐到最短长度
        n = min(len(df), len(frames))
        df = df.iloc[:n]
        frames = frames[:n]

    sum_vals = df[cols].sum(axis=1).to_numpy()
    if sum_vals.size == 0:
        return None
    peak_idx = int(np.argmax(sum_vals))
    peak_idx = _safe_index(peak_idx, len(frames))
    return frames[peak_idx] if peak_idx is not None else None


def select_keyframe(
    frames: List[str],
    au_values: Optional[np.ndarray] = None,
    method: str = 'middle'
) -> Optional[str]:

    if method == 'middle':
        return select_keyframe_middle(frames)
    elif method == 'au_peak':
        return select_keyframe_au_peak(frames, au_values)
    else:
        raise ValueError(f"未知的关键帧选择方法: {method}")
