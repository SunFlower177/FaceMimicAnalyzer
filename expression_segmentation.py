import os
import cv2
from typing import List, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from feature_extraction import extract_features_from_image
from frame_selector import select_keyframe_middle, select_keyframe_by_au_df



# ======================================================================
# 1) 简化分段（无分类，单片段，关键帧先用中位数）
# ======================================================================

def segment_expressions(
    frames_dir: str,
    min_length: int = 5,
    sample_rate: int = 1,
) -> List[Dict]:
    frame_paths = sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".png"))
    ])

    if not frame_paths:
        print(f"警告：在目录 {frames_dir} 中未找到任何图像帧。")
        return []

    # 采样
    frame_paths = [p for i, p in enumerate(frame_paths) if i % sample_rate == 0]
    if len(frame_paths) < min_length:
        print(f"警告：帧数少于 min_length={min_length}，仍将作为一个片段处理。")

    # 关键帧（兜底：中位数帧）
    key_path = select_keyframe_middle(frame_paths)
    key_img = cv2.imread(key_path) if key_path else None
    if key_img is None and frame_paths:
        key_path = frame_paths[0]
        key_img = cv2.imread(key_path)

    seg = {
        "expression": "unknown",  # 由上层 UI 覆盖为已知类别
        "key_frame": key_path,
        "key_frame_features": extract_features_from_image(key_img) if key_img is not None else {},
        "frames": frame_paths,
        # Step3 之后会填充：'timeseries_blendshapes'
    }
    print(
        f"分段完成：生成 1 个片段（不含分类），初始关键帧：{os.path.basename(key_path) if key_path else 'None'}，共 {len(frame_paths)} 帧。"
    )
    return [seg]


# ======================================================================
# 2) Step3 之后：用 AU 峰值重新选择关键帧，并重算关键帧特征
# ======================================================================

def reselect_keyframes_with_au_peak(
    segments: List[Dict],
    preferred_cols: Optional[Sequence[str]] = None,
) -> List[Dict]:

    if not segments:
        return []

    for seg in segments:
        frames = seg.get("frames", [])
        ts = seg.get("timeseries_blendshapes", None)
        if isinstance(ts, pd.DataFrame) and not ts.empty and frames:
            # 选峰值帧
            new_key = select_keyframe_by_au_df(frames, ts, columns=preferred_cols)
            if new_key and os.path.exists(new_key):
                old_key = seg.get("key_frame")
                if old_key != new_key:
                    seg["key_frame"] = new_key
                    img = cv2.imread(new_key)
                    seg["key_frame_features"] = extract_features_from_image(img) if img is not None else {}
                    print(f"关键帧切换：{os.path.basename(old_key) if old_key else 'None'} -> {os.path.basename(new_key)}")
        else:
            print("提示：无有效的 timeseries_blendshapes 或 frames，保持中位数关键帧不变。")

    return segments


# ======================================================================
# 3) 分段匹配：单片段 1:1 配对
# ======================================================================

def match_segments(std_segments: List[Dict], usr_segments: List[Dict]) -> List[Dict]:
    if not std_segments or not usr_segments:
        return []
    return [{
        "expression": std_segments[0].get("expression", "unknown"),
        "std_seg": std_segments[0],
        "usr_seg": usr_segments[0],
    }]
