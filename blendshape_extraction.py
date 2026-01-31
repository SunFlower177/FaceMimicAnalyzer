import os
from typing import List, Optional, Callable, Dict

import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



# ==============================================================================
# 1) 模型路径（可直接改成你的绝对路径）
# ==============================================================================
MODEL_PATH_DEFAULT = os.environ.get(
    "FACE_LANDMARKER_TASK_PATH", "face_landmarker.task"
)


# ==============================================================================
# 2) 初始化官方 Face Landmarker（开启 blendshapes）
# ==============================================================================

def initialize_landmarker(model_path: Optional[str] = None,
                          num_faces: int = 1,
                          min_face_detection_confidence: float = 0.5,
                          min_face_presence_confidence: float = 0.5,
                          min_tracking_confidence: float = 0.5):
    """创建 MediaPipe FaceLandmarker（IMAGE 模式，输出 blendshapes）。"""
    model_path = model_path or MODEL_PATH_DEFAULT
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"未找到模型文件: {model_path}\n"
            "请将 FACE_LANDMARKER_TASK_PATH 指向带 blendshapes 的 .task（例如 face_landmarker_v2_with_blendshapes.task）。"
        )

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,  # 逐帧图像
        num_faces=num_faces,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    print(f"[FaceLandmarker] Using model: {model_path}")
    return landmarker


# ==============================================================================
# 3) 适配不同 mediapipe 版本的 blendshapes 解析
# ==============================================================================

def _parse_blendshapes_categories(result) -> Optional[Dict[str, float]]:

    if result is None:
        return None
    fb = getattr(result, "face_blendshapes", None)
    if not fb or len(fb) == 0:
        return None

    first = fb[0]
    categories = None
    # 新版：对象上有 categories 属性
    if hasattr(first, "categories"):
        categories = first.categories
    # 旧式：直接就是一个 list/tuple 可迭代
    elif isinstance(first, (list, tuple)):
        categories = first
    else:
        try:
            categories = list(first)  # 尝试转为迭代
        except Exception:
            categories = None

    if not categories:
        return None

    data: Dict[str, float] = {}
    for c in categories:
        name = getattr(c, "category_name", None) or getattr(c, "display_name", None)
        score = getattr(c, "score", None)
        if name is None or score is None:
            continue
        try:
            data[str(name)] = float(score)
        except Exception:
            pass

    return data or None


# ==============================================================================
# 4) 单帧提取
# ==============================================================================

def extract_blendshapes_from_image(image_path: str, landmarker) -> Optional[pd.Series]:
    """从单帧图像读取官方 blendshapes（返回 pd.Series）。失败返回 None。"""
    if landmarker is None or not os.path.exists(image_path):
        return None
    try:
        mp_image = mp.Image.create_from_file(image_path)
        detection_result = landmarker.detect(mp_image)
        data = _parse_blendshapes_categories(detection_result)
        if data:
            return pd.Series(data)
        return None
    except Exception:
        return None


# ==============================================================================
# 5) 多帧时间序列
# ==============================================================================

def extract_blendshapes_for_frames(frames: List[str],
                                   landmarker,
                                   progress_cb: Optional[Callable[[int, int], None]] = None,
                                   fill_strategy: str = "interp_then_zero") -> pd.DataFrame:
    """对帧序列逐帧提取官方 blendshapes -> DataFrame（行=时间，列=AU 名称）。"""
    records: List[dict] = []
    total = len(frames)
    non_empty = 0

    for i, p in enumerate(frames):
        s = extract_blendshapes_from_image(p, landmarker)
        if s is None:
            records.append({})
        else:
            d = s.to_dict()
            records.append(d)
            if d:
                non_empty += 1
        if progress_cb:
            progress_cb(i + 1, total)

    df = pd.DataFrame.from_records(records, index=range(total))

    # 仅保留数值列
    for col in list(df.columns):
        try:
            if not np.issubdtype(df[col].dtype, np.number):
                df.drop(columns=[col], inplace=True)
        except Exception:
            df.drop(columns=[col], inplace=True)

    if fill_strategy == "interp_then_zero":
        if df.shape[1] > 0:
            df = df.astype(float)
            df = df.interpolate(axis=0, limit_direction="both")
            df = df.fillna(0.0)
    elif fill_strategy == "zero":
        df = df.fillna(0.0)

    print(f"[Blendshapes] frames={total}, non_empty={non_empty}, cols={df.shape[1]}")
    return df


# ==============================================================================
# 6) 管道对接：把时间序列写回 segments
# ==============================================================================

def process_and_add_blendshape_features(segments: List[dict], landmarker) -> List[dict]:
    if not segments:
        return []
    for seg in segments:
        frames = seg.get("frames", [])
        if not frames:
            seg["timeseries_blendshapes"] = pd.DataFrame()
            continue
        ts = extract_blendshapes_for_frames(frames, landmarker)
        if ts.empty:
            print("❌ 官方 blendshapes 为空：请确认 .task 是否为 with_blendshapes 版本、以及人脸是否成功检测到。")
        seg["timeseries_blendshapes"] = ts
    return segments


# ==============================================================================
# 7) 命令行调试入口（可选）
# ==============================================================================
if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description="提取官方 blendshapes 时间序列并写入 segments")
    parser.add_argument("--ref", required=False, help="参考 segments 的 pickle 路径")
    parser.add_argument("--usr", required=False, help="用户 segments 的 pickle 路径")
    parser.add_argument("--out_ref", required=False, help="输出参考 segments 的 pickle 路径")
    parser.add_argument("--out_usr", required=False, help="输出用户 segments 的 pickle 路径")
    parser.add_argument("--model", default=MODEL_PATH_DEFAULT, help=".task 模型路径（with_blendshapes）")
    args = parser.parse_args()

    landmarker = initialize_landmarker(args.model)

    if args.ref and args.usr and args.out_ref and args.out_usr:
        with open(args.ref, 'rb') as f:
            seg_ref = pickle.load(f)
        with open(args.usr, 'rb') as f:
            seg_usr = pickle.load(f)
        seg_ref = process_and_add_blendshape_features(seg_ref, landmarker)
        seg_usr = process_and_add_blendshape_features(seg_usr, landmarker)
        with open(args.out_ref, 'wb') as f:
            pickle.dump(seg_ref, f)
        with open(args.out_usr, 'wb') as f:
            pickle.dump(seg_usr, f)
        print("✔ 完成：已将官方 blendshapes 时间序列写入 segments。")
    else:
        print("已成功初始化 landmarker。可在 app.py 中复用 initialize_landmarker()。")
