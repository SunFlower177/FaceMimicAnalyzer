import os
from typing import Dict, List, Optional

import cv2
import numpy as np
import mediapipe as mp



# --------------------------------------------------------------------
# 1) FaceMesh 初始化（懒加载，复用实例）
# --------------------------------------------------------------------
mp_face_mesh = mp.solutions.face_mesh
_FACE_MESH_INSTANCE: Optional[mp.solutions.face_mesh.FaceMesh] = None

def _get_face_mesh() -> mp.solutions.face_mesh.FaceMesh:
    global _FACE_MESH_INSTANCE
    if _FACE_MESH_INSTANCE is None:
        _FACE_MESH_INSTANCE = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _FACE_MESH_INSTANCE

# --------------------------------------------------------------------
# 2) 关键点与几何特征
# --------------------------------------------------------------------

def extract_facial_landmarks_from_array(image: np.ndarray) -> Optional[np.ndarray]:
    """从 BGR 图像中提取 468 个归一化关键点，返回 ndarray[T=468, 3]。
    若未检测到人脸返回 None。
    """
    if image is None or image.size == 0:
        return None
    mesh = _get_face_mesh()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    pts = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark], dtype=np.float32)
    if pts.shape[0] != 468:
        # 理论为 468；若将来模型升级，这里做一次兜底
        return pts
    return pts


def _safe_norm(a: np.ndarray) -> float:
    v = float(np.linalg.norm(a))
    return v if v > 1e-8 else 1.0


def calculate_geometric_features(landmarks: Optional[np.ndarray]) -> Dict[str, float]:
    """根据 FaceMesh 关键点计算一组可解释的几何特征（已归一化到面部宽度）。
    返回 dict；若 landmarks=None，返回 {}。
    """
    if landmarks is None or getattr(landmarks, "shape", (0,))[0] < 400:
        return {}

    # 归一尺度：左右眼外角（33, 263）间距
    left_eye_outer = landmarks[33]
    right_eye_outer = landmarks[263]
    norm_w = _safe_norm(left_eye_outer - right_eye_outer)

    # 嘴部张开：上唇(13) - 下唇(14)
    mouth_open = _safe_norm(landmarks[13] - landmarks[14])
    mouth_opening_ratio = mouth_open / norm_w * 100.0

    # 嘴宽：嘴角(61)-(291)
    mouth_width = _safe_norm(landmarks[61] - landmarks[291])
    mouth_width_ratio = mouth_width / norm_w * 100.0

    # 眉毛抬起：以左眉(105)-左上眼睑(159)、右眉(334)-右上眼睑(386) 的平均垂直差作为近似
    # 这里只使用 y 分量（归一坐标），再用 norm_w 归一化到“面部宽度”尺度
    left_brow_raise = abs(float(landmarks[105][1] - landmarks[159][1]))
    right_brow_raise = abs(float(landmarks[334][1] - landmarks[386][1]))
    brow_raise_degree = ((left_brow_raise + right_brow_raise) / 2.0) / norm_w * 100.0

    # 眼睛睁开：上下眼睑距离（左：159-145；右：386-374）
    eye_open_left = abs(float(landmarks[159][1] - landmarks[145][1])) / norm_w * 100.0
    eye_open_right = abs(float(landmarks[386][1] - landmarks[374][1])) / norm_w * 100.0

    return {
        "mouth_opening_ratio": float(mouth_opening_ratio),
        "mouth_width_ratio": float(mouth_width_ratio),
        "brow_raise_degree": float(brow_raise_degree),
        "eye_open_left": float(eye_open_left),
        "eye_open_right": float(eye_open_right),
    }

# --------------------------------------------------------------------
# 3) 可视化
# --------------------------------------------------------------------

def draw_landmarks(image: np.ndarray, landmarks: Optional[np.ndarray]) -> np.ndarray:
    """将归一化关键点绘制到图像上（绿色小圆点）。"""
    if image is None:
        return image
    out = image.copy()
    if landmarks is None:
        return out
    h, w = out.shape[:2]
    for x, y, _ in landmarks:
        px = int(x * w)
        py = int(y * h)
        cv2.circle(out, (px, py), radius=2, color=(0, 255, 0), thickness=-1)
    return out

# --------------------------------------------------------------------
# 4) 单帧/批量接口
# --------------------------------------------------------------------

def extract_features_from_image(image: np.ndarray) -> Dict[str, object]:
    """对单帧提取：返回 {'landmarks': ndarray|None, 'geometric_features': dict}。"""
    lms = extract_facial_landmarks_from_array(image)
    feats = {
        "landmarks": lms,
        "geometric_features": calculate_geometric_features(lms),
    }
    return feats


def extract_features_for_segments(segments: List[Dict]) -> List[Dict]:

    for seg in segments:
        key_path = seg.get("key_frame") or seg.get("keyframe")
        if not key_path or not os.path.exists(key_path):
            seg["landmarks"] = None
            seg["geometric_features"] = {}
            seg["key_frame_features"] = {"landmarks": None, "geometric_features": {}}
            continue
        img = cv2.imread(key_path)
        f = extract_features_from_image(img)
        seg["landmarks"] = f.get("landmarks")
        seg["geometric_features"] = f.get("geometric_features", {})
        # 统一放回 key_frame_features，便于后续 Step6 使用
        seg["key_frame_features"] = f
    return segments
