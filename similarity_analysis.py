import cv2
import numpy as np
from typing import Optional, Dict
from skimage.metrics import structural_similarity as ssim



# 推荐的核心 blendshape 列（用于 DTW 优先选择）
PREFERRED_BLENDSHAPE_BY_EXPR = {
    "surprise": ["jawOpen", "eyeWideLeft", "eyeWideRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight"],
    "happy":    ["mouthSmileLeft", "mouthSmileRight", "cheekPuff"],
    "sad":      ["browInnerUp", "mouthFrownLeft", "mouthFrownRight"],
    "angry":    ["browDownLeft", "browDownRight", "mouthPressLeft", "mouthPressRight"],
    "disgust":  ["noseSneerLeft", "noseSneerRight", "mouthPressLeft", "mouthPressRight"],
    "fear":     ["eyeWideLeft", "eyeWideRight", "jawOpen", "browInnerUp"],
    "contempt": ["mouthDimpleLeft", "mouthDimpleRight", "mouthSmileLeft", "mouthSmileRight"],
    "neutral":  ["mouthClose", "jawOpen"],
}

# ---------------------------------------------------------------------
# 1) 静态：结构相似性 SSIM（图像）
# ---------------------------------------------------------------------

def compute_ssim_similarity(image1: np.ndarray, image2: np.ndarray) -> float:
    """对两张 BGR 图像求 SSIM，相似度范围裁剪到 [0,1]。"""
    if image1 is None or image2 is None:
        return 0.0
    h = min(image1.shape[0], image2.shape[0])
    w = min(image1.shape[1], image2.shape[1])
    if h < 8 or w < 8:
        return 0.0
    img1 = cv2.cvtColor(cv2.resize(image1, (w, h)), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.resize(image2, (w, h)), cv2.COLOR_BGR2GRAY)
    try:
        rng = img1.max() - img1.min()
        rng = int(rng) if rng > 0 else 1
        val = ssim(img1, img2, data_range=rng)
    except Exception:
        return 0.0
    return float(max(0.0, min(1.0, val)))

# ---------------------------------------------------------------------
# 2) 结构：关键点距离 & Procrustes 对齐
# ---------------------------------------------------------------------

def compute_landmark_distance(landmarks1: np.ndarray, landmarks2: np.ndarray) -> Optional[float]:
    """关键点均值欧氏距离（仅 xy 维）。形状不一致返回 None。"""
    if landmarks1 is None or landmarks2 is None:
        return None
    if landmarks1.shape != landmarks2.shape:
        return None
    diff = landmarks1[:, :2] - landmarks2[:, :2]
    dists = np.linalg.norm(diff, axis=1)
    return float(dists.mean())


def compute_procrustes_rmse(landmarks1: np.ndarray, landmarks2: np.ndarray) -> Optional[float]:
    """Procrustes 对齐后的 RMSE（仅 xy 维）。失败返回 None。"""
    if landmarks1 is None or landmarks2 is None:
        return None
    if landmarks1.shape[0] != landmarks2.shape[0]:
        return None
    X = landmarks1[:, :2].astype(np.float64)
    Y = landmarks2[:, :2].astype(np.float64)

    # 去中心化
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    # 归一化尺度
    X_norm = np.linalg.norm(Xc)
    Y_norm = np.linalg.norm(Yc)
    if X_norm < 1e-8 or Y_norm < 1e-8:
        return None
    Xc /= X_norm
    Yc /= Y_norm

    # SVD 求旋转
    U, _, Vt = np.linalg.svd(Xc.T @ Yc)
    R = U @ Vt
    X_aligned = Xc @ R

    rmse = np.sqrt(np.mean(np.sum((X_aligned - Yc) ** 2, axis=1)))
    return float(rmse)


def procrustes_similarity_from_rmse(rmse: Optional[float]) -> Optional[float]:
    """将 RMSE 转换为相似度 [0,1]：1/(1+rmse)。"""
    if rmse is None:
        return None
    return float(1.0 / (1.0 + rmse))

# ---------------------------------------------------------------------
# 3) 动态：DTW（1D 序列）
# ---------------------------------------------------------------------

def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """简单 DTW：|a_i - b_j| 代价，返回累计最小代价。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return float('inf')
    a = np.where(np.isnan(a), 0.0, a)
    b = np.where(np.isnan(b), 0.0, b)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = abs(ai - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def dtw_similarity_from_distance(dist: float) -> float:
    """将 DTW 距离转为相似度 [0,1]：1/(1+dist)。"""
    if not np.isfinite(dist):
        return 0.0
    return float(1.0 / (1.0 + dist))

# ---------------------------------------------------------------------
# 4) 综合分析入口
# ---------------------------------------------------------------------

def analyze_similarity(
    std_img: np.ndarray,
    usr_img: np.ndarray,
    std_landmarks: np.ndarray = None,
    usr_landmarks: np.ndarray = None,
    std_ts = None,   # pandas.DataFrame or None
    usr_ts = None,   # pandas.DataFrame or None
    expr: str = None
) -> Dict:
    """综合计算静态、结构、动态三类相似度指标。"""
    out = {}

    # 静态外观：SSIM
    out['ssim_static'] = compute_ssim_similarity(std_img, usr_img)

    # 结构度量：关键点平均距离 + Procrustes 相似度
    out['landmark_distance'] = compute_landmark_distance(std_landmarks, usr_landmarks)
    rmse = compute_procrustes_rmse(std_landmarks, usr_landmarks)
    out['procrustes_rmse'] = rmse
    out['procrustes_similarity'] = procrustes_similarity_from_rmse(rmse)

    # 动态：DTW（优先用 expr 对应的推荐列）
    out['dtw_distance'] = None
    out['dtw_similarity'] = None
    if std_ts is not None and usr_ts is not None:
        try:
            common_cols = [c for c in std_ts.columns if c in getattr(usr_ts, 'columns', [])]
            best_col = None
            # 1) 若指定了 expr，优先取推荐列里双方都存在的第一个
            if expr and expr in PREFERRED_BLENDSHAPE_BY_EXPR:
                for c in PREFERRED_BLENDSHAPE_BY_EXPR[expr]:
                    if c in common_cols:
                        best_col = c
                        break
            # 2) 否则改用变化范围最大的共同列
            if best_col is None and common_cols:
                ranges = [(col, (std_ts[col].max() - std_ts[col].min())) for col in common_cols]
                ranges.sort(key=lambda x: float(x[1]) if np.isfinite(x[1]) else -1, reverse=True)
                best_col = ranges[0][0]
            if best_col is not None:
                dist = dtw_distance(std_ts[best_col].values, usr_ts[best_col].values)
                out['dtw_distance'] = dist
                out['dtw_similarity'] = dtw_similarity_from_distance(dist)
        except Exception:
            # 若时序不可用或其他异常，动态相似度保持为 None
            pass

    return out
