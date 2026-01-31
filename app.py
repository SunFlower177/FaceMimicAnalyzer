import os
import shutil
from datetime import datetime
from typing import List, Dict
from pathlib import Path

import cv2
import streamlit as st

# ---- project modules (no CLIP) ----
from video_processing import extract_valid_frames
from expression_segmentation import (
    segment_expressions,
    match_segments,
    reselect_keyframes_with_au_peak,
)
from blendshape_extraction import initialize_landmarker, process_and_add_blendshape_features
from relative_change import compute_relative_features
from similarity_analysis import analyze_similarity, PREFERRED_BLENDSHAPE_BY_EXPR
from report_generator import generate_all_reports, CATEGORY_CN


# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="Facial expression analysis (Video & Image)", page_icon="😀", layout="wide")
st.title("😀 Facial Expression Imitation Analysis — Video & Image Support")
st.info(
    "Upload a standard media (video or image) and your imitation media, select the known expression, then click 'Start analysis'.\n"
    "PNG/JPG are treated as single‑frame videos; temporal metrics degrade gracefully."
)


# -----------------------------
# Landmarker (cached)
# -----------------------------
@st.cache_resource
def get_landmarker():
    return initialize_landmarker()


face_landmarker = get_landmarker()


# -----------------------------
# Sidebar: inputs
# -----------------------------
with st.sidebar:
    st.header("Upload media files")
    MEDIA_TYPES = ["mp4", "mov", "avi", "png", "jpg", "jpeg"]
    ref_file = st.file_uploader("1. Upload standard media (video/image)", type=MEDIA_TYPES)
    usr_file = st.file_uploader("2. Upload your imitation media (video/image)", type=MEDIA_TYPES)

    st.markdown("---")
    labels = ["neutral", "surprise", "happy", "sad", "angry", "disgust", "fear", "contempt"]
    selected_expr = st.selectbox("3. Known expression (applies to both)", options=labels, index=2)

    st.markdown("---")
    analyze = st.button("🚀 Start analysis", disabled=not (ref_file and usr_file), use_container_width=True, type="primary")


# -----------------------------
# Pipeline
# -----------------------------
def run_pipeline(ref_bytes: bytes, usr_bytes: bytes, expr: str, ref_name: str, usr_name: str):
    """Run the full pipeline on uploaded media.

    Images (png/jpg/jpeg) are treated as single‑frame videos by the frame extractor.
    """
    with st.status("🚀 Starting analysis...", expanded=True) as status:
        # Work dir
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f"temp_run_{run_id}"
        os.makedirs(base_dir, exist_ok=True)

        # Save uploads with original suffix (mp4/mov/avi/png/jpg/jpeg)
        ref_suffix = Path(ref_name).suffix.lower() or ".mp4"
        usr_suffix = Path(usr_name).suffix.lower() or ".mp4"
        ref_path = os.path.join(base_dir, f"ref_media{ref_suffix}")
        usr_path = os.path.join(base_dir, f"usr_media{usr_suffix}")
        with open(ref_path, "wb") as f:
            f.write(ref_bytes)
        with open(usr_path, "wb") as f:
            f.write(usr_bytes)

        # Step 1: frame extraction (image -> single frame)
        status.update(label="Step 1/7: Extracting valid frames …")
        ref_frames_dir = os.path.join(base_dir, "ref_frames")
        usr_frames_dir = os.path.join(base_dir, "usr_frames")
        os.makedirs(ref_frames_dir, exist_ok=True)
        os.makedirs(usr_frames_dir, exist_ok=True)
        # NOTE: current extract_valid_frames signature accepts (media_path, video_path, output_dir, ...)
        # For images, media_path==video_path and it returns a single saved frame.
        extract_valid_frames(ref_path, ref_path, ref_frames_dir)
        extract_valid_frames(usr_path, usr_path, usr_frames_dir)

        # Step 2: expression segmentation (our project treats as 1 segment; label is known)
        status.update(label="Step 2/7: Expression segmentation …")
        std_segs = segment_expressions(ref_frames_dir)
        usr_segs = segment_expressions(usr_frames_dir)
        for seg in std_segs:
            seg["expression"] = expr
        for seg in usr_segs:
            seg["expression"] = expr

        # Step 3: blendshapes timeseries + AU‑peak keyframe (official FaceLandmarker)
        status.update(label="Step 3/7: Extracting blendshapes (official) …")
        std_segs = process_and_add_blendshape_features(std_segs, face_landmarker)
        usr_segs = process_and_add_blendshape_features(usr_segs, face_landmarker)
        preferred_cols = PREFERRED_BLENDSHAPE_BY_EXPR.get(expr, None)
        std_segs = reselect_keyframes_with_au_peak(std_segs, preferred_cols=preferred_cols)
        usr_segs = reselect_keyframes_with_au_peak(usr_segs, preferred_cols=preferred_cols)

        # Step 4: relative feature change (baseline = min‑energy frame; for image it is the only frame)
        status.update(label="Step 4/7: Relative feature changes …")
        std_segs = compute_relative_features(std_segs, baseline_mode='min_energy')
        usr_segs = compute_relative_features(usr_segs, baseline_mode='min_energy')

        # Step 5: segment matching (1:1)
        status.update(label="Step 5/7: Matching segments …")
        matches = match_segments(std_segs, usr_segs)
        if not matches:
            status.update(label="No segments found", state="error")
            return [], base_dir

        # Step 6: similarity analysis (SSIM/Procrustes/DTW-if-available)
        status.update(label="Step 6/7: Similarity analysis …")
        analysis_results: List[Dict] = []
        for m in matches:
            std_seg, usr_seg = m['std_seg'], m['usr_seg']
            feats_std = std_seg.get('key_frame_features', {})
            feats_usr = usr_seg.get('key_frame_features', {})
            sim = analyze_similarity(
                cv2.imread(std_seg['key_frame']),
                cv2.imread(usr_seg['key_frame']),
                feats_std.get('landmarks'),
                feats_usr.get('landmarks'),
                std_seg.get('timeseries_blendshapes'),
                usr_seg.get('timeseries_blendshapes'),
                expr,
            )
            analysis_results.append({
                'expr': expr,
                'std_img': std_seg['key_frame'],
                'usr_img': usr_seg['key_frame'],
                'std_kp': feats_std.get('landmarks'),
                'usr_kp': feats_usr.get('landmarks'),
                'std_ts': std_seg.get('timeseries_blendshapes'),
                'usr_ts': usr_seg.get('timeseries_blendshapes'),
                'std_frames': std_seg.get('frames', []),
                'usr_frames': usr_seg.get('frames', []),
                'ssim_static': sim.get('ssim_static', 0.0),
                'dtw_similarity': sim.get('dtw_similarity', None),
                'landmark_dist': sim.get('landmark_distance', None),
                'procrustes_similarity': sim.get('procrustes_similarity', None),
            })

        # Step 7: report (neutral face + 3 regions time-series & color-linked keypoints + text)
        status.update(label="Step 7/7: Generating report …")
        report_dir = os.path.join(base_dir, 'reports')
        report_list = generate_all_reports(analysis_results, report_dir)
        status.update(label="🎉 Done", state="complete")
        return report_list, base_dir


# -----------------------------
# Trigger
# -----------------------------
if analyze:
    # clear old session state
    for k in ['reports', 'tmp_dir']:
        if k in st.session_state:
            del st.session_state[k]

    report_list, tmp_dir = run_pipeline(
        ref_file.getvalue(),
        usr_file.getvalue(),
        selected_expr,
        ref_file.name,
        usr_file.name,
    )

    if report_list:
        st.session_state['reports'] = report_list
        st.session_state['tmp_dir'] = tmp_dir
        st.rerun()


# -----------------------------
# Render results
# -----------------------------
if 'reports' in st.session_state and st.session_state['reports']:
    st.success("Analysis Complete! Below is a detailed evaluation report for each matching expression:")

    for info in st.session_state['reports']:
        expr = info['expr']
        cn_expr = CATEGORY_CN.get(expr, expr)
        with st.expander(f"### 🎭 {cn_expr} Expression analysis report", expanded=True):
            # top: neutral face comparison (if available)
            if info.get('neutral_compare_img'):
                st.image(info['neutral_compare_img'], caption="Neutral face — static comparison")

            # three regions: left(keypoints overlay) / right(timeseries)
            region_pairs = info.get('region_pairs', [])
            for i, pair in enumerate(region_pairs, start=1):
                l, r = st.columns([1, 1])
                with l:
                    st.image(pair['keypoints_img'], caption=f"Region {i}: {pair['region']} — landmark overlay (AU‑peak keyframe)")
                with r:
                    st.image(pair['timeseries'], caption=f"Region {i}: {pair['region']} — temporal comparison")

            # textual report
            try:
                with open(info['report_txt'], 'r', encoding='utf-8') as f:
                    txt = f.read()
                st.text_area("Detailed assessment and recommendations", txt, height=360, key=f"report_{expr}")
            except Exception as e:
                st.error(f"Error reading report text: {e}")

    st.markdown("---")
    if st.button("Clean up and start a new analysis", use_container_width=True):
        if 'tmp_dir' in st.session_state:
            shutil.rmtree(st.session_state['tmp_dir'], ignore_errors=True)
        for k in ['reports', 'tmp_dir']:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()