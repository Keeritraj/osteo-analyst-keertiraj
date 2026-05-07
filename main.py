import streamlit as st
import sys
import os

# -----------------------------------------------------------------------------
# CRITICAL PATH FIX
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from PIL import Image
import numpy as np
import matplotlib.cm as cm
from pathlib import Path

# Backend Imports
from backend.data_pipeline import preprocess_image
from backend.model import load_model, predict, generate_grad_cam
from backend.report_generator import create_final_report
from utils.dicom_handler import dicom_to_pil
from app.ui_components import render_sidebar, render_upload
from config.settings import TEMP_DIR, KL_GRADES

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Osteo-Analyst | Neural Diagnostic Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------------------------------------------
# NEXT-GEN CSS ANIMATION & STYLING
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* 1. GLOBAL FONTS & RESET */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #1e293b;
    }

    /* 2. LIVE FLOATING BACKGROUND PARTICLES */
    .area{
        background: #f8fafc;  
        width: 100%;
        height:100vh;
        position: fixed;
        top: 0;
        left: 0;
        z-index: -1;
    }
    .circles{
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        overflow: hidden;
    }
    .circles li{
        position: absolute;
        display: block;
        list-style: none;
        width: 20px;
        height: 20px;
        background: rgba(71, 118, 230, 0.15);
        animation: animate 25s linear infinite;
        bottom: -150px;
        border-radius: 50%;
    }
    .circles li:nth-child(1){ left: 25%; width: 80px; height: 80px; animation-delay: 0s; }
    .circles li:nth-child(2){ left: 10%; width: 20px; height: 20px; animation-delay: 2s; animation-duration: 12s; }
    .circles li:nth-child(3){ left: 70%; width: 20px; height: 20px; animation-delay: 4s; }
    .circles li:nth-child(4){ left: 40%; width: 60px; height: 60px; animation-delay: 0s; animation-duration: 18s; }
    .circles li:nth-child(5){ left: 65%; width: 20px; height: 20px; animation-delay: 0s; }
    .circles li:nth-child(6){ left: 75%; width: 110px; height: 110px; animation-delay: 3s; }
    .circles li:nth-child(7){ left: 35%; width: 150px; height: 150px; animation-delay: 7s; }
    .circles li:nth-child(8){ left: 50%; width: 25px; height: 25px; animation-delay: 15s; animation-duration: 45s; }
    .circles li:nth-child(9){ left: 20%; width: 15px; height: 15px; animation-delay: 2s; animation-duration: 35s; }
    .circles li:nth-child(10){ left: 85%; width: 150px; height: 150px; animation-delay: 0s; animation-duration: 11s; }

    @keyframes animate {
        0%{ transform: translateY(0) rotate(0deg); opacity: 1; border-radius: 0; }
        100%{ transform: translateY(-1000px) rotate(720deg); opacity: 0; border-radius: 50%; }
    }

    /* 3. GLASSMORPHISM HEADER */
    .header-box {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
    }
    .title-gradient {
        background: linear-gradient(to right, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
    }

    /* 4. MODERN CARDS */
    .glass-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        border: 1px solid #f1f5f9;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        border-color: #3b82f6;
    }

    /* 5. DIAGNOSIS PULSE ANIMATION */
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(46, 204, 113, 0); }
        100% { box-shadow: 0 0 0 0 rgba(46, 204, 113, 0); }
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(231, 76, 60, 0); }
        100% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0); }
    }
    .status-badge {
        padding: 8px 16px;
        border-radius: 20px;
        color: white;
        font-weight: 600;
        text-align: center;
        display: inline-block;
        width: 100%;
        margin-bottom: 10px;
    }

    /* 6. SUBTLE FOOTER (The Secret Credits) */
    .stealth-footer {
        position: fixed;
        bottom: 10px;
        right: 10px;
        font-size: 0.7rem;
        color: #94a3b8;
        opacity: 0.3;
        transition: opacity 0.3s ease;
        z-index: 9999;
        cursor: default;
    }
    .stealth-footer:hover {
        opacity: 1;
        color: #475569;
    }
    
    /* 7. Button Gradients */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border: none;
        color: white;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: scale(1.02);
    }
</style>

<!-- BACKGROUND ANIMATION HTML -->
<div class="area">
    <ul class="circles">
        <li></li><li></li><li></li><li></li><li></li>
        <li></li><li></li><li></li><li></li><li></li>
    </ul>
</div
""", unsafe_allow_html=True)

@st.cache_resource
def get_cached_model():
    return load_model()

def apply_heatmap_overlay(original_image, heatmap, alpha=0.4):
    original_rgb = original_image.convert("RGB")
    width, height = original_image.size
    
    heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    heatmap_colored = cm.jet(heatmap_norm)[:, :, :3]
    heatmap_img = Image.fromarray((heatmap_colored * 255).astype('uint8'))
    heatmap_img = heatmap_img.resize((width, height), resample=Image.BILINEAR)

    overlay = Image.blend(original_rgb, heatmap_img, alpha=alpha)
    return overlay, heatmap_img

def main():
    # 1. HEADER
    st.markdown("""
        <div class='header-box'>
            <h1 class='title-gradient'>OSTEO-ANALYST PRO</h1>
            <p style='font-size: 1.2rem; color: #64748b;'>Neural-Augmented Orthopedic Diagnostics</p>
        </div>
    """, unsafe_allow_html=True)

    # 2. INPUT SECTION
    patient_data = render_sidebar()
    
    col_input, col_dummy = st.columns([1, 0.1])
    with col_input:
        uploaded_file = render_upload()

    if not uploaded_file:
        st.markdown("""
            <div style='text-align: center; padding: 50px; opacity: 0.6;'>
                <h3>Waiting for Input Stream...</h3>
                <p>Secure DICOM/Image pipeline standing by.</p>
            </div>
        """, unsafe_allow_html=True)
        # Inject Stealth Footer here too so it's visible on idle screen
        st.markdown('<div class="stealth-footer">Engineered by Prawin Kumar & Arpita Singh</div>', unsafe_allow_html=True)
        return

    if not patient_data["name"].strip():
        st.warning("⚠️ CLINICAL PROTOCOL: Patient identification required in sidebar.")
        return

    # 3. PROCESSING
    try:
        file_ext = uploaded_file.name.lower().split('.')[-1]
        if file_ext == "dcm":
            image = dicom_to_pil(uploaded_file)
        else:
            image = Image.open(uploaded_file).convert("L")
    except Exception as e:
        st.error(f"Pipeline Error: {str(e)}")
        return

    with st.status("🧠 Initializing Neural Core...", expanded=True) as status:
        st.write("🔹 Converting Biological Data to Tensor Format...")
        try:
            input_tensor = preprocess_image(image)
        except Exception as e:
            status.update(label="Preprocessing Failed", state="error")
            st.error(str(e))
            return

        st.write("🔹 Running DenseNet121 Inference Engine...")
        model = get_cached_model()
        try:
            pred_class, confidence = predict(model, input_tensor)
        except Exception as e:
            status.update(label="Inference Failed", state="error")
            st.error(str(e))
            return

        st.write("🔹 Calculating Gradient-Weighted Class Activation Maps...")
        try:
            heatmap_raw = generate_grad_cam(model, input_tensor, pred_class)
        except Exception as e:
            status.update(label="Grad-CAM Failed", state="error")
            st.error(str(e))
            return
            
        status.update(label="✅ Analysis Complete", state="complete", expanded=False)

    # 4. RESULTS DASHBOARD
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Logic for Status Badge Animation
    grade_label = KL_GRADES[pred_class]
    if pred_class <= 1:
        anim_class = "pulse-green"
        bg_color = "#2ecc71"
    elif pred_class <= 3:
        anim_class = "pulse-orange" 
        bg_color = "#f1c40f"
    else:
        anim_class = "pulse-red"
        bg_color = "#e74c3c"

    # Metrics Row
    c1, c2, c3 = st.columns([1, 1, 1.5])
    with c1:
        st.markdown(f"""
            <div class='glass-card' style='text-align:center;'>
                <div style='font-size: 0.9rem; color:#64748b; margin-bottom:5px;'>SEVERITY GRADE</div>
                <div style='font-size: 2.5rem; font-weight:800; color:#1e293b;'>{pred_class}<span style='font-size:1rem; color:#94a3b8;'>/4</span></div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
            <div class='glass-card' style='text-align:center;'>
                <div style='font-size: 0.9rem; color:#64748b; margin-bottom:5px;'>AI CERTAINTY</div>
                <div style='font-size: 2.5rem; font-weight:800; color:#3b82f6;'>{confidence:.1%}</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        # Animated Badge
        st.markdown(f"""
            <div class='glass-card' style='display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%;'>
                <div style='font-size: 0.9rem; color:#64748b; margin-bottom:10px;'>DIAGNOSTIC STATUS</div>
                <div class='status-badge' style='background-color: {bg_color}; animation: {anim_class} 2s infinite;'>
                    {grade_label.upper()}
                </div>
            </div>
        """, unsafe_allow_html=True)

    # 5. VISUALIZATION DECK
    st.markdown("### 🔬 Radiological Evidence")
    
    col_vis1, col_vis2 = st.columns(2)
    
    with col_vis1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.caption("RAW INPUT X-RAY")
        st.image(image, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_vis2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.caption("AI ATTENTION MAP (PATHOLOGY LOCALIZATION)")
        alpha = st.slider("Heatmap Intensity", 0.0, 1.0, 0.4, key="alpha_slider")
        overlay_img, heatmap_rgb = apply_heatmap_overlay(image, heatmap_raw, alpha=alpha)
        st.image(overlay_img, use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 6. REPORT GENERATION
    st.markdown("### 📄 Clinical Actions")
    col_act1, col_act2 = st.columns([1, 2])
    
    with col_act1:
        if st.button("Generate Professional Report", type="primary"):
            with st.spinner("consulting LLM (Llama 3.2) for radiologist interpretation..."):
                os.makedirs(TEMP_DIR, exist_ok=True)
                cam_path = TEMP_DIR / "gradcam.png"
                heatmap_rgb.save(cam_path) 

                pdf_path = TEMP_DIR / f"Report_{patient_data['name'].replace(' ', '_')}.pdf"
                
                try:
                    create_final_report(
                        patient_data=patient_data,
                        kl_grade=pred_class,
                        confidence=confidence,
                        cam_image_path=str(cam_path),
                        output_pdf_path=str(pdf_path)
                    )
                    st.session_state['report_ready'] = True
                    st.session_state['pdf_path'] = str(pdf_path)
                    st.toast("Report Generated Successfully!", icon="✅")
                except Exception as e:
                    st.error(f"Failed to generate PDF: {str(e)}")

    with col_act2:
        if st.session_state.get('report_ready'):
            with open(st.session_state['pdf_path'], "rb") as f:
                st.download_button(
                    label="⬇️ Download PDF Record",
                    data=f,
                    file_name=os.path.basename(st.session_state['pdf_path']),
                    mime="application/pdf"
                )

    # 7. STEALTH CREDITS (The footer)
    st.markdown('<div class="stealth-footer">Developed by Keertiraj Singh</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()