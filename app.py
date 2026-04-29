from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.inference import (
    detect_and_crop_face,
    detect_largest_face_box,
    generate_gradcam,
    load_detector,
    predict_image,
)

try:
    import av
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
except ImportError:
    av = None
    WebRtcMode = None
    webrtc_streamer = None


CHECKPOINT_PATH = Path("models/best_model.pth")


st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("Deepfake Detection")
st.write("Upload an image, take a photo, or use live webcam detection with Grad-CAM.")


@st.cache_resource(show_spinner=False)
def load_trained_model():
    return load_detector(str(CHECKPOINT_PATH))


inference_lock = threading.Lock()


def render_prediction(image: Image.Image) -> None:
    if not CHECKPOINT_PATH.exists():
        st.error("Model checkpoint not found. Train the model first with: python src/train.py")
        return

    model, device = load_trained_model()

    with st.spinner("Running prediction and Grad-CAM..."):
        processed_image, face_detected = detect_and_crop_face(image)
        prediction = predict_image(processed_image, model, device, assume_cropped=True)
        target_label = 1 if prediction["prediction"] == "Fake" else 0
        explanation = generate_gradcam(
            processed_image,
            model,
            device,
            target_label=target_label,
            assume_cropped=True,
        )

    label = prediction["prediction"]
    confidence = prediction["confidence"] * 100.0
    fake_probability = prediction["fake_probability"] * 100.0

    if label == "Fake":
        st.error(f"Prediction: {label}")
    else:
        st.success(f"Prediction: {label}")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Confidence", f"{confidence:.2f}%")
    metric_col2.metric("Fake Probability", f"{fake_probability:.2f}%")
    metric_col3.metric("Threshold", f"{prediction['threshold']:.2f}")

    if face_detected:
        st.info("Face detected. Prediction and Grad-CAM are generated from the cropped face region.")
    else:
        st.warning("No face detected. Using the full image.")

    image_col1, image_col2, image_col3 = st.columns(3)
    with image_col1:
        st.image(image, caption="Input image", use_container_width=True)
    with image_col2:
        st.image(processed_image, caption="Processed face", use_container_width=True)
    with image_col3:
        st.image(explanation["overlay"], caption="Grad-CAM overlay", use_container_width=True)

    with st.expander("View raw Grad-CAM heatmap"):
        st.image(explanation["heatmap"], caption="Grad-CAM heatmap", use_container_width=True)

    st.caption("Red/yellow regions show the parts of the image most responsible for the model's decision.")


def build_live_frame(frame_bgr: np.ndarray) -> np.ndarray:
    model, device = load_trained_model()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    face_box, detected = detect_largest_face_box(image)

    annotated = frame_bgr.copy()
    if not detected or face_box is None:
        cv2.putText(
            annotated,
            "No face detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    left, top, right, bottom = face_box
    face_image = image.crop((left, top, right, bottom))

    with inference_lock:
        prediction = predict_image(face_image, model, device, assume_cropped=True)
        target_label = 1 if prediction["prediction"] == "Fake" else 0
        explanation = generate_gradcam(
            face_image,
            model,
            device,
            target_label=target_label,
            assume_cropped=True,
        )

    overlay_bgr = cv2.cvtColor(explanation["overlay"], cv2.COLOR_RGB2BGR)
    box_width = max(right - left, 1)
    box_height = max(bottom - top, 1)
    overlay_bgr = cv2.resize(overlay_bgr, (box_width, box_height), interpolation=cv2.INTER_LINEAR)
    annotated[top:bottom, left:right] = overlay_bgr

    label = prediction["prediction"]
    confidence = prediction["confidence"] * 100.0
    color = (0, 0, 255) if label == "Fake" else (0, 200, 0)

    cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)

    text = f"{label} {confidence:.1f}%"
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        2,
    )
    text_top = max(top - text_height - baseline - 8, 0)
    text_bottom = min(text_top + text_height + baseline + 8, annotated.shape[0] - 1)
    text_right = min(left + text_width + 12, annotated.shape[1] - 1)
    cv2.rectangle(annotated, (left, text_top), (text_right, text_bottom), color, -1)
    cv2.putText(
        annotated,
        text,
        (left + 6, text_bottom - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


tab_upload, tab_camera, tab_live = st.tabs(["Upload Image", "Webcam", "Live Detection"])

with tab_upload:
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        render_prediction(image)

with tab_camera:
    camera_file = st.camera_input("Take a photo")
    if camera_file is not None:
        image = Image.open(camera_file).convert("RGB")
        render_prediction(image)

with tab_live:
    st.write("Live detection draws a face box, writes Real/Fake on the frame, and refreshes Grad-CAM continuously.")

    if not CHECKPOINT_PATH.exists():
        st.error("Model checkpoint not found. Train the model first with: python src/train.py")
    elif webrtc_streamer is None or av is None:
        st.warning("Install live webcam support with: pip install streamlit-webrtc")
    else:
        rtc_config = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

        def video_frame_callback(frame):
            frame_bgr = frame.to_ndarray(format="bgr24")
            annotated = build_live_frame(frame_bgr)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        webrtc_streamer(
            key="deepfake-live-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )
