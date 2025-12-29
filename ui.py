import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Car Scratch Severity Detector", layout="centered")

st.title("Car Scratch Severity Detection")
st.write("Upload a car image to detect scratches and classify severity.")

def generate_scratch_mask(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 120, 240)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    mask = np.zeros_like(gray)

    for cnt in contours:
        length = cv2.arcLength(cnt, closed=False)
        area = cv2.contourArea(cnt)

        # Heuristics for scratches
        if 30 < length < 300 and area < 200:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=2)

    return mask


def compute_severity(mask):
    total_pixels = mask.shape[0] * mask.shape[1]
    scratch_pixels = np.sum(mask > 0)

    ratio = scratch_pixels / total_pixels

    if ratio <= 0.01:
        severity = "Low"
    elif ratio <= 0.03:
        severity = "Medium"
    else:
        severity = "High"

    return ratio, severity

uploaded_file = st.file_uploader(
    "Upload car image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    mask = generate_scratch_mask(img_bgr)
    ratio, severity = compute_severity(mask)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Scratch Mask")
        st.image(mask, clamp=True, use_column_width=True)

    st.markdown("---")
    st.subheader("Prediction")

    st.write(f"**Scratch Area Ratio:** `{ratio:.4f}`")
    st.write(f"**Predicted Severity:** `{severity}`")
