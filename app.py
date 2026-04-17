
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io

# =========================
# TEXT CONVERSION
# =========================

def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def bits_to_text(bits):
    chars = []

    for i in range(0, len(bits), 8):

        byte = bits[i:i+8]

        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))

    return ''.join(chars)

# =========================
# VEGETATION INDEX (ExG)
# =========================

def compute_exg(img):

    R = img[:,:,0].astype(float)
    G = img[:,:,1].astype(float)
    B = img[:,:,2].astype(float)

    return 2*G - R - B

# =========================
# TRUE VEGETATION VIOLATION
# =========================

def vegetation_violation(exg1, exg2, tol=5):

    diff = np.abs(exg1 - exg2)

    violation_pixels = diff > tol

    violation_rate = (
        np.sum(violation_pixels)
        / diff.size
    ) * 100

    return violation_rate, diff

# =========================
# GREEN MASK
# =========================

def green_mask(img):

    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    return (G > R) & (G > B)

# =========================
# GREENSTEG EMBEDDING
# =========================

def embed_text(img, message):

    bits = text_to_bits(message)

    length_bits = format(len(bits), '032b')

    full_bits = length_bits + bits

    mask = green_mask(img)

    flat_img = img.flatten()
    flat_mask = mask.flatten()

    bit_idx = 0

    for i in range(len(flat_img)):

        if flat_mask[i] and bit_idx < len(full_bits):

            flat_img[i] = (
                flat_img[i] & 254
            ) | int(full_bits[bit_idx])

            bit_idx += 1

    return flat_img.reshape(img.shape)

# =========================
# EXTRACTION
# =========================

def extract_text(img):

    flat = img.flatten()

    length_bits = ''.join(
        str(flat[i] & 1)
        for i in range(32)
    )

    message_length = int(length_bits, 2)

    message_bits = ''.join(
        str(flat[i] & 1)
        for i in range(
            32,
            32 + message_length
        )
    )

    return bits_to_text(message_bits)

# =========================
# UI
# =========================

st.title("🌱 GreenSteg Vegetation-Aware Steganography")

st.markdown(
"""
Upload an agricultural image,
embed secret data,
and verify vegetation preservation.
"""
)

# =========================
# EMBEDDING SECTION
# =========================

st.header("🔐 Embed Message")

uploaded_image = st.file_uploader(
    "Upload Cover Image",
    type=["png","jpg","jpeg"]
)

message = st.text_input(
    "Enter Secret Message"
)

if uploaded_image and message:

    img = Image.open(uploaded_image)

    img = np.array(img)

    # Compute vegetation BEFORE

    exg_original = compute_exg(img)

    # Embed message

    stego_img = embed_text(
        img.copy(),
        message
    )

    # Compute vegetation AFTER

    exg_stego = compute_exg(stego_img)

    # Compute violation

    violation_rate, diff_map = vegetation_violation(
        exg_original,
        exg_stego
    )

    st.success("Message Embedded Successfully!")

    st.write(
        f"Vegetation Violation Rate: {violation_rate:.4f}%"
    )

    # =========================
    # VISUALIZATION
    # =========================

    fig, ax = plt.subplots(
        1,
        4,
        figsize=(16,4)
    )

    ax[0].imshow(img)
    ax[0].set_title("Original Image")

    ax[1].imshow(stego_img)
    ax[1].set_title("Stego Image")

    ax[2].imshow(
        exg_original,
        cmap="Greens"
    )
    ax[2].set_title(
        "Vegetation Map"
    )

    ax[3].imshow(
        diff_map,
        cmap="hot"
    )
    ax[3].set_title(
        "Vegetation Change Heatmap"
    )

    for a in ax:
        a.axis("off")

    st.pyplot(fig)

    # =========================
    # DOWNLOAD
    # =========================

    stego_pil = Image.fromarray(stego_img)

    buffer = io.BytesIO()

    stego_pil.save(
        buffer,
        format="PNG"
    )

    st.download_button(
        label="Download Stego Image",
        data=buffer.getvalue(),
        file_name="GreenSteg_output.png",
        mime="image/png"
    )

# =========================
# EXTRACTION SECTION
# =========================

st.markdown("---")

st.header("🔓 Extract Message")

uploaded_stego = st.file_uploader(
    "Upload Stego Image",
    type=["png","jpg","jpeg"],
    key="extract"
)

if uploaded_stego:

    stego = Image.open(uploaded_stego)

    stego = np.array(stego)

    recovered_message = extract_text(stego)

    st.success("Recovered Message:")

    st.code(recovered_message)
