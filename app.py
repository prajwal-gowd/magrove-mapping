import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
import config
from models.hybrid_model import HybridTCCFNet
from utils.inference import predict

# --- Page Configuration ---
st.set_page_config(
    page_title="TCCFNet Segmentation Demo",
    page_icon="🖼️",
    layout="wide"
)

# --- CSS Styling ---
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1CB5E0 0%, #000851 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(28, 181, 224, 0.4);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Loads the model, preferably from checkpoint if available."""
    model = HybridTCCFNet(num_classes=config.NUM_CLASSES)
    
    if os.path.exists(config.SAVE_PATH):
        try:
            model.load_state_dict(torch.load(config.SAVE_PATH, map_location=config.DEVICE))
            st.success("Loaded trained model checkpoint!")
        except Exception as e:
            st.warning(f"Could not load checkpoint: {e}")
    else:
        st.info("No checkpoint found. Using untrained model.")
        
    model.to(config.DEVICE)
    model.eval()
    return model

# --- UI Header ---
st.title("🌿 Mangrove Image Segmentation")
st.markdown("""
Upload an image to perform **pixel-wise segmentation** using our Hybrid CNN-Transformer architecture.
""")

# --- Model Loading ---
with st.spinner("Initializing Model..."):
    model = load_model()

# --- Image Upload ---
uploaded_file = st.file_uploader("Upload an Image...", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    # --- Inference ---
    if st.button("Run Segmentation", use_container_width=True):
        with st.spinner("Processing..."):
            try:
                # Get binary mask
                mask_array = predict(model, image, config.DEVICE)
                
                with col2:
                    st.subheader("Predicted Mask")
                    st.image(mask_array, use_container_width=True, clamp=True, channels="GRAY")
                
                with col3:
                    st.subheader("Overlay")
                    # Create an overlay
                    mask_img = Image.fromarray(mask_array).convert("L")
                    # Resize original to match mask output size
                    image_resized = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
                    
                    # Create a colored mask (e.g. green for mangroves)
                    colored_mask = Image.new("RGB", image_resized.size, (0, 255, 0))
                    
                    # Composite original and colored mask using the predicted mask as alpha
                    overlay = Image.composite(colored_mask, image_resized, mask_img)
                    
                    # Blend them for a nice visualization
                    blended = Image.blend(image_resized, overlay, alpha=0.5)
                    st.image(blended, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during inference: {e}")

# --- Footer ---
st.markdown("---")
st.caption("Powered by PyTorch & Streamlit | TCCFNet Segmentation")
