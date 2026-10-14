import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import timm

--extra-index-url https://download.pytorch.org/whl/cpu
streamlit
torch
torchvision
timm
opencv-python-headless
Pillow
numpy

# -----------------------------------------------------------------------------
# 1. Page Configuration & UI Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Ocular Pathologies Diagnostic System",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ Anterior Segment Ocular Diagnostic System")
st.write(
    "Upload a clear photograph of an eye to run automated diagnostic screening "
    "across 6 anterior segment conditions."
)

CLASS_NAMES = [
    "Allergic Conjunctivitis (Base_AC)",
    "Cataract (Base_Cataract)",
    "Normal Eye (Base_Normal)",
    "Uveitis (Base_Uveitis)",
    "Drooping Eyelid / Ptosis (Base_droppy_Eyelid)",
    "Vernal Keratoconjunctivitis (VKC)"
]

# -----------------------------------------------------------------------------
# 2. Custom Feature Engineering (Dual-View Generator)
# -----------------------------------------------------------------------------
def create_diagnostic_view(image_pil):
    """
    Generates the custom 3-channel diagnostic feature map:
    - Channel 1: CLAHE (Corneal/Surface Texture)
    - Channel 2: Green Channel (Vascular Hyperemia)
    - Channel 3: Unsharp Masking (Structural Edges)
    """
    img_np = np.array(image_pil.convert('RGB'))
    
    # 1. CLAHE (Texture Enhancement)
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # 2. Green Channel (Vascular Extraction)
    green_ch = img_np[:, :, 1]
    
    # 3. Unsharp Masking (Edge Sharpening)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    unsharp = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    
    # Stack channels into a single composite 3-channel image
    diagnostic_map = np.stack([cl, green_ch, unsharp], axis=-1)
    return Image.fromarray(diagnostic_map)


# -----------------------------------------------------------------------------
# 3. Dual-View Swin Transformer Model Definition
# -----------------------------------------------------------------------------
class DualViewSwinModel(nn.Module):
    def __init__(self, num_classes=6, backbone_name="swin_base_patch4_window12_384"):
        super(DualViewSwinModel, self).__init__()
        # Global RGB Backbone
        self.global_branch = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        # Diagnostic Feature Map Backbone
        self.diag_branch = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        
        in_features = self.global_branch.num_features * 2  # 1024 + 1024 = 2048
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.GELU(),
            nn.Linear(in_features, 512),
            nn.Dropout(p=0.2),
            nn.GELU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_global, x_diag):
        feat_global = self.global_branch(x_global)
        feat_diag = self.diag_branch(x_diag)
        fused = torch.cat((feat_global, feat_diag), dim=1)
        out = self.classifier(fused)
        return out


# -----------------------------------------------------------------------------
# 4. Model Loading with Streamlit Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model(weights_path="final_swin_fold_1.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualViewSwinModel(num_classes=6)
    
    # Load model weights
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model, device

# -----------------------------------------------------------------------------
# 5. Image Preprocessing Pipelines
# -----------------------------------------------------------------------------
transform_pipeline = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -----------------------------------------------------------------------------
# 6. User Interface & Execution Logic
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Choose an eye photograph...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    raw_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(raw_image, caption="Uploaded Global View", use_column_width=True)
    
    with col2:
        # Generate and show diagnostic feature map
        diag_image = create_diagnostic_view(raw_image)
        st.image(diag_image, caption="Generated Diagnostic Map (Vascular/Texture)", use_column_width=True)
        
    if st.button("Run Diagnostic Screening"):
        with st.spinner("Analyzing ocular features through Dual-View Swin Transformer..."):
            try:
                # Load model
                model, device = load_trained_model("final_swin_fold_1.pth")
                
                # Preprocess both inputs
                tensor_global = transform_pipeline(raw_image).unsqueeze(0).to(device)
                tensor_diag = transform_pipeline(diag_image).unsqueeze(0).to(device)
                
                # Inference
                with torch.no_grad():
                    outputs = model(tensor_global, tensor_diag)
                    probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                
                predicted_idx = np.argmax(probabilities)
                predicted_label = CLASS_NAMES[predicted_idx]
                confidence = probabilities[predicted_idx] * 100
                
                # Display Results
                st.success(f"**Primary Diagnosis:** {predicted_label}")
                st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
                
                # Display Probability Distribution
                st.subheader("Diagnostic Probability Breakdown")
                for i, class_name in enumerate(CLASS_NAMES):
                    prob_pct = probabilities[i] * 100
                    st.write(f"**{class_name}**: {prob_pct:.2f}%")
                    st.progress(float(probabilities[i]))
                    
            except Exception as e:
                st.error(f"Error during model inference: {e}")
