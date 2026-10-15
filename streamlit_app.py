import os
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import timm
import requests
from huggingface_hub import hf_hub_download

# -----------------------------------------------------------------------------
# 1. Page Configuration & UI Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Anterior Ocular Diagnostic System",
    page_icon="👁️",
    layout="centered"
)

st.title("👁️ Anterior Ocular Diagnostic System")
st.write(
    "Upload a photograph of an anterior segment eye condition to run diagnostic "
    "screening using your Swin Transformer model (Single-Fold Optimized)."
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
# 2. Hugging Face Repository Configuration
# -----------------------------------------------------------------------------
REPO_ID = "tav1212/VKC-swin-weights"

# -----------------------------------------------------------------------------
# 3. Custom Feature Engineering (Dual-View Generator)
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
# 4. Dual-View Swin Transformer Model Architecture
# -----------------------------------------------------------------------------
class DualViewSwinModel(nn.Module):
    def __init__(self, num_classes=6, backbone_name="swin_base_patch4_window12_384"):
        super(DualViewSwinModel, self).__init__()
        # Global RGB Branch
        self.global_branch = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        # Diagnostic Feature Branch
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
# 5. Single-Fold Model Loader & Key-Remapper
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    filename = "final_swin_fold_1.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Download weights if not cached locally
    if not os.path.exists(filename):
        url = f"https://huggingface.co/{REPO_ID}/resolve/main/{filename}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        
        with st.spinner("Downloading model weights for Fold 1 from Hugging Face..."):
            try:
                weights_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
            except Exception:
                response = requests.get(url, headers=headers, stream=True)
                if response.status_code == 200:
                    with open(filename, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    weights_path = filename
                else:
                    st.error(f"Failed to download weights (HTTP Status {response.status_code}). Ensure repository '{REPO_ID}' is Public.")
                    st.stop()
    else:
        weights_path = filename
            
    model = DualViewSwinModel(num_classes=6)
    checkpoint = torch.load(weights_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
        
    # Remap state dict keys dynamically to match DualViewSwinModel architecture
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("global_branch.") or k.startswith("diag_branch.") or k.startswith("classifier."):
            new_state_dict[k] = v
        elif k.startswith("backbone."):
            clean_k = k.replace("backbone.", "")
            new_state_dict[f"global_branch.{clean_k}"] = v
            new_state_dict[f"diag_branch.{clean_k}"] = v
        else:
            if "classifier" in k or "fc" in k or "head" in k:
                new_state_dict[k.replace("model.", "")] = v
            else:
                new_state_dict[f"global_branch.{k}"] = v
                new_state_dict[f"diag_branch.{k}"] = v
                
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, device

# -----------------------------------------------------------------------------
# 6. Preprocessing Pipeline
# -----------------------------------------------------------------------------
transform_pipeline = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -----------------------------------------------------------------------------
# 7. User Interface & Single-Model Prediction Logic
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload an eye image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(raw_image, caption="Global RGB View")
    
    with col2:
        diag_image = create_diagnostic_view(raw_image)
        st.image(diag_image, caption="Diagnostic View (CLAHE/Green/Unsharp)")
        
    if st.button("Run Diagnostic Analysis"):
        with st.spinner("Running model inference..."):
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                tensor_global = transform_pipeline(raw_image).unsqueeze(0).to(device)
                tensor_diag = transform_pipeline(diag_image).unsqueeze(0).to(device)
                
                model, dev = load_model()
                with torch.no_grad():
                    outputs = model(tensor_global, tensor_diag)
                    probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                
                predicted_idx = int(np.argmax(probs))
                predicted_label = CLASS_NAMES[predicted_idx]
                confidence = float(probs[predicted_idx] * 100)
                
                st.success(f"**Predicted Diagnosis:** {predicted_label}")
                st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
                
                st.subheader("Class Probability Distribution")
                for i, class_name in enumerate(CLASS_NAMES):
                    prob_pct = float(probs[i] * 100)
                    st.write(f"**{class_name}**: {prob_pct:.2f}%")
                    st.progress(float(probs[i]))
                    
            except Exception as e:
                st.error(f"Error executing model inference: {e}")
