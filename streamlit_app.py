import os
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import timm
import gdown

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
    "screening using a 5-Fold Dual-View Swin Transformer Ensemble."
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
# 2. Google Drive File IDs Mapping (Replace with your actual IDs)
# -----------------------------------------------------------------------------
DRIVE_FILE_IDS = {
    1: "1knWq2jfBhrAA_cUIaw2u0iEih-IUh9ir",
    2: "11YOuy5KexQocaaDl03hFlkBWPTCN0ytr",
    3: "1JwIi84FLbGLCftRo_9tktjjg5Mlf3GN-",
    4: "1WwxbeukB3HIMH9OGXQOyCpN9Lp1ZcV6A",
    5: "1w9M28roDaM49sawfROJBipcRIDKd3BDK",
}

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
# 5. Model Loading & Auto-Downloading Function
# -----------------------------------------------------------------------------

@st.cache_resource
def load_single_fold_model(fold_num):
    weights_path = f"final_swin_fold_{fold_num}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Download weights if not locally cached
    if not os.path.exists(weights_path):
        file_id = DRIVE_FILE_IDS.get(fold_num)
        if not file_id or "FILE_ID_" in file_id:
            st.error(f"Google Drive File ID for Fold {fold_num} is missing or unconfigured.")
            st.stop()
            
        with st.spinner(f"Downloading model weights for Fold {fold_num} from Google Drive..."):
            # Using gdown's id parameter directly
            gdown.download(id=file_id, output=weights_path, quiet=False)
            
    model = DualViewSwinModel(num_classes=6)
    checkpoint = torch.load(weights_path, map_location=device)
    
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
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
# 7. User Interface & Ensemble Prediction Logic
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload an eye image (JPG, JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(raw_image, caption="Global RGB View", use_column_width=True)
    
    with col2:
        diag_image = create_diagnostic_view(raw_image)
        st.image(diag_image, caption="Diagnostic View (CLAHE/Green/Unsharp)", use_column_width=True)
        
    if st.button("Run Diagnostic Analysis"):
        with st.spinner("Running 5-Fold Ensemble Committee Analysis..."):
            try:
                # Preprocess inputs
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                tensor_global = transform_pipeline(raw_image).unsqueeze(0).to(device)
                tensor_diag = transform_pipeline(diag_image).unsqueeze(0).to(device)
                
                ensemble_probs = np.zeros(6)
                
                # Predict across all 5 folds
                for fold in range(1, 6):
                    model, dev = load_single_fold_model(fold)
                    with torch.no_grad():
                        outputs = model(tensor_global, tensor_diag)
                        fold_probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                        ensemble_probs += fold_probs / 5.0
                
                predicted_idx = int(np.argmax(ensemble_probs))
                predicted_label = CLASS_NAMES[predicted_idx]
                confidence = float(ensemble_probs[predicted_idx] * 100)
                
                # Display Prediction Results
                st.success(f"**Predicted Diagnosis (Ensemble Consensus):** {predicted_label}")
                st.metric(label="Ensemble Model Confidence", value=f"{confidence:.2f}%")
                
                # Display Probability Breakdown
                st.subheader("Class Probability Distribution")
                for i, class_name in enumerate(CLASS_NAMES):
                    prob_pct = float(ensemble_probs[i] * 100)
                    st.write(f"**{class_name}**: {prob_pct:.2f}%")
                    st.progress(float(ensemble_probs[i]))
                    
            except Exception as e:
                st.error(f"Error executing model inference: {e}")
