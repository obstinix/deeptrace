import cv2
import numpy as np
import torch
import torch.nn.functional as F
import base64
from PIL import Image
import io

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook the target layer
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        b, c, h, w = input_tensor.size()
        
        # Forward pass
        self.model.eval()
        logits = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = logits.argmax(dim=-1).item()
            
        score = logits[0, class_idx]
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pooling on gradients
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Apply weights to activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w_i in enumerate(weights):
            cam += w_i * activations[i]
            
        cam = F.relu(cam)
        cam = cv2.resize(cam.cpu().numpy(), (w, h))
        
        # Normalize
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam

def _resolve_target_layer(model):
    """Robustly resolve the last convolutional layer for GradCAM."""
    # ResNet path: DeepfakeResNet18 uses self.features
    if hasattr(model, "features"):
        features = model.features
        last_block = features[-1]
        if hasattr(last_block, "conv2"):
            return last_block.conv2
        return last_block

    # EfficientNet path: DeepfakeEfficientNet uses self.backbone (timm)
    if hasattr(model, "backbone"):
        backbone = model.backbone
        if hasattr(backbone, "blocks") and len(backbone.blocks) > 0:
            return backbone.blocks[-1]

    return None

def generate_gradcam_base64(model, input_tensor, original_image):
    """
    Generate Grad-CAM heatmap and overlay on original image, returning base64 PNG.
    """
    target_layer = _resolve_target_layer(model)
    if target_layer is None:
        return None

    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(input_tensor)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Resize original image to match tensor size
    _, _, h, w = input_tensor.size()
    original_image = original_image.resize((w, h))
    img_np = np.array(original_image)
    
    # Overlay (60% heatmap, 40% original)
    overlay = cv2.addWeighted(img_np, 0.4, heatmap, 0.6, 0)
    
    # Encode to base64
    overlay_img = Image.fromarray(overlay)
    buffered = io.BytesIO()
    overlay_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return f"data:image/png;base64,{img_str}"
