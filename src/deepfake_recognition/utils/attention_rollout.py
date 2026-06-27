"""
src/deepfake_recognition/utils/attention_rollout.py

Attention Rollout for ViT-B/16.
Reference: Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (2020)
           https://arxiv.org/abs/2005.00928

Usage:
    rollout = AttentionRollout(model)
    mask    = rollout(input_tensor)   # shape: (H, W) float32 numpy array
    overlay = rollout.overlay(original_pil_image, mask)
"""
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class AttentionRollout:
    """
    Hooks into the attention weight tensors of a timm ViT-B/16 model
    and computes Attention Rollout across all transformer blocks.
    """

    def __init__(self, model: nn.Module, head_fusion: str = "mean",
                 discard_ratio: float = 0.9):
        """
        Args:
            model:         A timm vit_base_patch16_224 with num_classes=2
            head_fusion:   How to merge multi-head attention: "mean" | "max" | "min"
            discard_ratio: Fraction of lowest attention weights to zero out
                           before rollout. Reduces noise. Range: [0.0, 1.0)
        """
        self.model         = model
        self.head_fusion   = head_fusion
        self.discard_ratio = discard_ratio
        self._attention_weights: List[torch.Tensor] = []
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Register a forward hook on every attention module in the ViT blocks."""
        self._hooks = []
        for block in self.model.blocks:
            # timm ViT blocks expose attention weights via block.attn.attn_drop
            # We hook the softmax output of the attention module directly.
            hook = block.attn.register_forward_hook(self._hook_fn)
            self._hooks.append(hook)

    def _hook_fn(self, module, input, output):
        # timm's Attention module stores the raw attention map before dropout
        # in module.attn_weights when you set attn_drop_rate=0.
        # Fallback: recompute from the QK product captured in the module.
        if hasattr(module, "attn_weights"):
            self._attention_weights.append(module.attn_weights.detach())
        else:
            # Reconstruct from the output — this path is for timm < 1.0
            # where attn_weights is not stored as an attribute.
            # We catch it by patching forward; see _patch_attn_forward() below.
            pass

    def _patch_attn_forward(self):
        """
        Monkey-patch timm attention blocks to store attention weights.
        Called automatically if the hook-based approach finds no weights.
        """
        for block in self.model.blocks:
            orig_forward = block.attn.forward

            def make_patched(orig):
                def patched_forward(x, *args, **kwargs):
                    B, N, C = x.shape
                    qkv = block.attn.qkv(x)
                    qkv = qkv.reshape(B, N, 3, block.attn.num_heads,
                                      C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    scale   = math.sqrt(q.shape[-1])
                    attn    = (q @ k.transpose(-2, -1)) / scale
                    attn    = attn.softmax(dim=-1)
                    # Store for rollout
                    self._attention_weights.append(attn.detach())
                    attn_drop = block.attn.attn_drop(attn)
                    out = (attn_drop @ v).transpose(1, 2).reshape(B, N, C)
                    out = block.attn.proj(out)
                    out = block.attn.proj_drop(out)
                    return out
                return patched_forward

            block.attn.forward = make_patched(orig_forward)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Run the model on input_tensor and return an attention heatmap.

        Args:
            input_tensor: Shape (1, 3, 224, 224), normalised, on the model's device.

        Returns:
            numpy array of shape (14, 14) with values in [0, 1].
            Upscale to (224, 224) before overlaying on the image.
        """
        self._attention_weights = []
        self.model.eval()

        with torch.no_grad():
            _ = self.model(input_tensor)

        # If hooks didn't capture weights, try the patched approach
        if len(self._attention_weights) == 0:
            self.remove_hooks()
            self._patch_attn_forward()
            self._attention_weights = []
            with torch.no_grad():
                _ = self.model(input_tensor)

        if len(self._attention_weights) == 0:
            raise RuntimeError(
                "AttentionRollout: could not capture attention weights. "
                "Ensure timm >= 0.9 and attn_drop_rate=0.0 in build_model()."
            )

        return self._rollout()

    def _rollout(self) -> np.ndarray:
        # Each tensor: (B, num_heads, num_tokens, num_tokens)
        # For ViT-B/16: num_heads=12, num_tokens=197 (1 cls + 196 patches)
        result = torch.eye(
            self._attention_weights[0].size(-1),
            device=self._attention_weights[0].device,
        )

        for attn in self._attention_weights:
            # Fuse heads
            if self.head_fusion == "mean":
                attn = attn.mean(dim=1)          # (B, N, N)
            elif self.head_fusion == "max":
                attn = attn.max(dim=1).values
            elif self.head_fusion == "min":
                attn = attn.min(dim=1).values
            else:
                raise ValueError(f"Unknown head_fusion: {self.head_fusion}")

            attn = attn[0]                       # single image: (N, N)

            # Discard lowest-attention tokens
            flat       = attn.flatten().cpu()
            threshold  = flat.kthvalue(int(self.discard_ratio * flat.numel())).values
            attn[attn < threshold.to(attn.device)] = 0.0

            # Add identity (residual connection accounts for skip connections)
            attn  = attn + torch.eye(attn.size(0), device=attn.device)
            attn  = attn / attn.sum(dim=-1, keepdim=True)

            result = attn @ result

        # Extract CLS token row → patch attention scores
        # result[0] = how much the CLS token attends to each patch
        mask = result[0, 1:]                     # (196,) — skip CLS itself

        # Reshape to grid
        n_patches = int(math.sqrt(mask.size(0)))  # 14 for ViT-B/16
        mask = mask.reshape(n_patches, n_patches)

        # Normalise to [0, 1]
        mask = mask - mask.min()
        if mask.max() > 0:
            mask = mask / mask.max()

        return mask.cpu().numpy()

    @staticmethod
    def overlay(image: Image.Image, mask: np.ndarray,
                alpha: float = 0.5,
                colormap: str = "jet") -> Image.Image:
        """
        Overlay the attention mask on the original PIL image.

        Args:
            image:    Original PIL image (any size)
            mask:     Attention mask array (14×14 from rollout)
            alpha:    Blend weight for the heatmap overlay
            colormap: matplotlib colormap name

        Returns:
            PIL Image with the heatmap blended in
        """
        import matplotlib
        if hasattr(matplotlib, "colormaps"):
            cmap = matplotlib.colormaps[colormap]
        else:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(colormap)

        # Upsample mask to image size
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(
            image.size, resample=Image.BILINEAR
        )
        mask_arr = np.array(mask_img, dtype=np.float32) / 255.0

        # Apply colormap
        heatmap = (cmap(mask_arr)[:, :, :3] * 255).astype(np.uint8)
        heat_img = Image.fromarray(heatmap).convert("RGB")

        # Blend
        orig_rgb = image.convert("RGB")
        blended  = Image.blend(orig_rgb, heat_img, alpha=alpha)
        return blended
