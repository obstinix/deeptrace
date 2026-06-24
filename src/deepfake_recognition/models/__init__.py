from deepfake_recognition.models.resnet import DeepfakeResNet18, build_model as build_resnet
from deepfake_recognition.models.efficientnet import DeepfakeEfficientNet, build_model as build_efficientnet
from deepfake_recognition.models.vit import DeepfakeViT, build_model as build_vit

MODEL_REGISTRY = {
    "resnet18": build_resnet,
    "efficientnet_b3": build_efficientnet,
    "vit_base": build_vit,
}

def get_model(cfg: dict):
    name = cfg["name"]
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](cfg)
