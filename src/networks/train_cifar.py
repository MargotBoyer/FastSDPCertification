import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from typing import Literal
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from typing import List, Dict

SUPPORTED_DATASETS = {
    'cifar10': 10,
    'cifar100': 100,
    'mnist': 10,
    'imagenet': 1000
}
class ffn(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 10, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input if needed
        x =  self.net(x)
        x = self.fc(x)
        x = F.sigmoid(x)
        return x
    
def get_model(name: str, dataset: str, pretrained: bool = False, input_dim=784) -> nn.Module:
    if name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif name == 'ffn':
        model = ffn(input_dim=input_dim, hidden_dim=128, dropout=0.0)
    elif name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif name == 'vit_b_16':
        model = models.vit_b_16(pretrained=pretrained)
    elif name == 'vit_b_32':
        model = models.vit_b_32(pretrained=pretrained)

        raise ValueError(f"Unsupported model: {name}")

    num_classes = SUPPORTED_DATASETS.get(dataset.lower())
    if num_classes is None:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Replace the final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
def extract_multiple_features(model: nn.Module, layer_names: list[str], x: torch.Tensor, pooling: str = "flatten") -> dict:
    return_nodes = {name: name for name in layer_names}
    extractor = create_feature_extractor(model, return_nodes=return_nodes)
    with torch.no_grad():
        features = extractor(x)
    if pooling == "flatten":
        for k in features:
            features[k] = features[k].view(features[k].size(0), -1)
    elif pooling == "avgpool":
            for k in features:
                features[k] = torch.nn.functional.adaptive_avg_pool2d(features[k], 1).view(features[k].size(0), -1)
    elif pooling == "maxpool":
            for k in features:
                features[k] = torch.nn.functional.adaptive_max_pool2d(features[k], 1).view(features[k].size(0), -1)
    elif pooling != "none":
            raise ValueError(f"Invalid pooling method: {pooling}")
    return features
def collect_activations(model: nn.Module, dataloader, layers: List[str], device="cpu") -> Dict[str, torch.Tensor]:
    model.eval()
    outputs = {name: [] for name in layers}

    def get_hook(name):
        def hook(module, input, output):
            outputs[name].append(output.detach().cpu())
        return hook

    hooks = []
    for name, module in model.named_modules():
        if name in layers:
            hooks.append(module.register_forward_hook(get_hook(name)))

    with torch.no_grad():
        for x, _, _ in tqdm(dataloader, desc=f"Collect activations for {model.__class__.__name__}"):
            x = x.to(device)
            model(x)

    for hook in hooks:
        hook.remove()

    return {k: torch.cat(v, dim=0) for k, v in outputs.items()}
def list_model_layers(model: nn.Module) -> list[str]:
    return [name for name, _ in model.named_modules()]
def get_regressor(arch: str, input_dim: int) -> nn.Module:
    if arch == "mlp_small":
        return nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    elif arch == "mlp_large":
        return nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    else:
        raise ValueError(f"Unknown regressor architecture: {arch}")