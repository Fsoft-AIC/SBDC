import torch

_MODELS = {
    'dinov2_vits14': 'dinov2_vits14',
    'dinov2_vitb14': 'dinov2_vitb14',
    'dinov2_vitl14': 'dinov2_vitl14',
    'dinov2_vitg14': 'dinov2_vitg14'
    }

class DinoV2(torch.nn.Module):
    def __init__(self, model_name, num_classes=10):
        super().__init__()
        self.model = torch.hub.load('/home/datnc13/LCT/sample_selection/models/dinov2', 
                                    _MODELS.get(model_name, 'dinov2_vitl14'), source="local")
        self.embed_dim = self.model.num_features
        self.logits = torch.nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.model(x)
        return self.logits(features)
