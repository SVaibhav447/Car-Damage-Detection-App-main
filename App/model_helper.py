import json
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torch import nn

trained_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load class names saved by train.py — falls back to VehiDE defaults
_names_file = Path("class_names.json")
if _names_file.exists():
    with open(_names_file) as f:
        class_names = json.load(f)
else:
    class_names = [
        "broken_glass", "broken_lights", "dent",
        "lost_parts", "non_damaged", "punctured",
        "scratch", "torn"
    ]


class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(weights="DEFAULT")
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def predict(image_path):
    global trained_model

    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    if trained_model is None:
        trained_model = CarClassifierResNet(len(class_names))
        trained_model.load_state_dict(
            torch.load("detection_model.pth",
                       map_location=device,
                       weights_only=True)
        )
        trained_model.to(device)
        trained_model.eval()

    with torch.no_grad():
        output = trained_model(image_tensor)
        probs  = F.softmax(output, dim=1)[0]
        conf, pred_idx = torch.max(probs, 0)

        # Log all class probabilities for debugging
        for name, prob in zip(class_names, probs):
            print(f"  {name}: {prob:.3f}")

    if conf.item() < 0.50:
        return "uncertain"

    return class_names[pred_idx.item()]