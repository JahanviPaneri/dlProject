import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

trained_model = None
class_names = ['No Damage', 'Minor Damage', 'Major Damage', 'Severe Damage']

# Load the pre-trained ResNet18 model (same as training)
class CarClassifierResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = models.resnet18(weights='DEFAULT')
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    global trained_model
    if trained_model is None:
        checkpoint = torch.load("model/saved_model.pth", map_location=torch.device('cpu'))
        new_checkpoint = {}
        for key in checkpoint.keys():
            new_checkpoint["model." + key] = checkpoint[key]
        trained_model = CarClassifierResNet18()
        trained_model.load_state_dict(new_checkpoint)
        trained_model.eval()
        

    with torch.no_grad():
        output = trained_model(image_tensor)
        _, predicted_class = torch.max(output, 1)
        return class_names[predicted_class.item()]
