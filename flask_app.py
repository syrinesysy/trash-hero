
# app.py - API Flask pour Trash Hero

from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import torchvision.transforms as transforms
import io

app = Flask(__name__)

# Charger le modèle au démarrage
device = torch.device('cpu')
model = mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, 6)  # 6 classes
)
model.load_state_dict(torch.load('mobilenet_final_best.pth', map_location=device))
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Lire l'image
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

    # Préprocesser
    img_tensor = transform(image).unsqueeze(0)

    # Prédire
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        confidence, pred_idx = torch.max(probs, 0)

    result = {
        'class': class_names[pred_idx.item()],
        'confidence': float(confidence.item()),
        'probabilities': {cls: float(prob) for cls, prob in zip(class_names, probs)}
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
