from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
import io
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app) # Permet la communication entre le frontend et le backend

# --- 1. Définition du Modèle (Doit correspondre à ton fichier .pth) ---
class SimpleGenerator(nn.Module):
    def __init__(self):
        super(SimpleGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 64, 4, 1, 0),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)

model = SimpleGenerator()
# model.load_state_dict(torch.load('generator.pth', map_location='cpu'))
model.eval()

# --- 2. Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').lower()
    
    # Logique simple de "cerveau" pour ton IA
    if "dessine" in user_message:
        return jsonify({
            "text": "Je génère l'image demandée...",
            "image": "/generate_image" 
        })
    else:
        return jsonify({
            "text": f"J'ai bien reçu ton message : '{user_message}'. Je ne peux pas encore répondre intelligemment, mais je suis en ligne !"
        })

@app.route('/generate_image')
def generate():
    # Génération de bruit et passage dans le modèle
    noise = torch.randn(1, 100, 1, 1)
    with torch.no_grad():
        img_tensor = model(noise)
    
    # Conversion du tensor (IA) en image (Format compréhensible par le navigateur)
    img = (img_tensor[0].permute(1, 2, 0).numpy() + 1) / 2
    img = (img * 255).astype(np.uint8)
    image = Image.fromarray(img)
    
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(port=5000)
