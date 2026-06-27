from flask import Flask, render_template, send_file
import torch
import torch.nn as nn
import io
from PIL import Image
import numpy as np

app = Flask(__name__)

# Définition d'un générateur très léger (CNN)
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

# Chargement du modèle (Attention : il faut un fichier 'generator.pth' entraîné)
model = SimpleGenerator()
# model.load_state_dict(torch.load('generator.pth', map_location='cpu'))
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate')
def generate():
    noise = torch.randn(1, 100, 1, 1)
    with torch.no_grad():
        img = model(noise)
    
    # Conversion tensor vers image
    img = (img[0].permute(1, 2, 0).numpy() + 1) / 2
    img = (img * 255).astype(np.uint8)
    image = Image.fromarray(img)
    
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(port=5000)
