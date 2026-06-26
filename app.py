import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from diffusers import AutoPipelineForText2Image

app = Flask(__name__)
CORS(app)

OUTPUT_DIR = "static_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Chargement des modèles IA optimisés...")

# 1. Configuration pour charger le texte en 4-bit (prend 4x moins de place)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_text_id = "MistralAI/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_text_id)
text_model = AutoModelForCausalLM.from_pretrained(
    model_text_id, 
    quantization_config=quantization_config, 
    device_map="auto"
)

# 2. Utilisation de SDXL-Turbo (beaucoup plus rapide et léger)
model_image_id = "stabilityai/sdxl-turbo"
image_pipeline = AutoPipelineForText2Image.from_pretrained(
    model_image_id, 
    torch_dtype=torch.float16, 
    variant="fp16"
).to("cuda")

print("Modèles prêts !")

conversation_history = []

@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history
    data = request.json
    user_message = data.get("message", "")
    
    # Détection simple pour l'image
    trigger_words = ["génère", "dessine", "crée une image"]
    wants_image = any(word in user_message.lower() for word in trigger_words)
    
    conversation_history.append({"role": "user", "content": user_message})
    if len(conversation_history) > 6: conversation_history = conversation_history[-6:]
        
    inputs = tokenizer.apply_chat_template(conversation_history, return_tensors="pt").to("cuda")
    
    # Génération texte
    with torch.no_grad():
        outputs = text_model.generate(inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
    
    text_response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    conversation_history.append({"role": "assistant", "content": text_response})
    
    # Génération image ultra-rapide (1-4 steps seulement)
    image_url = None
    if wants_image:
        image = image_pipeline(prompt=user_message, num_inference_steps=2, guidance_scale=0.0).images[0]
        filename = f"img_{torch.randint(0, 100000, (1,)).item()}.png"
        image.save(os.path.join(OUTPUT_DIR, filename))
        image_url = f"http://127.0.0.1:5000/images/{filename}"
    
    return jsonify({"text": text_response, "image": image_url})

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    app.run(port=5000)
