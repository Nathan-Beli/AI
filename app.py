import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionXLPipeline

app = Flask(__name__)
CORS(app)

OUTPUT_DIR = "static_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Chargement des modèles IA...")
model_text_id = "MistralAI/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_text_id)
text_model = AutoModelForCausalLM.from_pretrained(model_text_id, torch_dtype=torch.float16, device_map="auto")

model_image_id = "stabilityai/stable-diffusion-xl-base-1.0"
image_pipeline = StableDiffusionXLPipeline.from_pretrained(model_image_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
print("Modèles prêts !")

# --- TABLEAU POUR SAUVEGARDER LA MÉMOIRE DE LA DISCUSSION ---
conversation_history = []

@app.route('/chat', methods=['POST'])
def chat():
    global conversation_history
    data = request.json
    user_message = data.get("message", "")
    
    # 1. Détection si l'utilisateur veut une image
    trigger_words = ["génère", "dessine", "crée une image", "fais un dessin", "photo de"]
    wants_image = any(word in user_message.lower() for word in trigger_words)
    
    # 2. On ajoute le nouveau message à l'historique
    conversation_history.append({"role": "user", "content": user_message})
    
    # Garder seulement les 10 derniers messages (5 tours de parole) pour ne pas saturer la mémoire
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]
        
    # 3. Préparation du texte pour le modèle (avec tout le contexte historique)
    inputs = tokenizer.apply_chat_template(conversation_history, return_tensors="pt").to("cuda")
    outputs = text_model.generate(inputs, max_new_tokens=250, do_sample=True, temperature=0.7)
    
    # Récupérer uniquement la nouvelle réponse générée
    text_response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    
    # 4. On ajoute la réponse de l'IA à l'historique pour la prochaine fois
    conversation_history.append({"role": "assistant", "content": text_response})
    
    # 5. Gestion de l'image si demandée
    image_url = None
    if wants_image:
        image = image_pipeline(prompt=user_message, num_inference_steps=25).images[0]
        filename = f"img_{torch.randint(0, 100000, (1,)).item()}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        image.save(filepath)
        image_url = f"http://127.0.0.1:5000/images/{filename}"
    
    return jsonify({
        "text": text_response,
        "image": image_url
    })

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    app.run(port=5000, debug=False)
