import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionXLPipeline
import os

# Configuration de la page
st.set_page_config(page_title="IA Multimodale", layout="centered")
st.title("🤖 IA Multimodale : Mistral & SDXL")

# 1. Chargement des modèles avec mise en cache
@st.cache_resource
def load_models():
    # Chargement du modèle de texte
    model_text_id = "MistralAI/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_text_id)
    text_model = AutoModelForCausalLM.from_pretrained(
        model_text_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    # Chargement du modèle d'image
    model_image_id = "stabilityai/stable-diffusion-xl-base-1.0"
    image_pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_image_id, 
        torch_dtype=torch.float16, 
        use_safetensors=True
    ).to("cuda")
    
    return tokenizer, text_model, image_pipeline

# Charger les modèles (attention : nécessite un espace avec GPU sur HF)
try:
    tokenizer, text_model, image_pipeline = load_models()
except Exception as e:
    st.error(f"Erreur de chargement des modèles : {e}")

# 2. Interface utilisateur
user_message = st.text_input("Pose une question ou demande une image :")

if st.button("Envoyer"):
    if user_message:
        # Détection de la demande d'image
        trigger_words = ["génère", "dessine", "crée une image", "photo de"]
        wants_image = any(word in user_message.lower() for word in trigger_words)
        
        # Génération du texte
        with st.spinner("L'IA réfléchit..."):
            inputs = tokenizer.apply_chat_template([{"role": "user", "content": user_message}], return_tensors="pt").to("cuda")
            outputs = text_model.generate(inputs, max_new_tokens=200)
            text_response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
            st.write("### Réponse :")
            st.write(text_response)
        
        # Génération de l'image si demandée
        if wants_image:
            with st.spinner("Génération de l'image en cours..."):
                image = image_pipeline(prompt=user_message, num_inference_steps=20).images[0]
                st.image(image, caption="Image générée par l'IA")
    else:
        st.warning("Veuillez entrer un message.")
