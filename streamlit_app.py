import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionXLPipeline

# Configuration de la page
st.set_page_config(page_title="Mon IA", layout="centered")

# Chargement des modèles (utilise @st.cache_resource pour ne pas recharger à chaque interaction)
@st.cache_resource
def load_models():
    # ... ton code de chargement ici ...
    return tokenizer, text_model, image_pipeline

tokenizer, text_model, image_pipeline = load_models()

# Interface Streamlit
st.title("🤖 IA Multimodale")

user_input = st.text_input("Pose ta question ou demande une image :")

if st.button("Envoyer"):
    # Logique de génération texte
    # ...
    st.write(text_response)
    
    # Logique image
    if "dessine" in user_input.lower():
        image = image_pipeline(prompt=user_input).images[0]
        st.image(image, caption="Généré par IA")
