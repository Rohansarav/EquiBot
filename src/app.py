import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import streamlit as st

# ---------- CONFIG ----------
MODEL_PATH = "models/intent_model.keras"
LABELS_PATH = "models/label_classes.npy"
INTENTS_PATH = "data/intents.json"

# (Optional) simple “password” gate for limited users
APP_SECRET = os.getenv("EQUIBOT_SECRET", "")  # set this env var on your machine/host

# ---------- LOAD ASSETS ----------
@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_labels():
    return np.load(LABELS_PATH, allow_pickle=True)

@st.cache_data
def load_intents():
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    resp_map = {it["tag"]: it.get("responses", []) for it in data["intents"]}
    return resp_map

model = load_model()
label_classes = load_labels()
responses = load_intents()

# ---------- HELPERS ----------
def predict_intent(text: str):
    x = tf.constant([text], dtype=tf.string)
    pred = model.predict(x, verbose=0)
    idx = int(np.argmax(pred))
    tag = str(label_classes[idx])
    conf = float(np.max(pred))
    return tag, conf

def reply_for(tag: str):
    options = responses.get(tag, [])
    if not options:
        return "I'm not sure yet — could you rephrase?"
    return random.choice(options)

# ---------- UI ----------
st.set_page_config(page_title="EquiBot (AIESEC)", page_icon="🤖", layout="centered")

# Gate (optional): limit to invited users with a shared secret
if APP_SECRET:
    with st.sidebar:
        st.subheader("Access")
        user_secret = st.text_input("Enter access code", type="password")
        if st.button("Unlock") and user_secret == APP_SECRET:
            st.session_state["unlocked"] = True
        elif st.button("Unlock"):
            st.error("Invalid code")

    if not st.session_state.get("unlocked"):
        st.info("This demo is limited-access. Ask the owner for the access code.")
        st.stop()

st.title("🤖 EquiBot — Equify Project Assistant")
st.caption("Ask me about Project Equify, logistics, accommodation, and who to contact.")

# Simple chat bubbles
chat_container = st.container()
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "text": "Hi! I’m EquiBot. How can I help you today?"}
    ]

# render history
with chat_container:
    for m in st.session_state.messages:
        if m["role"] == "user":
            st.markdown(f"**You:** {m['text']}")
        else:
            st.markdown(f"**EquiBot:** {m['text']}")

# input box
with st.form("chat_form", clear_on_submit=True):
    user_msg = st.text_input("Type your message…")
    send = st.form_submit_button("Send")

if send and user_msg.strip():
    st.session_state.messages.append({"role": "user", "text": user_msg})
    tag, conf = predict_intent(user_msg)
    bot_text = reply_for(tag) + f"  \n_(intent: `{tag}`, conf: {conf:.2f})_"
    st.session_state.messages.append({"role": "assistant", "text": bot_text})
    st.rerun()
