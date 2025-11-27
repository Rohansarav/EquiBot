import os
import json
import random
from html import escape

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

# ---------- UI THEME ----------
st.set_page_config(page_title="EquiBot (AIESEC)", page_icon="🤖", layout="centered")

CUSTOM_CSS = """
<style>
:root {
    --accent: #de0d7a;
    --accent-soft: rgba(222, 13, 122, 0.25);
    --bg: #050509;
    --surface: #141420;
    --surface-alt: #1b1b2a;
    --text: #f5f5f7;
    --muted: #b3b3c3;
}

/* Main app background */
[data-testid="stAppViewContainer"] {
    background-color: var(--bg);
    color: var(--text);
}

/* SDG watermark background */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image: url("sdg_logo.png");
    background-repeat: no-repeat;
    background-position: center;
    background-size: 45%;
    opacity: 0.1;
    pointer-events: none;
    z-index: -1;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #080812;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

/* Typography */
h1, h2, h3, h4, h5, h6 {
    color: var(--text);
}

p, span, label, .stMarkdown {
    color: var(--text);
}

/* Buttons */
.stButton > button {
    background: var(--accent);
    color: #ffffff;
    border-radius: 999px;
    border: none;
    padding: 0.45rem 1.4rem;
    font-weight: 600;
    font-size: 0.95rem;
    box-shadow: 0 0 0 0 var(--accent-soft);
    transition: all 140ms ease-out;
}
.stButton > button:hover {
    background: #ff3f9b;
    box-shadow: 0 0 18px var(--accent-soft);
    transform: translateY(-1px);
}

/* Forms / text input */
[data-testid="stTextInputRoot"] {
    background: var(--surface);
    border-radius: 999px;
    padding: 0.1rem 0.75rem;
}
[data-testid="stTextInputRoot"] input {
    background: transparent;
    color: var(--text);
}
[data-testid="stTextInputRoot"] input::placeholder {
    color: var(--muted);
}

/* Chat container card with gradient background */
.chat-wrapper {
    background:
        radial-gradient(circle at top left, rgba(222,13,122,0.35), transparent 55%),
        radial-gradient(circle at bottom right, rgba(87,100,255,0.28), transparent 55%),
        rgba(10, 10, 20, 0.9);
    border-radius: 1.3rem;
    padding: 1.1rem 1.25rem;
    border: 1px solid rgba(255, 255, 255, 0.06);
    backdrop-filter: blur(10px);
}

/* Chat rows */
.chat-row {
    display: flex;
    align-items: flex-end;
    margin-bottom: 0.7rem;
    gap: 0.5rem;
}

/* Row alignment */
.chat-row.bot {
    justify-content: flex-start;
}
.chat-row.user {
    justify-content: flex-end;
}

/* Avatar circles */
.chat-row.bot::before,
.chat-row.user::after {
    flex-shrink: 0;
    width: 32px;
    height: 32px;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    background: #1f1f2b;
    box-shadow: 0 0 10px rgba(0,0,0,0.7);
}
.chat-row.bot::before {
    content: "🤖";
}
.chat-row.user::after {
    content: "🙂";
}

/* Chat bubbles */
.chat-bubble {
    max-width: 80%;
    padding: 0.7rem 1rem;
    border-radius: 1rem;
    font-size: 0.95rem;
    line-height: 1.4;
    position: relative;
    transition: box-shadow 150ms ease-out, transform 150ms ease-out;
}

/* Hover glow */
.chat-bubble:hover {
    box-shadow: 0 0 16px var(--accent-soft);
    transform: translateY(-1px);
}

/* User bubble */
.chat-bubble.user-bubble {
    background: linear-gradient(135deg, #de0d7a, #ff6fb5);
    color: #ffffff;
    border-bottom-right-radius: 0.27rem;
}

/* Bot bubble */
.chat-bubble.bot-bubble {
    background: linear-gradient(135deg, #171727, #26263a);
    color: var(--text);
    border-bottom-left-radius: 0.27rem;
    border: 1px solid rgba(255, 255, 255, 0.06);
}

/* Small intent/confidence text */
.bot-meta {
    margin-top: 0.25rem;
    font-size: 0.78rem;
    color: var(--muted);
}

/* Caption under title */
.app-caption {
    color: var(--muted);
}

/* Typing indicator */
.typing-text {
    font-size: 0.9rem;
    color: var(--muted);
    margin-right: 0.3rem;
}
.typing-dots {
    display: inline-flex;
    gap: 3px;
}
.typing-dot {
    width: 6px;
    height: 6px;
    border-radius: 999px;
    background: var(--muted);
    opacity: 0.5;
    animation: typing-bounce 1s infinite ease-in-out alternate;
}
.typing-dot:nth-child(2) {
    animation-delay: 0.18s;
}
.typing-dot:nth-child(3) {
    animation-delay: 0.36s;
}
@keyframes typing-bounce {
    from {
        transform: translateY(0);
        opacity: 0.3;
    }
    to {
        transform: translateY(-4px);
        opacity: 1;
    }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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


# ---------- SESSION STATE SETUP & CLEANUP ----------
greeting_msg = {
    "role": "assistant",
    "text": "Hi! I’m EquiBot. How can I help you today?",
    "meta": "",
}

# initialise base keys
if "messages" not in st.session_state:
    st.session_state.messages = [greeting_msg]

if "pending_inference" not in st.session_state:
    st.session_state.pending_inference = False

if "last_user_text" not in st.session_state:
    st.session_state.last_user_text = ""

# clean out any legacy stray `</div>` messages from older versions
cleaned = []
for m in st.session_state.messages:
    if isinstance(m, dict):
        txt = str(m.get("text", "")).strip()
        # drop pure "</div>" or "<div>" leftovers
        if txt in ("</div>", "<div>", "<div/>"):
            continue
        cleaned.append(m)
st.session_state.messages = cleaned

# if chat somehow ended up empty, re-add greeting
if len(st.session_state.messages) == 0:
    st.session_state.messages.append(greeting_msg)


# ---------- GATE (OPTIONAL) ----------
if APP_SECRET:
    with st.sidebar:
        st.subheader("Access")
        user_secret = st.text_input("Enter access code", type="password")
        if st.button("Unlock"):
            if user_secret == APP_SECRET:
                st.session_state["unlocked"] = True
            else:
                st.error("Invalid code")

    if not st.session_state.get("unlocked"):
        st.info("This demo is limited-access. Ask the owner for the access code.")
        st.stop()

# ---------- MAIN UI ----------
st.title("🤖 EquiBot — Equify Project Assistant")
st.caption("Ask me about Project Equify, logistics, accommodation, and who to contact.")
st.markdown(
    '<p class="app-caption">Powered by AI, aligned with SDG 10: Reduced Inequalities.</p>',
    unsafe_allow_html=True,
)

chat_container = st.container()

# ---------- RENDER CHAT ----------
with chat_container:
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    for m in st.session_state.messages:
        role = m.get("role", "assistant")

        # Typing placeholder
        if m.get("typing", False):
            st.markdown(
                """
                <div class="chat-row bot">
                    <div class="chat-bubble bot-bubble">
                        <span class="typing-text">EquiBot is typing</span>
                        <span class="typing-dots">
                            <span class="typing-dot"></span>
                            <span class="typing-dot"></span>
                            <span class="typing-dot"></span>
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            continue

        if role == "user":
            safe_text = escape(m.get("text", ""))
            st.markdown(
                f"""
                <div class="chat-row user">
                    <div class="chat-bubble user-bubble">
                        {safe_text}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:  # assistant normal message
            text = m.get("text", "")
            meta = m.get("meta", "")
            safe_text = escape(text)
            safe_meta = escape(meta)

            st.markdown(
                f"""
                <div class="chat-row bot">
                    <div class="chat-bubble bot-bubble">
                        {safe_text}
                        {"<div class='bot-meta'>" + safe_meta + "</div>" if meta else ""}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- INPUT BOX ----------
with st.form("chat_form", clear_on_submit=True):
    user_msg = st.text_input("Type your message…")
    send = st.form_submit_button("Send")

# When user sends a message: add user + typing bubble, then rerun
if send and user_msg.strip():
    st.session_state.last_user_text = user_msg.strip()
    st.session_state.messages.append(
        {"role": "user", "text": st.session_state.last_user_text, "meta": ""}
    )
    st.session_state.messages.append(
        {"role": "assistant", "text": "", "meta": "", "typing": True}
    )
    st.session_state.pending_inference = True
    st.rerun()

# After rerun, if pending, replace typing bubble with real response
if st.session_state.pending_inference:
    user_text = st.session_state.last_user_text

    tag, conf = predict_intent(user_text)
    bot_main = reply_for(tag)
    bot_meta = f"intent: `{tag}` · confidence: {conf:.2f}"

    # Find typing message index
    typing_index = None
    for i, m in enumerate(st.session_state.messages):
        if m.get("typing", False):
            typing_index = i
            break

    if typing_index is not None:
        st.session_state.messages[typing_index] = {
            "role": "assistant",
            "text": bot_main,
            "meta": bot_meta,
        }

    st.session_state.pending_inference = False
    st.rerun()
