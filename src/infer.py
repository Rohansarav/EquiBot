import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

# Load full model (vectorizer already inside)
model = keras.models.load_model("models/intent_model.keras")

# Load label classes
label_classes = np.load("models/label_classes.npy", allow_pickle=True)

# Load responses
with open("data/intents.json", "r", encoding="utf-8") as f:
    data = json.load(f)
responses = {it["tag"]: it["responses"] for it in data["intents"]}

def predict_intent(text):
    x = tf.constant([text], dtype=tf.string)
    pred = model.predict(x, verbose=0)
    idx = int(np.argmax(pred))
    tag = str(label_classes[idx])
    conf = float(np.max(pred))
    return tag, conf

if __name__ == "__main__":
    print("EquiBot ready! Type something or 'exit' to quit.")
    while True:
        msg = input("You: ")
        if msg.lower() in ["quit", "exit"]:
            break
        tag, conf = predict_intent(msg)
        reply = random.choice(responses.get(tag, ["I'm not sure, could you rephrase?"]))
        print(f"EquiBot ({conf:.2f}): {reply}")
