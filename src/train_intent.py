import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder

# 1. Load data
with open(os.path.join("data", "intents.json"), "r", encoding="utf-8") as f:
    intents = json.load(f)

texts = []
labels = []

for intent in intents["intents"]:
    tag = intent["tag"]
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(tag)

# 2. Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
os.makedirs("models", exist_ok=True)
np.save("models/label_classes.npy", label_encoder.classes_)

# 3. TextVectorization (adapt before training)
max_tokens = 5000
seq_len = 20
vectorizer = layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=seq_len,
)
vectorizer.adapt(texts)   # builds the vocabulary table

# 4. Build the model
inputs = keras.Input(shape=(1,), dtype=tf.string) #input layer -> takes string
x = vectorizer(inputs) #vectorization layer
x = layers.Embedding(input_dim=max_tokens, output_dim=64)(x) #embedding layer -> converts words to vectors
x = layers.GlobalAveragePooling1D()(x) 
x = layers.Dense(64, activation="relu")(x) #dense layer
outputs = layers.Dense(len(label_encoder.classes_), activation="softmax")(x) #output layer
model = keras.Model(inputs, outputs) #define the model

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 5. Train
texts_tf = tf.constant(texts, dtype=tf.string)
labels_np = np.asarray(labels_encoded, dtype=np.int32)
model.fit(texts_tf, labels_np, epochs=200, verbose=0)

# 6. Save the fully initialized model
model.save("models/intent_model.keras")

print("Training complete — saved to models/intent_model (includes vectorizer)")
