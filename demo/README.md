---
title: Vektora Wake Word Demo
emoji: 🎙️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
short_description: A tiny CNN that listens for one word, "Vektora"
---

# Vektora — wake-word detector

Press record, say **"Vektora"**, and a ~24,000-parameter convolutional neural
network, trained from scratch on our team's recordings, decides whether it
heard its wake word.

- **Input:** 1-second windows of 16 kHz audio → log-mel spectrogram (98 × 40)
- **Model:** 3-block CNN → global average pooling → sigmoid score
- **Data:** ~60 "Vektora" recordings from 5–6 speakers, look-alike words
  (vector, victor), everyday speech, and noise; heavily augmented
- **Built with:** TensorFlow/Keras, librosa, Streamlit

Part of project SIH26172.
