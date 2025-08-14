import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt

# =============================
# 1. Create & Train Model
# =============================
def create_model():
    try:
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        # Normalize and reshape
        train_images = train_images.astype('float32') / 255.0
        test_images = test_images.astype('float32') / 255.0
        train_images = train_images.reshape(-1, 28, 28, 1)
        test_images = test_images.reshape(-1, 28, 28, 1)

        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        model.fit(train_images, train_labels, epochs=3, batch_size=128,
                  validation_data=(test_images, test_labels), verbose=1)

        return model, (test_images, test_labels)

    except Exception as e:
        st.error(f"Error creating model: {e}")
        return None, (None, None)

# =============================
# 2. FGSM Attack
# =============================
def fgsm(model, images, labels, epsilon=0.2):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    with tf.GradientTape() as tape:
        tape.watch(images)
        prediction = model(images)
        loss = loss_object(labels, prediction)

    gradient = tape.gradient(loss, images)
    signed_grad = tf.sign(gradient)
    adv_images = images + epsilon * signed_grad
    adv_images = tf.clip_by_value(adv_images, 0, 1)

    return adv_images

# =============================
# 3. Compare Photos
# =============================
def comparephotos(original, adversarial, labels):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(5):
        axes[0, i].imshow(original[i].numpy().squeeze(), cmap='gray')
        axes[0, i].set_title(f"Orig: {labels[i].numpy()}")
        axes[0, i].axis('off')

        axes[1, i].imshow(adversarial[i].numpy().squeeze(), cmap='gray')
        axes[1, i].set_title("Adv")
        axes[1, i].axis('off')

    st.pyplot(fig)

# =============================
# Streamlit UI
# =============================
st.set_page_config(page_title="MNIST FGSM Demo", layout="wide")

st.title("MNIST Model with FGSM Attack 🧠⚡")

if st.button("Train & Attack Model"):
    with st.spinner("Training model... please wait"):
        model, (test_images, test_labels) = create_model()

    if model:
        idx = np.random.choice(len(test_images), 5)
        sample_images = tf.convert_to_tensor(test_images[idx])
        sample_labels = tf.convert_to_tensor(test_labels[idx])

        st.success("Model trained! Running FGSM attack...")
        adv_images = fgsm(model, sample_images, sample_labels)

        st.subheader("Original vs Adversarial Examples")
        comparephotos(sample_images, adv_images, sample_labels)

# Sidebar Chat
st.sidebar.title("💬 Mini Chatbot")
faq = {
    "What is FGSM?": "FGSM is a fast gradient sign method used to generate adversarial examples.",
    "What is MNIST?": "MNIST is a dataset of handwritten digits (0–9) used for training image processing systems."
}

selected_q = st.sidebar.selectbox("📋 Choose a question:", ["", *faq.keys()])
if selected_q:
    st.sidebar.write(faq[selected_q])
