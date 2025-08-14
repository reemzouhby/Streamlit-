import streamlit as st
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.title("Effect of different Epsilon on accuracy of Mnist Dataset")

from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier

# Load only test data since we're using a pre-trained model
(_, _), (test_images, test_labels) = mnist.load_data()

# Process only test data (much smaller memory footprint)
test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Try to load the model, with error handling
@st.cache_resource
def load_model():
    try:
        # Try different possible paths for the model
        model_paths = [
            "mnist_model.h5",
            "Practice/task1/mnist_model.h5",
            "./mnist_model.h5"
        ]
        
        model = None
        for path in model_paths:
            try:
                model = tf.keras.models.load_model(path)
                st.success(f"Model loaded successfully from: {path}")
                break
            except:
                continue
        
        if model is None:
            st.error("Model file not found. Creating a simple model for demonstration...")
            # Create a simple model for demonstration if the trained model is not available
            model = create_simple_model()
            
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return create_simple_model()

def create_simple_model():
    """Create a simple CNN model for MNIST (for demonstration purposes)"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    st.warning("Using untrained model for demonstration. Results may not be meaningful.")
    return model

# Load the model
model = load_model()

# Create ART KerasClassifier
if model:
    classifier = KerasClassifier(model=model, clip_values=(0, 1))

def fgsm(epsi):
    if not model:
        st.error("Model not available!")
        return None, None, None
        
    # Generate adversarial examples
    attack = FastGradientMethod(estimator=classifier, eps=epsi)
    x_test_adv = attack.generate(x=test_images)

    # Evaluate on clean and adversarial examples
    loss_clean, accuracy_clean = model.evaluate(test_images, test_labels, verbose=0)
    loss_adv, accuracy_adv = model.evaluate(x_test_adv, test_labels, verbose=0)

    return accuracy_clean, accuracy_adv, x_test_adv

def comparephotos(x_test_adv):
    if not model:
        st.error("Model not available!")
        return
        
    # --- Predictions ---
    pred_clean = np.argmax(model.predict(test_images), axis=1)
    pred_adv = np.argmax(model.predict(x_test_adv), axis=1)

    # --- Accuracy counts ---
    correct_clean = np.sum(pred_clean == test_labels)
    correct_adv = np.sum(pred_adv == test_labels)

    st.write(f"✅ Correct predictions (clean): {correct_clean}")
    st.write(f"⚠️ Correct predictions (adv): {correct_adv}")
    st.write(f"❌ Wrong predictions (clean): {len(test_labels) - correct_clean}")
    st.write(f"❌ Wrong predictions (adv): {len(test_labels) - correct_adv}")

    # --- Visual comparison ---
    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    for i in range(10):
        # Clean image
        axes[0, i].imshow(test_images[i].reshape(28, 28), cmap="gray")
        axes[0, i].set_title(
            f"C:{pred_clean[i]}\nT:{test_labels[i]}",
            color=("blue" if pred_clean[i] == test_labels[i] else "red"),
            fontsize=8
        )
        axes[0, i].axis("off")

        # Adversarial image
        axes[1, i].imshow(x_test_adv[i].reshape(28, 28), cmap="gray")
        axes[1, i].set_title(
            f"A:{pred_adv[i]}\nT:{test_labels[i]}",
            color=("blue" if pred_adv[i] == test_labels[i] else "red"),
            fontsize=8
        )
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Clean", fontsize=10)
    axes[1, 0].set_ylabel("Adv", fontsize=10)
    fig.suptitle("Clean Images vs Adversarial Images", fontsize=14)
    plt.tight_layout()

    # Show plot in Streamlit
    st.pyplot(fig)

# Main interface
if model:
    val = st.slider("Enter epsilon for FGSM ATTACK", min_value=0.0, max_value=2.0, step=0.01, help="Higher values = stronger attack = lower accuracy")
    
    if st.button("🚀 Run FGSM Attack", type="primary"):
        with st.spinner("⏳ Running FGSM attack... Please wait"):
            acc_clean, acc_adv, test_adv = fgsm(val)
            if acc_clean is not None and acc_adv is not None and test_adv is not None:
                st.write(f"**Epsilon:** {val}")
                st.write(f"**Accuracy on clean:** {acc_clean:.4f}")
                st.write(f"**Accuracy on adversarial:** {acc_adv:.4f}")
                comparephotos(test_adv)
else:
    st.error("Cannot proceed without a valid model.")

# Sidebar information
st.sidebar.markdown("""
### About FGSM Attack

The Fast Gradient Sign Method (FGSM) generates adversarial examples by:
1. Computing gradients of loss w.r.t. input
2. Taking the sign of gradients  
3. Adding small perturbation: x' = x + ε × sign(∇loss)

### Parameters
- **Epsilon (ε)**: Controls perturbation magnitude
- Larger ε → stronger attack → lower accuracy
- ε = 0 → no attack (original accuracy)
""")

st.sidebar.markdown("---")

# --- Project questions with answers ---
faq = {
    "What is the purpose of FGSM Attack?": "FGSM generates adversarial examples by adding small, calculated noise to fool the model.",
    "How does the model get affected by epsilon?": "The larger ε is, the stronger the attack, and the lower the accuracy on adversarial examples.",
    "What is the difference between accuracy on clean and adversarial data?": "Accuracy on clean = model performance on original data, accuracy on adversarial = performance after attack.",
}

# --- Initialize chat messages ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Chat interface in sidebar ---
st.sidebar.title("💬 Mini Chatbot")

# Select a question
selected_q = st.sidebar.selectbox("📋 Choose a question:", ["Select a question..."] + list(faq.keys()))

if st.sidebar.button("Send") and selected_q != "Select a question...":
    # Add user message
    st.session_state.messages.append({"role": "user", "content": selected_q})
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": faq[selected_q]})

# Display conversation
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.sidebar.markdown(f"🧑 **You:** {msg['content']}")
    else:
        st.sidebar.markdown(f"🤖 **Bot:** {msg['content']}")
