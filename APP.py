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

model = tf.keras.models.load_model("mnist_model.h5")
# Create ART KerasClassifier
classifier = KerasClassifier(model=model, clip_values=(0, 1))

def fgsm(epsi,subset_size=1000):
        # Option to use subset for better performance

            indices = np.random.choice(len(test_images), subset_size, replace=False)
            images_to_use = test_images[indices]
            labels_to_use = test_labels[indices]
            st.info(f"Using {subset_size} random samples from {len(test_images)} total images")


            # Generate adversarial examples
            attack = FastGradientMethod(estimator=classifier, eps=epsi)
            x_test_adv = attack.generate(x=images_to_use)

        # Evaluate on clean and adversarial examples
            loss_clean, accuracy_clean = model.evaluate(images_to_use, labels_to_use, verbose=0)
            loss_adv, accuracy_adv = model.evaluate(x_test_adv, labels_to_use, verbose=0)

            return accuracy_clean, accuracy_adv, x_test_adv
    




def comparephotos(x_test_adv):

    pred_clean = np.argmax(model.predict(test_images), axis=1)
    pred_adv = np.argmax(model.predict(x_test_adv), axis=1)


    correct_clean = np.sum(pred_clean == test_labels)
    correct_adv = np.sum(pred_adv == test_labels)



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





val = st.slider("Enter epsilon for FGSM ATTACK", min_value=0.0, max_value=2.0, step=0.01,  help="Higher values = stronger attack = lower accuracy")
if st.button("🚀 Run FGSM Attack", type="primary"):
    with st.spinner("⏳ Running FGSM attack... Please wait"):
      acc_clean, acc_adv, test_adv = fgsm(val)
      if (acc_clean, acc_adv, test_adv):
          col1, col2, col3 = st.columns(3)
          with col1:
              st.metric("Clean Accuracy", f"{acc_clean:.3f}", f"{acc_clean * 100:.1f}%")
          with col2:
              st.metric("Adversarial Accuracy", f"{acc_adv:.3f}", f"{acc_adv * 100:.1f}%")
          with col3:
              accuracy_drop = (acc_clean - acc_adv) * 100
              st.metric("Accuracy Drop", f"{accuracy_drop:.1f}%", f"-{accuracy_drop:.1f}%")

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
selected_q = st.sidebar.selectbox("📋 Choose a question:", ["", *faq.keys()])

if selected_q:
    st.session_state.messages.append({"role": "user", "content": selected_q})
    st.session_state.messages.append({"role": "assistant", "content": faq[selected_q]})


if st.sidebar.button("Send"):

        st.session_state.messages.append({"role": "user", "content": selected_q})

        st.session_state.messages.append({"role": "assistant", "content": faq[selected_q]})

# Display conversation
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.sidebar.markdown(f"🧑 **You:** {msg['content']}")
    else:
        st.sidebar.markdown(f"🤖 **Bot:** {msg['content']}")
