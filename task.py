import os
# Disable GPU completely before importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
import streamlit as st

st.title("Effect of different Epsilon on accuracy of Mnist Dataset")
from keras.datasets import mnist
import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

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


def fgsm(epsi):
    # Generate adversarial examples
    attack = FastGradientMethod(estimator=classifier, eps=epsi)
    x_test_adv = attack.generate(x=test_images)

    # Evaluate on clean and adversarial examples
    loss_clean, accuracy_clean = model.evaluate(test_images, test_labels, verbose=0)
    loss_adv, accuracy_adv = model.evaluate(x_test_adv, test_labels, verbose=0)

    return accuracy_clean, accuracy_adv, x_test_adv


val = st.slider("Enter epsilon for FGSM ATTACK", min_value=0.0, max_value=2.0, step=0.01)
acc_clean, acc_adv, test_adv = fgsm(val)
st.write(f"**Epsilon:** {val}")
st.write(f"**Accuracy on clean:** {acc_clean}")
st.write(f"**Accuracy on adversarial:** {acc_adv}")


def comparephotos(x_test_adv):
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



comparephotos(test_adv)
