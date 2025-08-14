import os
# Disable GPU completely before importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
import streamlit as st
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.datasets import mnist
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier

st.title("Effect of different Epsilon on accuracy of MNIST Dataset")

@st.cache_data
def load_and_preprocess_data():
  
    (_, _), (test_images, test_labels) = mnist.load_data()
    # Normalize and add channel dimension
    test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    return test_images, test_labels

@st.cache_resource
def create_model():
    
    
    # Load data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    # Add channel dimension for grayscale images
    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    FULLY_CONNECT_NUM = 128
    batch_size = 128
    NUM_CLASSES = len(class_names)
    
    # Create model with your architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    # Train with fewer epochs (3 instead of 10)
    st.info("Training model on full MNIST dataset... This will take a few minutes.")
    progress_bar = st.progress(0)
    
    class ProgressCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress_bar.progress((epoch + 1) / 3)  # Updated for 3 epochs
    
    history = model.fit(train_images, train_labels,
                       epochs=3,  # Reduced from 10 to 3 epochs
                       batch_size=128,
                       validation_data=(test_images, test_labels),
                       verbose=0,
                       callbacks=[ProgressCallback()])
    
    progress_bar.empty()
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    st.success(f"Model training completed! Test accuracy: {test_acc*100:.2f}%")
    

# Load data and model
test_images, test_labels = load_and_preprocess_data()
model = create_model()

# Create ART KerasClassifier
classifier = KerasClassifier(model=model, clip_values=(0, 1))

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def fgsm(epsi):
    """Perform FGSM attack on full test dataset"""
    # Use full test dataset
    test_subset = test_images
    labels_subset = test_labels
    
    # Generate adversarial examples
    attack = FastGradientMethod(estimator=classifier, eps=epsi)
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    x_test_adv = []
    
    progress_bar = st.progress(0)
    st.write("Generating adversarial examples...")
    
    for i in range(0, len(test_subset), batch_size):
        batch = test_subset[i:i+batch_size]
        batch_adv = attack.generate(x=batch)
        x_test_adv.append(batch_adv)
        progress_bar.progress((i + batch_size) / len(test_subset))
    
    x_test_adv = np.vstack(x_test_adv)
    progress_bar.empty()

    # Evaluate on clean and adversarial examples
    st.write("Evaluating model performance...")
    loss_clean, accuracy_clean = model.evaluate(test_subset, labels_subset, verbose=0)
    loss_adv, accuracy_adv = model.evaluate(x_test_adv, labels_subset, verbose=0)

    return accuracy_clean, accuracy_adv, x_test_adv, test_subset, labels_subset

# Streamlit interface
val = st.slider("Enter epsilon for FGSM ATTACK", min_value=0.0, max_value=2.0, step=0.01, value=0.1)

if st.button("Run Attack"):
    with st.spinner("Running FGSM attack..."):
        acc_clean, acc_adv, test_adv, test_subset, labels_subset = fgsm(val)
        
        st.write(f"**Epsilon:** {val}")
        st.write(f"**Accuracy on clean:** {acc_clean:.4f}")
        st.write(f"**Accuracy on adversarial:** {acc_adv:.4f}")
        
        def comparephotos(x_test_adv, test_subset, labels_subset):
     
            st.write("Computing predictions on full dataset...")
            
            # Process predictions in batches
            batch_size = 1000
            pred_clean = []
            pred_adv = []
            
            for i in range(0, len(test_subset), batch_size):
                batch_clean = test_subset[i:i+batch_size]
                batch_adv = x_test_adv[i:i+batch_size]
                
                pred_clean.extend(np.argmax(model.predict(batch_clean, verbose=0), axis=1))
                pred_adv.extend(np.argmax(model.predict(batch_adv, verbose=0), axis=1))
            
            pred_clean = np.array(pred_clean)
            pred_adv = np.array(pred_adv)

            # --- Accuracy counts on full dataset ---
            correct_clean = np.sum(pred_clean == labels_subset)
            correct_adv = np.sum(pred_adv == labels_subset)
            total_samples = len(labels_subset)

            st.write(f"📊 **Full Dataset Results (Total: {total_samples} samples)**")
            st.write(f"✅ Correct predictions (clean): {correct_clean} ({100*correct_clean/total_samples:.2f}%)")
            st.write(f"⚠️ Correct predictions (adv): {correct_adv} ({100*correct_adv/total_samples:.2f}%)")
            st.write(f"❌ Wrong predictions (clean): {total_samples - correct_clean} ({100*(total_samples-correct_clean)/total_samples:.2f}%)")
            st.write(f"❌ Wrong predictions (adv): {total_samples - correct_adv} ({100*(total_samples-correct_adv)/total_samples:.2f}%)")
            st.write(f"📉 **Accuracy drop**: {100*(correct_clean-correct_adv)/total_samples:.2f}%")

            # --- Visual comparison (first 10 samples only for display) ---
            st.write("**Sample Images Comparison (First 10):**")
            fig, axes = plt.subplots(2, 10, figsize=(15, 4))
            for i in range(10):
                # Clean image
                axes[0, i].imshow(test_subset[i].reshape(28, 28), cmap="gray")
                axes[0, i].set_title(
                    f"C:{pred_clean[i]}\nT:{labels_subset[i]}",
                    color=("blue" if pred_clean[i] == labels_subset[i] else "red"),
                    fontsize=8
                )
                axes[0, i].axis("off")

                # Adversarial image
                axes[1, i].imshow(x_test_adv[i].reshape(28, 28), cmap="gray")
                axes[1, i].set_title(
                    f"A:{pred_adv[i]}\nT:{labels_subset[i]}",
                    color=("blue" if pred_adv[i] == labels_subset[i] else "red"),
                    fontsize=8
                )
                axes[1, i].axis("off")

            axes[0, 0].set_ylabel("Clean", fontsize=10)
            axes[1, 0].set_ylabel("Adversarial", fontsize=10)
            fig.suptitle("Clean Images vs Adversarial Images (Sample)", fontsize=14)
            plt.tight_layout()

            # Show plot in Streamlit
            st.pyplot(fig)
            
         
        comparephotos(test_adv, test_subset, labels_subset)

st.sidebar.markdown("""
### About
This app demonstrates the Fast Gradient Sign Method (FGSM) adversarial attack on the **full MNIST dataset** (10,000 test samples).

- **Epsilon**: Controls the strength of the attack
- Higher epsilon = stronger attack = lower accuracy
- The model trains automatically on the full training dataset (60,000 samples)
- Results are computed on all 10,000 test samples
- Uses improved CNN architecture with padding='same'

### Performance Note
- Model training: ~3-5 minutes (3 epochs, one time only, cached)
- Attack generation: ~2-5 minutes depending on epsilon
- Full dataset evaluation provides more accurate results

### Model Architecture
- Conv2D layers with 'same' padding
- Two dense layers (128 and 64 neurons)
- SparseCategoricalCrossentropy loss (no need to convert labels)
""")

