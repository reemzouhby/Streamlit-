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
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier

st.title("Effect of different Epsilon on accuracy of MNIST Dataset")

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess MNIST data"""
    from keras.datasets import mnist
    (_, _), (test_images, test_labels) = mnist.load_data()
    test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    return test_images, test_labels

@st.cache_resource
def create_model():
    """Create and train a CNN model on full MNIST dataset"""
    from keras.datasets import mnist
    
    # Load full training data
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Preprocess data
    train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Convert labels to categorical
    train_labels = keras.utils.to_categorical(train_labels, 10)
    test_labels = keras.utils.to_categorical(test_labels, 10)
    
    # Create model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train on full dataset
    st.info("Training model on full MNIST dataset... This will take several minutes.")
    progress_bar = st.progress(0)
    
    class ProgressCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            progress_bar.progress((epoch + 1) / 5)
    
    model.fit(train_images, train_labels, 
              epochs=5, batch_size=128, verbose=0,
              validation_data=(test_images, test_labels),
              callbacks=[ProgressCallback()])
    
    progress_bar.empty()
    st.success("Model training completed!")
    
    return model

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
    loss_clean, accuracy_clean = model.evaluate(test_subset, 
                                               keras.utils.to_categorical(labels_subset, 10), 
                                               verbose=0)
    loss_adv, accuracy_adv = model.evaluate(x_test_adv, 
                                           keras.utils.to_categorical(labels_subset, 10), 
                                           verbose=0)

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
            # --- Predictions on full dataset ---
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
            
            # Show accuracy by digit
            st.write("**Accuracy by Digit:**")
            digit_accuracy_clean = []
            digit_accuracy_adv = []
            
            for digit in range(10):
                mask = labels_subset == digit
                if np.sum(mask) > 0:
                    acc_clean = np.sum(pred_clean[mask] == digit) / np.sum(mask)
                    acc_adv = np.sum(pred_adv[mask] == digit) / np.sum(mask)
                    digit_accuracy_clean.append(acc_clean)
                    digit_accuracy_adv.append(acc_adv)
                else:
                    digit_accuracy_clean.append(0)
                    digit_accuracy_adv.append(0)
            
            # Create accuracy comparison chart
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(10)
            width = 0.35
            
            ax.bar(x - width/2, digit_accuracy_clean, width, label='Clean', alpha=0.8)
            ax.bar(x + width/2, digit_accuracy_adv, width, label='Adversarial', alpha=0.8)
            
            ax.set_xlabel('Digit')
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy by Digit: Clean vs Adversarial')
            ax.set_xticks(x)
            ax.set_xticklabels([str(i) for i in range(10)])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

        comparephotos(test_adv, test_subset, labels_subset)

st.sidebar.markdown("""
### About
This app demonstrates the Fast Gradient Sign Method (FGSM) adversarial attack on the **full MNIST dataset** (10,000 test samples).

- **Epsilon**: Controls the strength of the attack
- Higher epsilon = stronger attack = lower accuracy
- The model trains automatically on the full training dataset (60,000 samples)
- Results are computed on all 10,000 test samples

### Performance Note
- Model training: ~5-10 minutes (one time only, cached)
- Attack generation: ~2-5 minutes depending on epsilon
- Full dataset evaluation provides more accurate results
""")
