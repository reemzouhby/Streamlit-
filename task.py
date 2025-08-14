import os
# Disable GPU completely before importing TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
import streamlit as st
import tensorflow as tf
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.datasets import mnist
import matplotlib.pyplot as plt
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Import ART with error handling
try:
    from art.attacks.evasion import FastGradientMethod
    from art.estimators.classification import KerasClassifier
    ART_AVAILABLE = True
except ImportError:
    st.error("ART library not available. Please install adversarial-robustness-toolbox")
    ART_AVAILABLE = False

st.title("FGSM Attack on MNIST Dataset")

if not ART_AVAILABLE:
    st.stop()

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess MNIST test data"""
    try:
        (_, _), (test_images, test_labels) = mnist.load_data()
        # Normalize and add channel dimension
        test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        return test_images, test_labels
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_resource
def create_model():
    """Create and train CNN model"""
    try:
        with st.spinner("Loading MNIST data..."):
            # Load data
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
            
            # Normalize and reshape
            train_images = train_images.astype('float32') / 255.0
            test_images = test_images.astype('float32') / 255.0
            train_images = train_images.reshape(-1, 28, 28, 1)
            test_images = test_images.reshape(-1, 28, 28, 1)
        
        st.info("Creating and training model...")
        
        # Create model
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
        
        # Train with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        class ProgressCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = (epoch + 1) / 3
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch + 1}/3 - Loss: {logs.get('loss', 0):.4f} - Acc: {logs.get('accuracy', 0):.4f}")
        
        # Train model
        history = model.fit(
            train_images, train_labels,
            epochs=3,
            batch_size=128,
            validation_data=(test_images, test_labels),
            verbose=0,
            callbacks=[ProgressCallback()]
        )
        
        progress_bar.empty()
        status_text.empty()
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
        st.success(f"✅ Model training completed! Test accuracy: {test_acc*100:.2f}%")
        
        return model
        
    except Exception as e:
        st.error(f"Error creating model: {e}")
        return None

# Initialize
data_load_state = st.text("Loading data and model...")
test_images, test_labels = load_and_preprocess_data()

if test_images is None:
    st.stop()

model = create_model()
data_load_state.text("")

if model is None:
    st.stop()

# Create ART classifier
try:
    classifier = KerasClassifier(model=model, clip_values=(0, 1))
except Exception as e:
    st.error(f"Error creating ART classifier: {e}")
    st.stop()

def run_fgsm_attack(epsilon):
    """Run FGSM attack with given epsilon"""
    try:
        st.info(f"Running FGSM attack with epsilon = {epsilon}")
        
        # Create attack
        attack = FastGradientMethod(estimator=classifier, eps=epsilon)
        
        # Use subset for faster processing
        subset_size = min(1000, len(test_images))  # Use max 1000 samples
        test_subset = test_images[:subset_size]
        labels_subset = test_labels[:subset_size]
        
        # Generate adversarial examples
        with st.spinner("Generating adversarial examples..."):
            x_test_adv = attack.generate(x=test_subset)
        
        # Evaluate performance
        st.info("Evaluating model performance...")
        
        # Get predictions
        pred_clean = np.argmax(model.predict(test_subset, verbose=0), axis=1)
        pred_adv = np.argmax(model.predict(x_test_adv, verbose=0), axis=1)
        
        # Calculate accuracies
        acc_clean = np.mean(pred_clean == labels_subset)
        acc_adv = np.mean(pred_adv == labels_subset)
        
        return acc_clean, acc_adv, x_test_adv, test_subset, labels_subset, pred_clean, pred_adv
        
    except Exception as e:
        st.error(f"Error running attack: {e}")
        return None

def display_results(results):
    """Display attack results"""
    if results is None:
        return
        
    acc_clean, acc_adv, x_test_adv, test_subset, labels_subset, pred_clean, pred_adv = results
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Clean Accuracy", f"{acc_clean:.3f}", f"{acc_clean*100:.1f}%")
    with col2:
        st.metric("Adversarial Accuracy", f"{acc_adv:.3f}", f"{acc_adv*100:.1f}%")
    with col3:
        accuracy_drop = (acc_clean - acc_adv) * 100
        st.metric("Accuracy Drop", f"{accuracy_drop:.1f}%", f"-{accuracy_drop:.1f}%")
    
    # Sample comparison
    st.subheader("Sample Comparison (First 10 Images)")
    
    try:
        fig, axes = plt.subplots(2, 10, figsize=(15, 4))
        
        for i in range(10):
            if i >= len(test_subset):
                break
                
            # Clean image
            axes[0, i].imshow(test_subset[i].reshape(28, 28), cmap="gray")
            axes[0, i].set_title(
                f"P:{pred_clean[i]}\nT:{labels_subset[i]}", 
                color=("green" if pred_clean[i] == labels_subset[i] else "red"),
                fontsize=8
            )
            axes[0, i].axis("off")
            
            # Adversarial image
            axes[1, i].imshow(x_test_adv[i].reshape(28, 28), cmap="gray")
            axes[1, i].set_title(
                f"P:{pred_adv[i]}\nT:{labels_subset[i]}", 
                color=("green" if pred_adv[i] == labels_subset[i] else "red"),
                fontsize=8
            )
            axes[1, i].axis("off")
        
        axes[0, 0].set_ylabel("Clean", fontsize=10, rotation=0, labelpad=20)
        axes[1, 0].set_ylabel("Adversarial", fontsize=10, rotation=0, labelpad=20)
        fig.suptitle("Clean vs Adversarial Images", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    except Exception as e:
        st.error(f"Error creating visualization: {e}")

# Streamlit UI
st.subheader("Attack Configuration")
epsilon = st.slider(
    "Epsilon (Attack Strength)", 
    min_value=0.0, 
    max_value=0.5, 
    step=0.01, 
    value=0.1,
    help="Higher values = stronger attack = lower accuracy"
)

if st.button("🚀 Run FGSM Attack", type="primary"):
    results = run_fgsm_attack(epsilon)
    if results:
        st.subheader("Attack Results")
        display_results(results)

# Sidebar info
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

### Model Info
- CNN with 3 Conv2D layers + 2 Dense layers
- Trained for 3 epochs on MNIST
- Uses CPU only (GPU disabled)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("*Built with Streamlit & TensorFlow*")
