import streamlit as st
import os

from absl.testing.parameterized import parameters
from art.utils import to_categorical

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
from art.estimators.classification import KerasClassifier
from art.estimators.classification import KerasClassifier

(_, _), (test_images, test_labels) = mnist.load_data()

# Process only test data (much smaller memory footprint)
test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

model = tf.keras.models.load_model("mnist_model.h5")
classifier = KerasClassifier(model=model, clip_values=(0, 1))
from art.attacks.evasion import (
    FastGradientMethod,
    ProjectedGradientDescent,
    BasicIterativeMethod,
    DeepFool, CarliniL0Method, CarliniL2Method, CarliniLInfMethod
, ElasticNet, BoundaryAttack, NewtonFool, SaliencyMapMethod, HopSkipJump

)


# Time warning function
def show_time_warning(attack_type, parameters=None):
    """Display time warnings for different attacks"""
    time_warnings = {
        "FGSM": {
            "time": "⚡ Very Fast (~10-30 seconds)",
            "color": "success",
            "advice": "💡 **Tip**: This is the fastest attack - perfect for quick testing!"
        },
        "PGD": {
            "time": "🚀 Fast (~1-3 minutes)",
            "color": "success",
            "advice": "💡 **Tip**: Reduce max_iter to 20-30 for faster results with still good effectiveness!"
        },
        "BIM": {
            "time": "🚀 Fast (~1-3 minutes)",
            "color": "success",
            "advice": "💡 **Tip**: Keep max_iter around 10-20 for optimal speed vs effectiveness balance!"
        },
        "DeepFool": {
            "time": "🕐 Moderate (~5-10 minutes)",
            "color": "warning",
            "advice": "💡 **Tip**: Reduce max_iter to 20-30 for faster execution. Higher epsilon also speeds up convergence!"
        },
        "C&W": {
            "time": "⏰ Very Slow (~30 minutes - 2 hours)",
            "color": "error",
            "advice": "💡 **Tip**: Use max_iter=10-20 for testing. L∞ is faster than L2. Consider coffee break! ☕"
        },
        "Elastic Net": {
            "time": "⏰ Very Slow (~45 minutes - 2 hours)",
            "color": "error",
            "advice": "💡 **Tip**: Reduce max_iter to 10-20 and binary_search_steps to 5-7 for faster results!"
        },
        "Boundry Attack": {
            "time": "🕐 Moderate (~10-20 minutes)",
            "color": "warning",
            "advice": "💡 **Tip**: Keep max_iter under 100 and increase delta (0.02-0.05) for faster convergence!"
        },
        "NewtonFool": {
            "time": "🕐 Moderate (~5-15 minutes)",
            "color": "warning",
            "advice": "💡 **Tip**: Reduce max_iter to 20-40 for faster execution while maintaining effectiveness!"
        },
        "JSMA": {
            "time": "🕐 Moderate (~8-15 minutes)",
            "color": "warning",
            "advice": "💡 **Tip**: Use smaller gamma values (0.05-0.1) and batch_size=32 for faster processing!"
        },
        "Hopskijump": {
            "time": "🕐 Moderate (~10-25 minutes)",
            "color": "warning",
            "advice": "💡 **Tip**: Reduce max_eval to 2000-3000 and max_iter to 30-50 for faster results!"
        }
    }

    if attack_type in time_warnings:
        warning_info = time_warnings[attack_type]

        if warning_info["color"] == "error":
            st.error(f"⚠️ **{warning_info['time']}**")
        elif warning_info["color"] == "warning":
            st.warning(f"⚠️ **{warning_info['time']}**")
        else:
            st.success(f"✅ **{warning_info['time']}**")

        st.info(warning_info["advice"])

        # Additional specific warnings based on parameters
        if attack_type == "C&W" and parameters:
            if parameters.get("max_iter", 20) > 30:
                st.error("🚨 **Warning**: max_iter > 30 may take several hours! Consider reducing to 10-20.")
            if parameters.get("L_Type") == "L2":
                st.warning("📝 **Note**: L2 variant is slower than L∞. Consider L∞ for faster results.")

        elif attack_type == "Elastic Net" and parameters:
            if parameters.get("max_iter", 20) > 30 or parameters.get("binary_search_steps", 10) > 10:
                st.error("🚨 **Warning**: High iterations detected! This may take 2+ hours!")

        elif attack_type == "PGD" and parameters:
            if parameters.get("max_iter", 50) > 80:
                st.warning("📝 **Note**: max_iter > 80 may slow down the attack significantly.")

        # Show sample progress for slow attacks
        if warning_info["color"] == "error":
            st.info(
                "📊 **Progress Tracking**: The attack will show progress updates during execution. Feel free to grab a coffee! ☕")


def run_attacks(attack_type, parameters):
    subset_sizes = {
        "fast_attacks": 1000,  # FGSM, PGD, BIM
        "medium_attacks": 500,  # DeepFool, Boundry
        "slow_attacks": 20  # C&W, ElasticNet
    }
    if attack_type in ["FGSM", "PGD", "BIM"]:
        subset_size = subset_sizes["fast_attacks"]
    elif attack_type in ["DeepFool", "Boundry Attack", "NewtonFool", "JSMA", "Hopskijump"]:
        subset_size = subset_sizes["medium_attacks"]
    else:
        subset_size = subset_sizes["slow_attacks"]

    x_test_small = test_images[:subset_size]
    y_test_small = test_labels[:subset_size]
    if attack_type == "FGSM":
        attack = FastGradientMethod(estimator=classifier, targeted=parameters.get('targeted', False),
                                    eps=parameters["eps"])
        x_test_adv = attack.generate(x=test_images)
    elif attack_type == "PGD":
        attack = ProjectedGradientDescent(estimator=classifier, eps_step=0.03, eps=parameters["eps"],
                                          max_iter=parameters["max_iter"],
                                          random_eps=parameters["random_eps"],
                                          targeted=parameters.get('targeted', False), batch_size=100)
    elif attack_type == "BIM":
        attack = BasicIterativeMethod(estimator=classifier, targeted=parameters.get('targeted', False), eps_step=0.03,
                                      eps=parameters["eps"], max_iter=10)  # eps  small
    elif attack_type == "DeepFool":

        attack = DeepFool(classifier, max_iter=parameters["max_iter"], epsilon=parameters["eps"], batch_size=100)
    elif attack_type == "C&W":
        if parameters["L_Type"] == "L2":
            attack = CarliniL2Method(classifier, max_iter=1000, targeted=parameters.get('targeted', False),
                                     learning_rate=parameters["learning_rate"], confidence=parameters["confidence"],
                                     batch_size=100)
        elif parameters["L_Type"] == "L∞":
            attack = CarliniLInfMethod(classifier, targeted=parameters.get('targeted', False),
                                       max_iter=parameters["max_iter"], batch_size=100,
                                       confidence=parameters["confidence"])
        else:
            attack = CarliniL0Method(classifier, targeted=parameters.get('targeted', False),
                                     max_iter=parameters["max_iter"], confidence=parameters["confidence"],
                                     learning_rate=parameters["learning_rate"])
    elif attack_type == "Elastic Net":
        attack = ElasticNet(classifier=classifier, confidence=parameters["confidence"],
                            targeted=parameters.get('targeted', False), learning_rate=0.01, binary_search_steps=9,
                            max_iter=parameters["max_iter"], beta=0.001, initial_const=0.001, batch_size=1,
                            decision_rule=parameters["decision_rule"], verbose=True)  # eps  small
    elif attack_type == "Boundry Attack":
        attack = BoundaryAttack(estimator=classifier,
                                targeted=parameters.get('targeted', False),
                                max_iter=parameters["max_iter"],  # iterations
                                delta=parameters["delta"],
                                epsilon=parameters["eps"],
                                verbose=True)
    elif attack_type == "JSMA":
        attack = SaliencyMapMethod(
            classifier=classifier,
            theta=parameters["theta"],
            gamma=parameters["gamma"],
            batch_size=parameters["batch_size"],
            verbose=True
        )


    elif attack_type == "NewtonFool":
        attack = NewtonFool(
            classifier=classifier,
            max_iter=parameters["max_iter"],
            batch_size=parameters["batch_size"]
        )
        x_adv = attack.generate(x=x_test_small)

    elif attack_type == "HopSkipJump":
        norm_map = {"L2": 2, "L∞": np.inf, "L1": 1}
        selected_norm = norm_map[parameters["norm"]]
        attack = HopSkipJump(
            classifier=classifier,
            max_iter=parameters["max_iter"],
            max_eval=parameters["max_eval"],
            norm=selected_norm,
            init_eval=parameters["init_eval"],
            batch_size=parameters["batch_size"],
            targeted=parameters.get("targeted", False)
        )

    if parameters.get("targeted", False):
        target_class = parameters.get("target_class", 5)
        target_labels = np.full((subset_size,), target_class)
        target_labels_one_hot = to_categorical(target_labels, nb_classes=10)
        x_adv = attack.generate(x=x_test_small, y=target_labels_one_hot)
    else:
        x_adv = attack.generate(x=x_test_small)
    loss_clean, acc_clean = model.evaluate(x_test_small, y_test_small, verbose=0)
    loss_adv, acc_adv = model.evaluate(x_adv, y_test_small, verbose=0)
    pred_clean = np.argmax(model.predict(x_test_small), axis=1)
    pred_adv = np.argmax(model.predict(x_adv), axis=1)

    correct_clean = np.sum(pred_clean == y_test_small)
    correct_adv = np.sum(pred_adv == y_test_small)
    if (acc_clean, acc_adv):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Clean Accuracy", f"{acc_clean:.3f}", f"{acc_clean * 100:.1f}%")
        with col2:
            st.metric("Adversarial Accuracy", f"{acc_adv:.3f}", f"{acc_adv * 100:.1f}%")
        with col3:
            accuracy_drop = (acc_clean - acc_adv) * 100
            st.metric("Accuracy Drop", f"{accuracy_drop:.1f}%", f"-{accuracy_drop:.1f}%")

    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    for i in range(10):
        # Clean image - FIXED: Use x_test_small[i] instead of y_test_small[i]
        axes[0, i].imshow(x_test_small[i].reshape(28, 28), cmap="gray")
        axes[0, i].set_title(
            f"C:{pred_clean[i]}\nT:{y_test_small[i]}",
            color=("blue" if pred_clean[i] == y_test_small[i] else "red"),
            fontsize=8
        )
        axes[0, i].axis("off")

        # Adversarial image
        axes[1, i].imshow(x_adv[i].reshape(28, 28), cmap="gray")
        axes[1, i].set_title(
            f"A:{pred_adv[i]}\nT:{y_test_small[i]}",
            color=("blue" if pred_adv[i] == y_test_small[i] else "red"),
            fontsize=8
        )
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Clean", fontsize=10)
    axes[1, 0].set_ylabel("Adv", fontsize=10)
    fig.suptitle("Clean Images vs Adversarial Images", fontsize=14)
    plt.tight_layout()

    # Show plot in Streamlit
    st.pyplot(fig)

    return acc_clean, acc_adv, x_adv


st.set_page_config(
    page_title="Adversarial Attacks Demo",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Adversarial Attacks on MNIST")
st.markdown("---")

st.sidebar.header("⚔️ Attack Configuration")
# first ask about taget and untarget
attack_mode = st.sidebar.radio(
    "Attack Mode:",
    ["Untarget", "Target"],
    help="Untargeted: Try to misclassify to any wrong class\nTargeted: Try to misclassify to a specific target class"
)
# after select if white box or black
# after select the mode now move to the type of the ATTAcks
# if target --> fgsm , pgd , BIM
# Untarget --> fgsm ,pgd, deepfool, bim
attack_mode_type = st.sidebar.radio("Attack Type:", ["White-box", "Black-box"])
if attack_mode_type == "White-box":
    available_attacks = ["FGSM", "PGD", "BIM", "DeepFool", "C&W", "Elastic Net", "NewtonFool", "JSMA"]
else:
    available_attacks = ["Boundary Attack", "Hopskijump"]

if (attack_mode == "Target"):
    if attack_mode_type == "White-box":
        available_attacks = ["FGSM", "PGD", "BIM", "C&W", "Elastic Net", "JSMA", "NewtonFool"]
    else:
        available_attacks = ["Boundry Attack", "Hopskijump"]
    st.sidebar.info("Target attacks try to fool the model into a specific target class")
else:
    if attack_mode_type == "White-box":
        available_attacks = ["FGSM", "PGD", "BIM", "DeepFool", "C&W", "Elastic Net", "NewtonFool"]
    else:
        available_attacks = ["Boundry Attack", "Hopskijump"]

    st.sidebar.info(" Untarget attacks try to fool the model into any incorrect prediction")
attack_type = st.sidebar.selectbox("Select the type of ATtack ", options=available_attacks)

# choose the attak parameters if target choose frst the nb  that i want to make missaclassiification
st.sidebar.subheader(attack_type + "Parameters")
# now if target show a select box of nb of target nb
parameters = {}
if (attack_mode == "Target"):
    parameters["Targeted"] = True
    parameters["Target_class"] = st.sidebar.selectbox("Choose the nb ", options=range(10),
                                                      help="The class you want the model to misclassify images as"
                                                      )
if (attack_type == "FGSM"):
    # parameters nly epsi
    parameters["eps"] = st.sidebar.slider("Enter epsilon for FGSM ATTACK", min_value=0.0, max_value=2.0, step=0.01,
                                          help="Higher values = stronger attack = lower accuracy")
if (attack_type == "PGD"):
    # for pgd  eps , max iteration , random eps
    parameters["eps"] = st.sidebar.slider("Enter epsilon for PGD ATTACK", min_value=0.0, max_value=2.0, step=0.01,
                                          help="Higher values = stronger attack = lower accuracy")
    parameters['random_eps'] = st.sidebar.checkbox("Random Epsilon", True,
                                                   help="Start with random perturbation")
    parameters["max_iter"] = st.sidebar.slider("Max Iterations", 10, 100, 50, 10,
                                               help=" larger iteration → stronger attacks , but can be slow .")
if (attack_type == "BIM"):
    parameters["eps"] = st.sidebar.slider("Enter epsilon for BIM ATTACK", min_value=0.0, max_value=2.0, step=0.01,
                                          help="Higher values = stronger attack = lower accuracy")
    parameters["max_iter"] = st.sidebar.slider("Max Iterations", 10, 100, 50, 10,
                                               help=" larger iteration → stronger attacks , but can be slow .")

if (attack_type == "DeepFool"):
    parameters["eps"] = st.sidebar.slider("Enter epsilon for DeepFool ATTACK", min_value=0.0000, max_value=0.001,
                                          step=0.0001, format="%.4f",
                                          help="but a small overshoot factor to make sure the perturbed image crosses the decision boundary")
    parameters["max_iter"] = st.sidebar.slider("Max Iterations", 10, 100, 50, 10,
                                               help=" larger iteration → stronger attacks , but can be slow .")

if (attack_type == "C&W"):
    parameters["L_Type"] = st.sidebar.selectbox("Enter L0, l2 or Linfinite", options=["L0", "L2", "L∞"])
    if parameters["L_Type"] == "L0":
        parameters["learning_rate"] = st.sidebar.slider(
            "Learning Rate",
            min_value=0.01, max_value=1.0, value=0.01, step=0.01,
            help="Learning rate for the attack optimization"
        )
        parameters["confidence"] = st.sidebar.slider(
            "Confidence",
            min_value=0.0, max_value=50.0, value=0.0, step=1.0,
            help="Confidence parameter - higher values make attack stronger"
        )
    parameters["max_iter"] = st.sidebar.slider("Max Iterations", 10, 50, 20, 10,
                                               help=" larger iteration → stronger attacks , but can be slow .")

    if (parameters["L_Type"] == "L2"):
        parameters["learning_rate"] = st.sidebar.slider(
            "Learning Rate",
            min_value=0.001, max_value=0.1, value=0.01, step=0.001,
            help="Learning rate for the attack optimization"
        )

        parameters["confidence"] = st.sidebar.slider(
            "Confidence",
            min_value=0.0, max_value=50.0, value=0.0, step=1.0,
            help="Confidence parameter - higher values make attack stronger"
        )
        parameters["binary_search_steps"] = st.sidebar.slider(
            "binary_search_steps",
            min_value=5.0, max_value=20.0, value=10.0, step=1.0,
            help=" Better c tuning → stronger attacks."
        )
        parameters["max_iter"] = st.sidebar.slider("Max Iterations", 10, 50, 20, 10,
                                                   help=" larger iteration → stronger attacks , but can be slow .")

    if parameters["L_Type"] == "L∞":
        parameters["learning_rate"] = st.sidebar.slider(
            "Learning Rate",
            min_value=0.01, max_value=0.1, value=0.01, step=0.01,
            help="Learning rate for the attack optimization"
        )
        parameters["max_iter"] = st.sidebar.slider("Max Iterations", 10, 50, 20, 10,
                                                   help=" larger iteration → stronger attacks , but can be slow .")

        parameters["confidence"] = st.sidebar.slider(
            "Confidence",
            min_value=0.0, max_value=50.0, value=0.0, step=1.0,
            help="Confidence parameter - higher values make attack stronger"
        )
if attack_type == "Elastic Net":
    parameters["learning_rate"] = st.sidebar.slider(
        "Learning Rate",
        min_value=0.001, max_value=0.1, value=0.01, step=0.001,
        help="Learning rate for the attack optimization"
    )

    parameters["confidence"] = st.sidebar.slider(
        "Confidence",
        min_value=0.0, max_value=50.0, value=0.0, step=1.0,
        help="Confidence parameter - higher values make attack stronger"
    )
    parameters["binary_search_steps"] = st.sidebar.slider(
        "binary_search_steps",
        min_value=5.0, max_value=20.0, value=10.0, step=1.0,
        help=" Better c tuning → stronger attacks."
    )
    parameters["decision_rule"] = st.sidebar.selectbox("Choose the Decision Rules ", options=["L1", "L2", "EN"],
                                                       index=2)
if attack_type == "Boundry Attack":
    parameters["max_iter"] = st.sidebar.slider(
        "Max Iterations",
        10, 200, 50, 10,
        help="Number of iterations for the attack. Higher values = stronger attack but longer runtime."
    )
    parameters["delta"] = st.sidebar.slider(
        "Delta",
        0.001, 0.05, 0.01, 0.001,
        help="Small step to adjust images during the search for minimal difference between original and adversarial image."
    )
    parameters["eps"] = st.sidebar.slider(
        "Epsilon",
        0.05, 0.2, 0.1, 0.01,
        help="Maximum noise per step. Higher value = bigger perturbation to the image."
    )
if attack_type == "JSMA":
    parameters["theta"] = st.sidebar.slider("Theta (pixel change per step)", 0.01, 0.5, 0.1, 0.01)
    parameters["gamma"] = st.sidebar.slider("Gamma (max fraction of pixels to change)", 0.01, 0.5, 0.1, 0.01)
    parameters["batch_size"] = st.sidebar.slider("Batch size", 1, 128, 64, 1)

if attack_type == "NewtonFool":
    parameters["max_iter"] = st.sidebar.slider("Max Iterations", 10, 100, 50, 10)
    parameters["batch_size"] = st.sidebar.slider("Batch size", 1, 128, 64, 1)

if attack_type == "Hopskijump":
    parameters["max_iter"] = st.sidebar.slider("Max Iterations", 10, 100, 50, 10)
    parameters["max_eval"] = st.sidebar.slider("Max Evaluations", 1000, 20000, 5000, 100)
    parameters["init_eval"] = st.sidebar.slider("Initial Evaluations", 10, 500, 50, 10)
    parameters["batch_size"] = st.sidebar.slider("Batch size", 1, 128, 64, 1)
    parameters["norm"] = st.sidebar.selectbox("Choose Norm for HopSkipJump", options=["L2", "L∞", "L1"])

# Show time warning before the run button
st.markdown("### ⏱️ Expected Execution Time")
show_time_warning(attack_type, parameters)

if st.button("🚀 Run " + attack_type + "  " + "Attack", type="primary"):
    with st.spinner("⏳ Running " + attack_type + "  " + " attack... Please wait"):
        run_attacks(attack_type, parameters)
