"""
Streamlit Lab - Modified Version
Original Lab: Iris Flower Prediction with FastAPI backend
Modifications:
    1. Different dataset: Wine Quality dataset instead of Iris
    2. Different model: Random Forest + Gradient Boosting comparison
    3. Self-contained: Model trains within the app (no FastAPI dependency)
    4. Added visualizations: Confusion matrix, feature importance, PCA
    5. Added sidebar hyperparameter tuning controls
    6. Added model comparison functionality
    7. Added data exploration page
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Wine Quality Classifier - MLOps Lab",
    page_icon="🍷",
    layout="wide"
)

# ============================================================
# Load and Cache Data
# ============================================================
@st.cache_data
def load_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    df['target_name'] = df['target'].map(
        {i: name for i, name in enumerate(wine.target_names)}
    )
    logger.info(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    return df, wine

# ============================================================
# Train Model (Cached)
# ============================================================
@st.cache_resource
def train_model(model_type, n_estimators, max_depth, test_size, random_state):
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            random_state=random_state
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else 3,
            random_state=random_state
        )

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=wine.target_names, output_dict=True)

    logger.info(f"Model trained: {model_type}, Accuracy: {accuracy:.4f}")

    return model, scaler, accuracy, f1, cm, report, X_test_scaled, y_test, y_pred, wine.feature_names

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.title("🍷 Wine Classifier")
    st.markdown("---")

    page = st.radio("Navigate", ["🏠 Home", "📊 Data Explorer", "🤖 Model Training", "🔮 Predict"])

    st.markdown("---")
    st.info("**Modified Lab**\nOriginal: Iris + FastAPI\nModified: Wine + Multi-model + Visualizations")

    st.markdown("### ⚙️ Model Config")
    model_type = st.selectbox("Model", ["Random Forest", "Gradient Boosting"])
    n_estimators = st.slider("Number of Estimators", 10, 300, 100, 10)
    max_depth = st.slider("Max Depth (0 = None)", 0, 20, 0)
    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", 0, 100, 42)

    st.markdown("---")
    st.success("App Status: Online ✅")

# ============================================================
# Load Data & Train Model
# ============================================================
df, wine_data = load_data()

model, scaler, accuracy, f1, cm, report, X_test, y_test, y_pred, feature_names = train_model(
    model_type, n_estimators, max_depth, test_size, random_state
)

# ============================================================
# Page: Home
# ============================================================
if page == "🏠 Home":
    st.write("# 🍷 Wine Quality Classifier")
    st.write("### MLOps Streamlit Lab - Modified Version")

    st.markdown("""
    **Modifications from Original Lab (Iris + FastAPI):**
    - **Different Dataset**: Wine Quality dataset (3 classes, 13 features)
    - **Different Models**: Random Forest & Gradient Boosting with hyperparameter tuning
    - **Self-contained**: No FastAPI backend dependency
    - **Added Visualizations**: Confusion matrix, feature importance, PCA scatter
    - **Data Explorer**: Interactive EDA page
    - **Live Prediction**: Manual input via sliders
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Accuracy", f"{accuracy:.2%}")
    col2.metric("📊 F1 Score", f"{f1:.2%}")
    col3.metric("📁 Samples", f"{len(df)}")
    col4.metric("🔢 Features", f"{len(feature_names)}")

    st.markdown("---")
    st.write("### Dataset Overview")
    st.dataframe(df.head(10), use_container_width=True)

# ============================================================
# Page: Data Explorer
# ============================================================
elif page == "📊 Data Explorer":
    st.write("# 📊 Data Explorer")

    tab1, tab2, tab3 = st.tabs(["Distribution", "Correlation", "PCA"])

    with tab1:
        st.write("### Feature Distributions by Wine Class")
        selected_feature = st.selectbox("Select Feature", feature_names)
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, name in enumerate(wine_data.target_names):
            subset = df[df['target'] == i]
            ax.hist(subset[selected_feature], alpha=0.6, label=name, bins=20)
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)

    with tab2:
        st.write("### Correlation Matrix")
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = df[list(feature_names)].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        st.write("### PCA Visualization (2D)")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(StandardScaler().fit_transform(wine_data.data))
        fig, ax = plt.subplots(figsize=(10, 7))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=wine_data.target,
                            cmap='viridis', alpha=0.7, edgecolors='k', s=50)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.colorbar(scatter, ax=ax, label="Wine Class")
        st.pyplot(fig)

# ============================================================
# Page: Model Training
# ============================================================
elif page == "🤖 Model Training":
    st.write(f"# 🤖 Model Training — {model_type}")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=wine_data.target_names,
                    yticklabels=wine_data.target_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col2:
        st.write("### Feature Importance")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(
            [feature_names[i] for i in indices],
            importances[indices],
            color='steelblue'
        )
        ax.set_xlabel("Importance")
        ax.invert_yaxis()
        st.pyplot(fig)

    st.markdown("---")
    st.write("### Classification Report")
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

# ============================================================
# Page: Predict
# ============================================================
elif page == "🔮 Predict":
    st.write("# 🔮 Make a Prediction")

    input_method = st.radio("Input Method", ["Sliders", "JSON Upload"])

    if input_method == "Sliders":
        st.write("### Adjust Features")
        col1, col2 = st.columns(2)
        input_values = []

        for i, fname in enumerate(feature_names):
            min_val = float(df[fname].min())
            max_val = float(df[fname].max())
            mean_val = float(df[fname].mean())
            step = (max_val - min_val) / 100

            if i < len(feature_names) // 2:
                with col1:
                    val = st.slider(fname, min_val, max_val, mean_val, step, format="%.2f")
            else:
                with col2:
                    val = st.slider(fname, min_val, max_val, mean_val, step, format="%.2f")
            input_values.append(val)

        predict_button = st.button("🔮 Predict Wine Class", use_container_width=True)

        if predict_button:
            input_array = np.array(input_values).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]

            wine_class = wine_data.target_names[prediction]
            st.success(f"🍷 The predicted wine class is: **{wine_class}**")

            st.write("### Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Wine Class': wine_data.target_names,
                'Probability': probabilities
            })
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#2ecc71' if i == prediction else '#3498db' for i in range(3)]
            ax.bar(prob_df['Wine Class'], prob_df['Probability'], color=colors)
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            for i, (cls, prob) in enumerate(zip(prob_df['Wine Class'], prob_df['Probability'])):
                ax.text(i, prob + 0.02, f'{prob:.2%}', ha='center', fontweight='bold')
            st.pyplot(fig)

    else:
        st.write("### Upload JSON File")
        st.code(json.dumps({
            "input_test": {fname: round(float(df[fname].mean()), 2) for fname in feature_names}
        }, indent=2), language="json")

        test_input_file = st.file_uploader("Upload test prediction file", type=['json'])

        if test_input_file:
            st.write("**Preview:**")
            test_input_data = json.load(test_input_file)
            st.json(test_input_data)

            predict_json_button = st.button("🔮 Predict from JSON", use_container_width=True)

            if predict_json_button:
                try:
                    values = list(test_input_data["input_test"].values())
                    input_array = np.array(values).reshape(1, -1)
                    input_scaled = scaler.transform(input_array)
                    prediction = model.predict(input_scaled)[0]
                    wine_class = wine_data.target_names[prediction]
                    st.success(f"🍷 The predicted wine class is: **{wine_class}**")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")