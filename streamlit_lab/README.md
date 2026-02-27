# Streamlit Lab - Wine Quality Classifier 🍷

## Based On
Original Streamlit Lab-1 from [Prof. Ramin Mohammadi's MLOps Repo](https://github.com/raminmohammadi/MLOps/tree/main/Labs/API_Labs/Streamlit_Labs) — Iris Flower Prediction with FastAPI backend.

## Modifications from Original

| Original | Modified |
|----------|----------|
| Iris dataset (4 features) | **Wine dataset** (13 features) |
| Single model via FastAPI | **Random Forest + Gradient Boosting** with hyperparameter tuning |
| Requires FastAPI backend | **Self-contained** — trains within the app |
| Single page | **4 pages**: Home, Data Explorer, Model Training, Predict |
| No visualizations | **Confusion matrix, feature importance, PCA, correlation heatmap** |
| JSON upload only | **Sliders + JSON upload** for prediction |
| No model metrics | **Accuracy, F1 score, classification report** displayed |

## How to Run
```bash
cd streamlit_lab
pip install -r requirements.txt
streamlit run Dashboard.py
```
App opens at http://localhost:8501

## Project Structure
```
streamlit_lab/
├── Dashboard.py        # Main app
├── requirements.txt    # Dependencies
├── data/
│   └── test.json       # Sample test input
└── README.md
```