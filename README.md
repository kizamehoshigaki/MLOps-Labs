# FastAPI Lab - Wine Classifier API

A FastAPI application that serves a Random Forest classifier trained on the [Wine dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset) from scikit-learn.

## Overview

This lab demonstrates how to expose a machine learning model as a REST API using **FastAPI** and **uvicorn**. The workflow involves:

1. Training a Random Forest Classifier on the Wine dataset (13 features, 3 classes)
2. Serving the trained model through a `/predict` API endpoint

## Modifications from Original Lab

| | Original Lab | This Lab |
|---|---|---|
| **Dataset** | Iris | Wine |
| **Model** | Decision Tree | Random Forest |
| **Response** | Prediction int only | Prediction int + class name |

## Project Structure

```
fastapi_lab1/
├── .gitignore
├── README.md
├── requirements.txt
├── model/
│   └── wine_model.pkl
└── src/
    ├── __init__.py
    ├── data.py          # Loads Wine dataset
    ├── train.py         # Trains Random Forest, saves model
    ├── predict.py       # Loads model, returns predictions
    └── main.py          # FastAPI app with /predict endpoint
```

## Setup & Run

### Install dependencies
```bash
pip install fastapi[all] scikit-learn joblib
```

### Train the model
```bash
cd src
python train.py
```

### Start the API
```bash
uvicorn main:app --reload
```

### Test the API
Open http://127.0.0.1:8000/docs and use the Swagger UI to test the `/predict` endpoint.

**Sample request body:**
```json
{
  "alcohol": 13.0,
  "malic_acid": 1.5,
  "ash": 2.3,
  "alcalinity_of_ash": 15.0,
  "magnesium": 100.0,
  "total_phenols": 2.7,
  "flavanoids": 3.0,
  "nonflavanoid_phenols": 0.3,
  "proanthocyanins": 1.7,
  "color_intensity": 5.0,
  "hue": 1.0,
  "od280_od315": 3.0,
  "proline": 1000.0
}
```

**Sample response:**
```json
{
  "prediction": 0,
  "wine_class": "class_0"
}
```

## Tech Stack

- **FastAPI** - Web framework for building APIs
- **uvicorn** - ASGI server
- **scikit-learn** - ML model training
- **Pydantic** - Request/response validation