from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import predict, load_model

app = FastAPI(
    title="Wine Classifier API",
    description="A FastAPI app serving a Random Forest model trained on the Wine dataset.",
    version="1.0.0"
)

# Load model at startup
load_model()

class WineData(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315: float
    proline: float

class WineResponse(BaseModel):
    prediction: int
    wine_class: str

@app.get("/")
async def root():
    return {"message": "Wine Classifier API is running"}

@app.post("/predict", response_model=WineResponse)
async def predict_wine(data: WineData):
    try:
        features = [
            data.alcohol, data.malic_acid, data.ash,
            data.alcalinity_of_ash, data.magnesium, data.total_phenols,
            data.flavanoids, data.nonflavanoid_phenols, data.proanthocyanins,
            data.color_intensity, data.hue, data.od280_od315, data.proline
        ]
        pred = predict(features)
        wine_classes = {0: "class_0", 1: "class_1", 2: "class_2"}
        return WineResponse(prediction=pred, wine_class=wine_classes[pred])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))