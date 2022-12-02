from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import pickle


def transform_features(dataset: pd.DataFrame):
    dataset['mileage'] = dataset['mileage'].str.split(' ', n=1).str[0]
    dataset['engine'] = dataset['engine'].str.split(' ', n=1).str[0]
    dataset['max_power'] = dataset['max_power'].str.split(' bhp').str[0].replace("", np.nan)
    dataset.drop(columns=['torque'], inplace=True)

    if 'selling_price' in dataset.columns:
        y = dataset['selling_price']
        x = dataset.drop(columns=['selling_price'])
    else:
        y = None
        x = dataset

    return x, y


with open('modelweights.pickle', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    item_dict = pd.DataFrame.from_dict(item.dict(), orient='index').T
    x_transformed, y = transform_features(item_dict)
    return model.predict(x_transformed)[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    predict = []
    for item in items:
        item_dict = pd.DataFrame.from_dict(item.dict(), orient='index').T
        x_transformed, y = transform_features(item_dict)
        predict.append(model.predict(x_transformed)[0])

    return predict
