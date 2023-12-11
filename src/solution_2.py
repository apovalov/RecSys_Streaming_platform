import os
import pickle
import sys
from typing import List

import implicit
import numpy as np
import pandas as pd
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
from scipy import sparse

app = FastAPI()

class User(BaseModel):
    """Class of json output"""
    user_id: int
    personal: List

def process_data(path_from: str):
    data = pd.read_csv(path_from)
    data["uid"] = data["uid"].astype("category")
    data["streamer_name"] = data["streamer_name"].astype("category")
    data["user_id"] = data["uid"].cat.codes
    data["streamer_id"] = data["streamer_name"].cat.codes
    sparse_item_user = sparse.csr_matrix((data['time_end'] - data['time_start'], (data['streamer_id'], data['user_id'])))
    return data, sparse_item_user

def fit_model(sparse_item_user, model_path: str, iterations: int = 12, factors: int = 500, regularization: float = 0.2, alpha: float = 100, random_state: int = 42):
    model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=regularization, iterations=iterations, random_state=random_state)
    model.fit(sparse_item_user * alpha)
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
    return model

def load_model(model_path: str):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

def personal_recomendations(user_id: int, n_similar: int, model: implicit.als.AlternatingLeastSquares, data: pd.DataFrame) -> List:
    recommended = model.recommend(user_id, sparse_item_user.T.tocsr(), N=n_similar)
    similar_items = [data.streamer_name.loc[data.streamer_id == x[0]].iloc[0] for x in recommended]
    return similar_items

@app.get("/recomendations/user/{user_id}")
async def get_recomendation(user_id: int):
    personal = personal_recomendations(user_id, 100, model, data)
    user = User(user_id=user_id, personal=personal)
    return user

def main() -> None:
    uvicorn.run("solution:app", host="localhost")

if __name__ == "__main__":
    main()