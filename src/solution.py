import os
import pickle
import sys
from typing import List

import implicit
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy import sparse

app = FastAPI()


class User(BaseModel):
    """
    Class representing a user with recommendations.

    Attributes
    ----------
    user_id : int
        Identifier of the user.
    personal : List
        List of personal recommendations for the user.
    """
    user_id: int
    personal: List


def process_data(path_from: str):
    """
    Processes the input data from a specified path.

    Parameters
    ----------
    path_from : str
        Path from which to read the data.

    Returns
    -------
    data : pandas.DataFrame
        Processed DataFrame.
    sparse_item_user : scipy.sparse.csr_matrix
        Sparse matrix of item-user interactions.
    """
    column_names = ["uid", "session", "streamer_name", "time_start", "time_end"]
    data = pd.read_csv(path_from, names=column_names, header=None)
    data["uid"] = data["uid"].astype("category")
    data["streamer_name"] = data["streamer_name"].astype("category")
    data["user_id"] = data["uid"].cat.codes
    data["streamer_id"] = data["streamer_name"].cat.codes
    data['total_time_stream'] = data['time_end'] - data['time_start']

    sparse_item_user = sparse.csr_matrix(
        (data['total_time_stream'], (data['streamer_id'], data['user_id'])),
        shape=(data['streamer_id'].nunique(), data['user_id'].nunique())
    )

    return data, sparse_item_user


def fit_model(
    sparse_item_user,
    model_path: str,
    iterations: int = 12,
    factors: int = 500,
    regularization: float = 0.2,
    alpha: float = 100,
    random_state: int = 42,
):
    """
    Fits the ALS model and saves it to a file.

    Parameters
    ----------
    sparse_item_user : scipy.sparse.csr_matrix
        Sparse matrix of item-user interactions.
    model_path : str
        Path to save the model.
    iterations : int, optional
        Number of iterations (default is 12).
    factors : int, optional
        Number of factors (default is 500).
    regularization : float, optional
        Regularization parameter (default is 0.2).
    alpha : float, optional
        Alpha value for confidence (default is 100).
    random_state : int, optional
        Random state for reproducibility (default is 42).

    Returns
    -------
    model : implicit.als.AlternatingLeastSquares
        Trained ALS model.
    """
    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state
    )

    model.fit((sparse_item_user * alpha).astype('double'))

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    return model


def load_model(model_path: str):
    """
    Loads a model from the specified path.

    Parameters
    ----------
    model_path : str
        Path to the model file.

    Returns
    -------
    model : implicit.als.AlternatingLeastSquares
        Loaded ALS model.
    """
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def personal_recommendations(
    user_id: int,
    n_similar: int,
    model: implicit.als.AlternatingLeastSquares,
    data: pd.DataFrame,
    sparse_item_user: sparse.csr_matrix
) -> List:
    """
    Generates personal recommendations for a given user.

    Parameters
    ----------
    user_id : int
        Identifier of the user.
    n_similar : int
        Number of similar items to find.
    model : implicit.als.AlternatingLeastSquares
        The ALS model.
    data : pandas.DataFrame
        DataFrame containing user data.
    sparse_item_user : scipy.sparse.csr_matrix
        Sparse matrix of item-user interactions.

    Returns
    -------
    List
        List of recommended items.
    """
    if user_id not in data['uid'].unique():
        return []

    internal_user_id = data[data['uid'] == user_id]['user_id'].iloc[0]
    recommended = model.recommend(internal_user_id, sparse_item_user[internal_user_id], N=n_similar)

    if not recommended:
        return []

    similar_items = []
    for idx in recommended:
        streamer_id = idx[0]
        if streamer_id in data['streamer_id'].values:
            streamer_name = data.loc[data['streamer_id'] == streamer_id, 'streamer_name'].values[0]
            similar_items.append(streamer_name)

    return similar_items


@app.get("/recomendations/user/{user_id}")
async def get_recomendation(user_id: int) -> User:
    """
    API endpoint to get user recommendations.

    Parameters
    ----------
    user_id : int
        Identifier of the user.

    Returns
    -------
    User
        User object with recommendations.
    """
    data_path = os.path.join(sys.path[0], os.environ.get("data_path", "data_recsys.csv"))
    model_path = os.path.join(sys.path[0], os.environ.get("model_path", "model.pkl"))

    model = load_model(model_path)
    data, sparse_item_user = process_data(data_path)

    if user_id not in data['uid'].unique():
        raise HTTPException(status_code=404, detail="User not found")

    recommendations = personal_recommendations(user_id, 100, model, data, sparse_item_user)
    return User(user_id=user_id, personal=recommendations)


def main() -> None:
    """
    Main function to run the application.
    """
    uvicorn.run("solution:app", host="localhost")


if __name__ == "__main__":
    main()
