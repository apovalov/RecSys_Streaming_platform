import os
import pickle
import sys
from typing import List

import implicit
import numpy as np
import pandas as pd
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy import sparse

app = FastAPI()


class User(BaseModel):
    """Class of json output"""
    user_id: int
    personal: List


def process_data(path_from: str):
    """Function process

    Parameters
    ----------
    path_from : str
        path to read data

    Returns
    -------
    data: pandas.DataFrame
        dataframe after proccessing
    sparse_item_user: scipy.sparse.csc_matrix
        sparce item user csc matrix
    """

    column_names = ["uid", "session", "streamer_name", "time_start", "time_end"]
    data = pd.read_csv(path_from, names=column_names, header=None)

    data["uid"] = data["uid"].astype("category")
    data["streamer_name"] = data["streamer_name"].astype("category")
    data["user_id"] = data["uid"].cat.codes
    data["streamer_id"] = data["streamer_name"].cat.codes
    data['total_time_stream'] = data['time_end'] - data['time_start']

    # Создаем разреженную матрицу
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
    """Обучение модели ALS и сохранение её в файл

    Parameters
    ----------
    sparse_item_user : scipy.sparse.csr_matrix
        Разреженная матрица взаимодействий пользователь-стример
    model_path : str
        Путь для сохранения модели
    iterations : int, optional
        Количество итераций, по умолчанию 12
    factors : int, optional
        Количество факторов, по умолчанию 500
    regularization : float, optional
        Регуляризация, по умолчанию 0.2
    alpha : float, optional
        Коэффициент увеличения значений матрицы, по умолчанию 100
    random_state : int, optional
        Случайное состояние, по умолчанию 42

    Returns
    -------
    model : implicit.als.AlternatingLeastSquares
        Обученная модель ALS
    """
    # Создание модели ALS
    model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=random_state
    )

    # Обучение модели
    # Умножение на alpha для увеличения значений матрицы перед обучением
    model.fit((sparse_item_user * alpha).astype('double'), show_progress=True)

    # Сохранение модели
    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    return model

def load_model(
    model_path: str,
):
    """Function that load model from path

    Parameters
    ----------
    path : str
        Path to read model as pickle format

    Returns
    -------
    model: AlternatingLeastSquares
        Trained model
    """
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model


def personal_recommendations(
        user_id: int,
        n_similar: int,
        model: implicit.als.AlternatingLeastSquares,
        data: pd.DataFrame,
) -> List:
    """Генерация персональных рекомендаций с использованием модели ALS

    Parameters
    ----------
    user_id : int
        Идентификатор пользователя для генерации рекомендаций
    n_similar : int
        Количество рекомендуемых стримеров
    model : implicit.als.AlternatingLeastSquares
        Обученная модель ALS
    data : pd.DataFrame
        Данные, содержащие имена стримеров и их идентификаторы

    Returns
    -------
    similar_items : List
        Список рекомендованных стримеров
    """
    # Использование внутреннего кодированного представления 'user_id'
    if user_id not in data['user_id'].unique():
        return []

    # Получение рекомендаций для пользователя
    recommended = model.recommend(user_id, n_similar)

    # Преобразование индексов обратно в имена стримеров
    similar_items = [data.loc[data['streamer_id'] == idx, 'streamer_name'].values[0] for idx, _ in recommended]

    return similar_items


@app.get("/recommendations/user/{user_id}")
async def get_recommendation(user_id: int) -> User:
    # Проверка, есть ли пользователь в данных
    if user_id not in data['uid'].unique():
        raise HTTPException(status_code=404, detail="User not found")

    # Получение персональных рекомендаций
    personal_recommendations = personal_recommendations(user_id, 100, model, data)

    # Возвращение данных пользователю
    user_info = User(user_id=user_id, personal=personal_recommendations)
    return user_info

# # Путь к файлу данных
# data_path = "../data/data_recsys.csv"
#
# # Загрузка и обработка данных
# data, sparse_item_user = process_data(data_path)
#
# # Разделение на обучающую и тестовую выборки
# # train, test = train_test_split(sparse_item_user, test_size=0.2)
#
# fit_model(sparse_item_user, "../mls/model1.pkl")



# def main() -> None:
#     """Run application"""
#     uvicorn.run("solution:app", host="localhost")
#
#
# if __name__ == "__main__":
#     main()
