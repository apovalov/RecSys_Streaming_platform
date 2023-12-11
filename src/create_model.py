# from implicit.evaluation import precision_at_k, ndcg_at_k
# from sklearn.model_selection import ParameterGrid
# from implicit.als import AlternatingLeastSquares
# from collab_filter import process_data
# from sklearn.model_selection import train_test_split
#
# # Определение диапазонов параметров для Grid Search
# param_grid = {
#     'factors': [500], #[50, 100, 200],
#     'regularization': [0.2], #[0.01, 0.1, 0.5],
#     'alpha': [100], #[10, 40, 80],
#     'iterations': [12] #[10, 20, 30]
# }
#
# # Лучшая конфигурация и результаты
# best_precision = 0
# best_params = None
#
# # Путь к файлу данных
# data_path = "../data/data_recsys.csv"
#
# # Загрузка и обработка данных
# data, sparse_item_user = process_data(data_path)
#
# # Разделение на обучающую и тестовую выборки
# train, test = train_test_split(sparse_item_user, test_size=0.2)
#
#
# # Перебор комбинаций параметров
# # for params in ParameterGrid(param_grid):
#     model = AlternatingLeastSquares(factors=500,
#                                     regularization=params['regularization'],
#                                     iterations=params['iterations'])
#     model.fit(train * params['alpha'])
#
#     precision = precision_at_k(model, train, test, K=10)
#     ndcg = ndcg_at_k(model, train, test, K=10)
#
#     print('Precision@10: {:.4f}, NDCG@10: {:.4f} for {}'.format(precision, ndcg, params))
#
#     # if precision > best_precision:
#     #     best_precision = precision
#     #     best_params = params
#
# print("Лучшие параметры:", best_params)
