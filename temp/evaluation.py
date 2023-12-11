import numpy as np

def ndcg_at_k(predicted, truth, k):
    if not truth:
        return 0

    idcg = 1 + sum(1 / np.log2(i + 2) for i in range(min(len(truth), k)))
    dcg = sum(1 / np.log2(i + 2) for i, p in enumerate(predicted[:k]) if p in truth)

    return dcg / idcg


def average_precision_at_k(predicted, truth, k):
    if not truth:
        return 0

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted[:k]):
        if p in truth and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(truth), k)


def evaluate_model(model, test_data, k=10):
    ndcg_scores = []
    map_scores = []

    for user_id, truth in test_data.items():
        # Предсказать рекомендации для пользователя
        predicted = model.recommend(user_id, ...)

        # Вычислить NDCG@k и MAP@k
        ndcg = ndcg_at_k(predicted, truth, k)
        map_score = average_precision_at_k(predicted, truth, k)

        ndcg_scores.append(ndcg)
        map_scores.append(map_score)

    mean_ndcg = np.mean(ndcg_scores)
    mean_map = np.mean(map_scores)

    return mean_ndcg, mean_map
