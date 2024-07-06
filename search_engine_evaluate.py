from typing import TypeVar
import numpy as np
from scipy.stats import spearmanr, kendalltau

T = TypeVar("T")


def ranked_list_recall(ranked_list: list[T], relevant_list: set[T]) -> float:
    relevant_recommendations_count = len(set(ranked_list) & relevant_list)
    return relevant_recommendations_count / len(relevant_list)


def ranked_list_precision(ranked_list: list[T], relevant_list: set[T]) -> float:
    relevant_recommendations_count = len(set(ranked_list) & relevant_list)
    return relevant_recommendations_count / len(ranked_list)


def DCG(ranked_list: list[T], relevancy_scores: dict[T, float]) -> float:
    gains = [relevancy_scores.get(item, 0) for item in ranked_list]
    dcg = np.array(gains) / np.log2(np.arange(2, 2 + len(gains), dtype=np.float32))
    return dcg.sum()


def IDCG(ranked_list: list[T], relevancy_scores: dict[T, float]) -> float:
    max_gains = sorted(relevancy_scores.values(), reverse=True)[:len(ranked_list)]
    max_gains.extend(0 for _ in range(len(max_gains) - len(ranked_list)))
    idcg = np.array(max_gains) / np.log2(np.arange(2, 2 + len(max_gains), dtype=np.float32))
    return idcg.sum()


def nDCG(ranked_list: list[T], relevancy_scores: dict[T, float]):
    return DCG(ranked_list, relevancy_scores) / IDCG(ranked_list, relevancy_scores)


def average_precision(ranked_list: list[T], relevant_list: set[T]):
    average_precision_score = 0
    for i, item in enumerate(ranked_list):
        if item not in relevant_list:
            continue
        average_precision_score *= i
        average_precision_score += ranked_list_precision(ranked_list[:i + 1], relevant_list)
        average_precision_score /= (i + 1)
    return average_precision_score


def micro_f1(ranked_list: list[T], relevant_list: set[T]):
    precision = ranked_list_precision(ranked_list, relevant_list)
    recall = ranked_list_recall(ranked_list, relevant_list)
    return 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0


def macro_f1(ranked_list: list[T], relevant_list: set[T]):
    f1_scores = [micro_f1(ranked_list[:i + 1], relevant_list) for i in range(len(ranked_list))]
    return sum(f1_scores) / len(f1_scores)


def spearman_correlation(ranked_list: list[T], relevant_list: set[T]):
    relevance_scores = [1 if item in relevant_list else 0 for item in ranked_list]
    return spearmanr(np.arange(1, len(ranked_list) + 1), np.array(relevance_scores)).statistic


def kendall_tau(ranked_list: list[T], relevant_list: set[T]):
    relevance_scores = [1 if item in relevant_list else 0 for item in ranked_list]
    return kendalltau(np.arange(1, len(ranked_list) + 1), np.array(relevance_scores)).statistic
