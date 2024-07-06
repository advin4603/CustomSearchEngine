from TermDocumentMatrixProbModel import ProbabilisticTermDocumentMatrix
import numpy as np
from pathlib import Path
from os import PathLike
from queryBooleanRepresentationProbModel import query_to_vector

CONVERGE_EPSILON = 2e-10


def rank(query_vector: list[float], term_matrix: ProbabilisticTermDocumentMatrix, n: int) -> dict[
    Path | PathLike, float]:
    document_score = {}
    while True:
        prob = (np.log(term_matrix.p_t_R / (1 - term_matrix.p_t_R)) +
                np.log((1 - term_matrix.p_t_R_compliment) / term_matrix.p_t_R_compliment))
        prob_query = query_vector * prob
        similarity = np.array(term_matrix.matrix) @ prob_query
        document_score = sorted(zip(term_matrix.document_list, similarity),
                                key=lambda n: n[1],
                                reverse=True)[:n]
        V = set(document for document, _ in document_score)
        V_i_counts = np.array(list(len(V & term_matrix.inverted_index[i]) for i in range(len(term_matrix.word_list))))
        old_p_t_R = term_matrix.p_t_R
        old_p_t_R_compliment = term_matrix.p_t_R_compliment
        term_matrix.p_t_R = (V_i_counts + term_matrix.adjustment_factor) / (len(V) + 1)

        term_matrix.p_t_R_compliment = (np.array(
            term_matrix.inverse_document_frequency) - V_i_counts + term_matrix.adjustment_factor
                                        ) / (len(term_matrix.document_list) - len(V) + 1)

        difference = np.abs(term_matrix.p_t_R - old_p_t_R) + np.abs(term_matrix.p_t_R_compliment - old_p_t_R_compliment)
        if difference.sum() < CONVERGE_EPSILON:
            break

    return {document: score for document, score in document_score}


def main():
    import argparse

    parser = argparse.ArgumentParser(prog="RankingProbModel")
    parser.add_argument("probabilistic_term_matrix_path")
    parser.add_argument("n")
    args = parser.parse_args()
    n = int(args.n)
    prob_term_matrix_path = Path(args.probabilistic_term_matrix_path)
    prob_term_matrix = ProbabilisticTermDocumentMatrix.load(prob_term_matrix_path)
    query_vector = query_to_vector(input("Query: "), prob_term_matrix)
    ranks = rank(query_vector, prob_term_matrix, n)
    for path, score in ranks.items():
        print(path.name, score)


if __name__ == "__main__":
    main()
