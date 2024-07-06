from TermDocumentMatrixLatentSemanticIndexing import LSITermDocumentMatrix
from pathlib import Path
from os import PathLike
from collections import Counter
from preprocessing import extract_terms
import numpy as np
from sklearn.preprocessing import normalize


def query_to_vector(query: str, term_matrix: LSITermDocumentMatrix) -> list[float]:
    terms = extract_terms(query)
    frequencies = Counter(terms)
    term_frequencies = np.array([frequencies[word] for word in term_matrix.word_list], dtype=np.float32)
    term_frequencies /= term_frequencies.max() if term_frequencies.max() else 1
    tf_idf = term_frequencies * np.log(
        len(term_matrix.document_list) / np.array(term_matrix.inverse_document_frequency, dtype=np.float32))
    return list(tf_idf)


def rank(query: list[float], lsi_term_document_matrix: LSITermDocumentMatrix, n: int) -> dict[
    Path | PathLike, float]:
    query_vector = np.array(query)
    query_representation = query_vector @ lsi_term_document_matrix.document_transform
    query_representation /= np.linalg.norm(query_representation)
    cosine_similarity = lsi_term_document_matrix.normalised_document_representations @ query_representation
    document_score = sorted(zip(lsi_term_document_matrix.document_list, cosine_similarity), key=lambda n: n[1],
                            reverse=True)
    return {document: score for document, score in document_score[:n]}


def main():
    import argparse

    parser = argparse.ArgumentParser(prog="RankingLatentSemanticIndexing")
    parser.add_argument("lsi_term_matrix_path")
    parser.add_argument("n")
    args = parser.parse_args()
    n = int(args.n)
    lsi_term_matrix_path = Path(args.lsi_term_matrix_path)
    lsi_term_matrix = LSITermDocumentMatrix.load(lsi_term_matrix_path)
    query_vector = query_to_vector(input("Query: "), lsi_term_matrix)
    ranks = rank(query_vector, lsi_term_matrix, n)
    for path, score in ranks.items():
        print(path.name, score)


if __name__ == "__main__":
    main()
