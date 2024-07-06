from createTermDocumentMatrix import VectorTermDocumentMatrix
import numpy as np
from pathlib import Path
from os import PathLike
from sklearn.preprocessing import normalize


def top_n_similarity(query_vector: list[float], term_matrix: VectorTermDocumentMatrix, n: int) -> dict[
    Path | PathLike, float]:
    query_vector = np.array(query_vector)
    query_vector /= np.linalg.norm(query_vector)
    matrix = np.array(term_matrix.matrix)
    matrix = normalize(matrix)
    cosine_similarity = matrix @ query_vector
    cosine_similarity = cosine_similarity.tolist()
    document_score = sorted(zip(term_matrix.document_list, cosine_similarity), key=lambda n: n[1], reverse=True)
    return {document: score for document, score in document_score[:n]}

