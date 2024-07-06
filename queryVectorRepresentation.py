from preprocessing import extract_terms
from createTermDocumentMatrix import VectorTermDocumentMatrix
from collections import Counter
import numpy as np


def query_to_vector(query: str, term_matrix: VectorTermDocumentMatrix) -> list[float]:
    terms = extract_terms(query)
    frequencies = Counter(terms)
    term_frequencies = np.array([frequencies[word] for word in term_matrix.word_list], dtype=np.float32)
    term_frequencies /= term_frequencies.max() if term_frequencies.max() else 1
    tf_idf = term_frequencies * np.log(
        len(term_matrix.document_list) / np.array(term_matrix.inverse_document_frequency, dtype=np.float32))
    return list(tf_idf)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="QueryVectorRepresentation",
                                     description="creates the vector representation of the query")
    parser.add_argument("vector_term_matrix_path")
    args = parser.parse_args()
    vector_term_matrix_path = Path(args.vector_term_matrix_path)
    vector_matrix = VectorTermDocumentMatrix.load(vector_term_matrix_path)
    print(query_to_vector(input("Query: "), vector_matrix))
