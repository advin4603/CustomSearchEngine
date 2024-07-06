from similarityMeasure import top_n_similarity
from createTermDocumentMatrix import VectorTermDocumentMatrix
from queryVectorRepresentation import query_to_vector

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="compQueryVector",
                                     description="Find top N similar documents")
    parser.add_argument("vector_term_matrix_path")
    parser.add_argument("n")
    args = parser.parse_args()
    n = int(args.n)
    vector_term_matrix_path = Path(args.vector_term_matrix_path)
    vector_matrix = VectorTermDocumentMatrix.load(vector_term_matrix_path)
    query_vector = query_to_vector(input("Query: "), vector_matrix)
    for document, similarity in top_n_similarity(query_vector, vector_matrix, n).items():
        print(f"{document.name} -> {similarity}")
