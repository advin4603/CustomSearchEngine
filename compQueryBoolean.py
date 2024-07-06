from createTermDocumentMatrix import BooleanTermDocumentMatrix
from numpy import array
from pathlib import Path
from os import PathLike
from queryBooleanRepresentation import query_to_boolean


def get_documents_from_processed_query(query: str, term_matrix: BooleanTermDocumentMatrix) -> list[Path | PathLike]:
    documents: list[Path | PathLike] | bool = eval(query,
                                                   {"matrix": array(term_matrix.matrix), "array": array}).tolist()
    if not documents:
        return []
    matched_documents = (document for matched, document in zip(documents, term_matrix.document_list) if matched)
    return list(matched_documents)


def main():
    parser = argparse.ArgumentParser(prog="compQueryBoolean",
                                     description="Find matched documents")
    parser.add_argument("boolean_term_matrix_path")
    args = parser.parse_args()
    boolean_term_matrix_path = Path(args.boolean_term_matrix_path)
    boolean_matrix = BooleanTermDocumentMatrix.load(boolean_term_matrix_path)
    processed_query = query_to_boolean(input("Query: "), boolean_matrix)
    for document in get_documents_from_processed_query(processed_query, boolean_matrix):
        print(f"{document.name}")


if __name__ == "__main__":
    import argparse

    main()
