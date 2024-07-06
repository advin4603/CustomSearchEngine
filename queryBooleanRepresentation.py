

from preprocessing import process_boolean_query, SYMBOLS
from createTermDocumentMatrix import BooleanTermDocumentMatrix
import ast


def query_to_boolean(query: str, term_matrix: BooleanTermDocumentMatrix):
    processed_query = process_boolean_query(query)
    unsubstituted_query = processed_query.copy()
    word_to_index = {term: index for index, term in enumerate(term_matrix.word_list)}

    for index, term in enumerate(processed_query):
        if term in SYMBOLS:
            processed_query[index] = SYMBOLS[term]
        else:
            processed_query[index] = f"\"{term}\""
    ast.parse(" ".join(processed_query))
    processed_query = unsubstituted_query

    for index, term in enumerate(processed_query):
        if term in SYMBOLS:
            processed_query[index] = SYMBOLS[term]
        elif term in word_to_index:
            processed_query[index] = f"matrix[:, {word_to_index[term]}]"
        else:
            processed_query[index] = f"array(False)"

    processed_query = " ".join(processed_query)

    return processed_query


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="QueryBooleanRepresentation",
                                     description="creates the boolean representation of the query")
    parser.add_argument("boolean_term_matrix_path")
    args = parser.parse_args()
    boolean_term_matrix_path = Path(args.boolean_term_matrix_path)
    boolean_matrix = BooleanTermDocumentMatrix.load(boolean_term_matrix_path)
    print(query_to_boolean(input("Query: "), boolean_matrix))
