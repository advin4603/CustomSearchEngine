from pathlib import Path
from os import PathLike
import numpy as np
from typing import Any
import preprocessing
import pickle


def create_vector_representation(frequencies: dict[str, int], inverted_index: dict[str, set[int]],
                                 document_count: int) -> list[float]:
    frequency_vector = np.array(list(frequencies.values()), dtype=np.float32)
    max_frequency = frequency_vector.max()
    tf_vector = frequency_vector / max_frequency if max_frequency != 0 else frequency_vector
    idf_vector = np.log(document_count / np.array([len(inverted_index[word]) for word in frequencies]))
    return (tf_vector * idf_vector).tolist()


def create_boolean_representation(document_index: int, inverted_index: dict[str, set[int]]) -> list[bool]:
    boolean_vector = [document_index in document_set for document_set in inverted_index.values()]
    return boolean_vector


def create_inverted_index(document_index_map: dict[Path | PathLike, int], remove_stop_words=False) -> dict[
    str, set[int]]:
    inverted_index: dict[str, set[int]] = {}
    for document_path, document_index in document_index_map.items():
        with open(document_path, "rb") as document:
            for line in document:
                terms = preprocessing.extract_terms(line.decode(errors="replace"), remove_stop_words)
                for term in terms:
                    inverted_index.setdefault(term, set()).add(document_index)

    return inverted_index


def get_frequency(word_frequency_ordering: dict[str, Any], document_path: Path | PathLike, remove_stop_words=False) -> \
        dict[str, int]:
    frequencies: dict[str, int] = {word: 0 for word in word_frequency_ordering}
    with open(document_path, "rb") as document:
        for line in document:
            terms = preprocessing.extract_terms(line.decode(errors="replace"), remove_stop_words)
            for term in terms:
                if term in word_frequency_ordering:
                    frequencies[term] += 1
    return frequencies


class TermDocumentMatrix:
    def __init__(self, word_list: list[str], document_list: list[Path | PathLike]):
        self.matrix: list[list] = []
        self.word_list = word_list
        self.document_list = document_list

    def dump(self, path: Path | PathLike):
        with open(path, "wb") as f:
            pickle.dump(self, f)


class VectorTermDocumentMatrix(TermDocumentMatrix):
    def __init__(self, vector_term_document_matrix: list[list[float]], word_list: list[str],
                 document_list: list[Path | PathLike], inverse_document_frequency: list[int],
                 inverted_index: list[set[Path | PathLike]]):
        super().__init__(word_list, document_list)
        self.matrix: list[list[float]] = vector_term_document_matrix
        self.inverse_document_frequency = inverse_document_frequency
        self.inverted_index = inverted_index

    @classmethod
    def load(cls, path: Path | PathLike) -> "VectorTermDocumentMatrix":
        with open(path, "rb") as f:
            return pickle.load(f)


class BooleanTermDocumentMatrix(TermDocumentMatrix):
    def __init__(self, boolean_term_document_matrix: list[list[bool]], word_list: list[str],
                 document_list: list[Path | PathLike]):
        super().__init__(word_list, document_list)
        self.matrix: list[list[bool]] = boolean_term_document_matrix

    @classmethod
    def load(cls, path: Path | PathLike) -> "BooleanTermDocumentMatrix":
        with open(path, "rb") as f:
            return pickle.load(f)


def create_term_document_matrix(documents_parent_path: Path | PathLike, p: int = 0, remove_stop_words=False) -> (
        tuple)[BooleanTermDocumentMatrix, VectorTermDocumentMatrix]:
    documents_parent_path = Path(documents_parent_path)
    document_paths = [document_path for document_path in documents_parent_path.glob("**/*.txt") if
                      document_path.is_file() and not document_path.stem.endswith("top_p")]
    document_index_map: dict[Path, int] = {document_path: index for index, document_path in enumerate(document_paths)}
    inverted_index = create_inverted_index(document_index_map, remove_stop_words)
    vector_matrix = []
    boolean_matrix = []
    for document_path, document_index in document_index_map.items():
        frequencies = get_frequency(inverted_index, document_path, remove_stop_words)
        vector_representation = create_vector_representation(frequencies, inverted_index, len(document_index_map))

        vector_matrix.append(vector_representation)
        if not p:
            boolean_representation = create_boolean_representation(document_index, inverted_index)
            boolean_matrix.append(boolean_representation)
    if p:
        averaged_tfidf = np.array(vector_matrix).mean(0)
        top_p = sorted(zip(inverted_index.keys(), averaged_tfidf), key=lambda n: n[1], reverse=True)[:p]
        inverted_index = {term: inverted_index[term] for term, _ in top_p}
        vector_matrix = []
        boolean_matrix = []
        for document_path, document_index in document_index_map.items():
            frequencies = get_frequency(inverted_index, document_path, remove_stop_words)
            vector_representation = create_vector_representation(frequencies, inverted_index, len(document_index_map))

            vector_matrix.append(vector_representation)
            boolean_representation = create_boolean_representation(document_index, inverted_index)
            boolean_matrix.append(boolean_representation)

    return (BooleanTermDocumentMatrix(boolean_matrix, list(inverted_index.keys()),
                                      list(document_index_map.keys())),
            VectorTermDocumentMatrix(vector_matrix, list(inverted_index.keys()), list(document_index_map.keys()),
                                     list(len(i) for i in inverted_index.values()),
                                     list(set(document_paths[i] for i in docs) for docs in inverted_index.values())))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="CreateTermDocumentMatrix", description="creates the term document matrices")
    parser.add_argument("documents_folder_path")
    parser.add_argument("-o", "--output_folder", default=".")
    parser.add_argument("-p", "--top_p", default="0")
    parser.add_argument("-s", "--no_stop_words", action="store_true")

    args = parser.parse_args()
    folder_path = Path(args.documents_folder_path)
    output_path = Path(args.output_folder)
    p = int(args.top_p)
    remove_stop_words = bool(args.no_stop_words)
    boolean, vector = create_term_document_matrix(folder_path, p, remove_stop_words=remove_stop_words)
    boolean.dump(output_path / "boolean_term_matrix.bin")
    vector.dump(output_path / "vector_term_matrix.bin")
