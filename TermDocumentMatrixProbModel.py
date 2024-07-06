from createTermDocumentMatrix import VectorTermDocumentMatrix
from pathlib import Path
from os import PathLike
import numpy as np
import pickle


class ProbabilisticTermDocumentMatrix(VectorTermDocumentMatrix):
    def __init__(self, vector_term_document_matrix: list[list[float]], word_list: list[str],
                 document_list: list[Path | PathLike], inverse_document_frequency: list[int],
                 inverted_index: list[set[Path | PathLike]]):
        super().__init__(vector_term_document_matrix, word_list,
                         document_list, inverse_document_frequency, inverted_index)
        self.p_t_R = 0.5 * np.ones((len(self.word_list),), dtype=np.float32)
        self.p_t_R_compliment = np.array(inverse_document_frequency, dtype=np.float32) / len(document_list)
        self.adjustment_factor = self.p_t_R_compliment

    @classmethod
    def load(cls, path: Path | PathLike) -> "ProbabilisticTermDocumentMatrix":
        with open(path, "rb") as f:
            return pickle.load(f)


def main():
    import argparse

    parser = argparse.ArgumentParser(prog="TermDocumentMatrixProbModel")
    parser.add_argument("vector_term_matrix_path")
    parser.add_argument("-o", "--output_folder", default=".")
    args = parser.parse_args()
    output_path = Path(args.output_folder)
    vector_term_matrix_path = Path(args.vector_term_matrix_path)
    vector_matrix = VectorTermDocumentMatrix.load(vector_term_matrix_path)
    lsi = ProbabilisticTermDocumentMatrix(vector_matrix.matrix, vector_matrix.word_list, vector_matrix.document_list,
                                          vector_matrix.inverse_document_frequency, vector_matrix.inverted_index)
    lsi.dump(output_path / "probabilistic_term_matrix.bin")


if __name__ == "__main__":
    main()
