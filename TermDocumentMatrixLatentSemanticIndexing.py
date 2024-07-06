import numpy as np

from createTermDocumentMatrix import VectorTermDocumentMatrix
from pathlib import Path
from os import PathLike
from sklearn.preprocessing import normalize
import pickle


class LSITermDocumentMatrix(VectorTermDocumentMatrix):
    def __init__(self, vector_term_document_matrix: list[list[float]], word_list: list[str],
                 document_list: list[Path | PathLike], inverse_document_frequency: list[int],
                 reduced_dimensionality: int, inverted_index: list[set[Path | PathLike]]):
        super().__init__(vector_term_document_matrix, word_list,
                         document_list, inverse_document_frequency, inverted_index)
        term_doc_matrix = np.array(self.matrix)
        U, sigma, V = np.linalg.svd(np.transpose(term_doc_matrix))
        sigma = np.diag(sigma)
        U_l, sigma_l, V_l = U[:, :reduced_dimensionality], sigma[:reduced_dimensionality, :reduced_dimensionality], V[:,
                                                                                                                    :reduced_dimensionality]
        sigma_l_inverse = np.linalg.pinv(sigma_l)
        self.document_transform = U_l @ sigma_l_inverse
        self.document_representations = term_doc_matrix @ self.document_transform
        self.normalised_document_representations = normalize(self.document_representations)

    @classmethod
    def load(cls, path: Path | PathLike) -> "LSITermDocumentMatrix":
        with open(path, "rb") as f:
            return pickle.load(f)


def main():
    import argparse

    parser = argparse.ArgumentParser(prog="TermDocumentMatrixLatentSemanticIndexing")
    parser.add_argument("vector_term_matrix_path")
    parser.add_argument("-o", "--output_folder", default=".")
    parser.add_argument("-l", "--reduced_dimensions", default="50")
    args = parser.parse_args()
    output_path = Path(args.output_folder)
    l = int(args.reduced_dimensions)
    vector_term_matrix_path = Path(args.vector_term_matrix_path)
    vector_matrix = VectorTermDocumentMatrix.load(vector_term_matrix_path)
    lsi = LSITermDocumentMatrix(vector_matrix.matrix, vector_matrix.word_list, vector_matrix.document_list,
                                vector_matrix.inverse_document_frequency, l, vector_matrix.inverted_index)
    lsi.dump(output_path / "lsi_term_matrix.bin")


if __name__ == "__main__":
    main()
