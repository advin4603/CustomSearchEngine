import argparse
from pathlib import Path
from preprocessing import extract_terms
from createTermDocumentMatrix import VectorTermDocumentMatrix

parser = argparse.ArgumentParser(prog="cluster_and_plot",
                                 description="compare stems and extracted keywords in .key files")
parser.add_argument("vector_term_matrix_path")
parser.add_argument("keys_path")

args = parser.parse_args()

vector_term_matrix_path = Path(args.vector_term_matrix_path)
keys_path = Path(args.keys_path)

vector_term_matrix = VectorTermDocumentMatrix.load(vector_term_matrix_path)

if keys_path.is_file() and keys_path.suffix == ".key":
    files = [keys_path]
else:
    files = list(keys_path.glob("**/*.key"))

keys = set()
for file in files:
    with open(file, "rb") as f:
        text = f.read().decode(errors="replace")
        keys |= set(extract_terms(text, remove_stop_words=False))

stems = set(vector_term_matrix.word_list)
print("====Intersection====")
print("\n".join(stems & keys))
print("====Stems not in keywords====")
print("\n".join(stems - keys))
print("====Keywords not in stems====")
print("\n".join(keys - stems))
print("====Stats====")
print("Number of Common terms in stems and keywords =", len(stems & keys),
      f"= {len(stems & keys) / (len(stems) + len(keys)) * 100: .2f}% of both stems and keys")
print("Number of terms in stems but not in keywords =", len(stems - keys),
      f"= {len(stems - keys) / len(stems) * 100: .2f}% of stems")
print("Number of terms in keys but not in stems =", len(keys - stems),
      f"= {len(keys - stems) / len(keys) * 100: .2f}% of keys")
