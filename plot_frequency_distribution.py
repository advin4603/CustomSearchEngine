import argparse
from pathlib import Path
from preprocessing import extract_terms
import nltk

parser = argparse.ArgumentParser(prog="tag_cloud", description="creates the tag cloud")
parser.add_argument("file_path")
parser.add_argument("-w", "--max_words", default="20")
parser.add_argument("-s", "--no_stop_words", action="store_true")

args = parser.parse_args()
file_path = Path(args.file_path)
w = int(args.max_words)
remove_stop_words = bool(args.no_stop_words)
with open(file_path, "rb") as f:
    text = f.read().decode(errors="replace")

terms = extract_terms(text, remove_stop_words=remove_stop_words)

frequency_distribution = nltk.FreqDist(terms)
frequency_distribution.plot(w, cumulative=False)
