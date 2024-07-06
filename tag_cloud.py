import argparse
from pathlib import Path
from collections import Counter
from preprocessing import extract_terms
from wordcloud import WordCloud
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(prog="tag_cloud", description="creates the tag cloud")
parser.add_argument("documents_folder_path")
parser.add_argument("-m", "--max_documents", default="15")
parser.add_argument("-p", "--word_count", default="50")
parser.add_argument("-s", "--no_stop_words", action="store_true")

args = parser.parse_args()
folder_path = Path(args.documents_folder_path)
m = int(args.max_documents)
p = int(args.word_count)
remove_stop_words = bool(args.no_stop_words)

if folder_path.is_file():
    documents = [folder_path]
else:
    documents = list(folder_path.glob("**/*.txt", ))[:m]
counter = Counter()
for document in documents:
    with open(document, "rb") as f:
        for line in f:
            counter.update(extract_terms(line.decode(errors="replace"), remove_stop_words=remove_stop_words))

wordcloud = WordCloud().generate_from_frequencies(dict(counter.most_common(p)))

plt.imshow(wordcloud, interpolation='bicubic')
plt.axis("off")
plt.show()
