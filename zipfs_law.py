from nltk.tokenize import word_tokenize
from collections import Counter
import csv


def extract_words(text: str) -> list[str]:
    text = " ".join(text.split("-"))
    terms = word_tokenize(text, language="french")
    terms = (term.lower() for term in terms if term.isalpha())
    return list(terms)


word_counter = Counter()

with open("l'odysee.txt") as f:
    for line in f:
        word_counter.update(extract_words(line.strip()))

with open("metrics.csv", mode="w") as file:
    writer = csv.DictWriter(file, fieldnames=["Word", "Frequency", "Probability", "Theoretical Value"])
    writer.writeheader()
    total_frequency = sum(word_counter.values())
    for i, (word, frequency) in enumerate(word_counter.most_common()):
        writer.writerow({
            "Word": word,
            "Frequency": frequency,
            "Probability": frequency / total_frequency,
            "Theoretical Value": 0.1 / (i + 1)
        })
