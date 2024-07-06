from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

with open("english.stop") as stop_word_file:
    stop_words = set()
    for word in stop_word_file:
        stop_words.add(word.strip())

SYMBOLS = {"or": "|", "and": "&", "not": "~", "(": "(", ")": ")"}


def extract_terms(text: str, remove_stop_words=False) -> list[str]:
    stemmer = PorterStemmer()
    text = " ".join(text.split("-"))
    terms = word_tokenize(text)
    terms = (term.lower() for term in terms if term.isalpha())
    if remove_stop_words:
        terms = (term for term in terms if term not in stop_words)
    terms = (stemmer.stem(term) for term in terms)
    return list(terms)


def process_boolean_query(text: str) -> list[str]:
    terms = word_tokenize(text)
    stemmer = PorterStemmer()
    terms = (term.lower() for term in terms)
    terms = (stemmer.stem(term) if term not in SYMBOLS else term for term in terms)
    return list(terms)
