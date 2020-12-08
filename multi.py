
from flair.data import Sentence
from flair.models import SequenceTagger

if __name__ == "__main__":
    sentences = ['I love Berlin.', 'My name is Tom Smith.', 'I work at Aldi.', 'I love Miami.', 'My name is Jane Smith.', 'I work at Walmart.', 'I love New York.', 'My name is Giant Smith.', 'I work at University.']
    sentences = [Sentence(s) for s in sentences]
    tagger = SequenceTagger.load("ner")
    annotated_sentences = tagger.predict_multi(sentences)
    """
    for sentence in annotated_sentences:
        for token in sentence:
            print(token.labels)
    """
