from gensim.models import Word2Vec
from gensim.utils import simple_preprocess # Convert a document into a list of lowercase tokens, ignoring tokens that are too short or too long.
from gensim.parsing.preprocessing import remove_stopwords

def skipgram_wordembeddding(sentence_list):
    tokenized_sentences = [simple_preprocess(remove_stopwords(sentence)) for sentence in sentence_list]

    # Train a skip-gram Word2Vec model
    model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)