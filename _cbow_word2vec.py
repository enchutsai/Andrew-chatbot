from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

def cbow_wordembedding(sentence_list):
    tokenized_sentences = [simple_preprocess(remove_stopwords(sentence)) for sentence in sentence_list]

    # Train a skip-gram Word2Vec model
    model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4, sg=0)