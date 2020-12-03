import pandas as pd
import numpy as np
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')
from sklearn.base import BaseEstimator, TransformerMixin
import re

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        #print("text:{}".format(text))
        pos_tags = nltk.pos_tag(tokenize(text))
        if len(pos_tags) == 0:
            return False
        first_word, first_tag = pos_tags[0]
        if first_tag in ['VB', 'VBP'] or first_word == 'RT':
            return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(len)
        return pd.DataFrame(X_tagged)

def tokenize(text):
    '''
    :param text: document to be tokenized
    :return: array of tokens where the original document is reduced by removing punctuation, stop words, lemmatized
    '''
    # convert to lowercase
    text = text.lower()
    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # tokenize text
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok.strip())
        clean_tokens.append(clean_tok)
    return clean_tokens