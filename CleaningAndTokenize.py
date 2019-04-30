import re
import itertools
import string
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def nltk2wn_tag(nltk_tag):
    ## Ref.: https://simonhessner.de/lemmatize-whole-sentences-with-python-and-nltks-wordnetlemmatizer/
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wn_tagged = map(lambda x: (x[0].lower(), nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if word not in stop_words and word not in string.punctuation: ## This does not remove punctuations fully.
            if tag is None:
                res_words.append(word)
            else:
                res_words.append(lemmatizer.lemmatize(word, tag))
    return res_words


def cleaning(s):
    noHyperLink = re.sub('\.\.+', '.', re.sub('\]\(.+\)', '', s)) ## This leaves many URLs intact
    sentences = nltk.sent_tokenize(text=noHyperLink)
    lemmatized_tokens = [lemmatize_sentence(sent) for sent in sentences]
    return list(itertools.chain.from_iterable(lemmatized_tokens))


text_cleaning = udf(cleaning, ArrayType(StringType()))
## This is not stable if run in Zeppelin - not sure why, and other people on Stack Overflow also experienced similar symptom with NLTK on Spark.