#coding:utf-8
import re
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer


def tokenize(sentence):
    '''
        Removal of redundant blanks, participles, word marking
    '''
    sentence = re.sub(r'\s+', ' ', sentence)
    token_words = word_tokenize(sentence)
    token_words = pos_tag(token_words)
    return token_words


wordnet_lematizer = WordNetLemmatizer()


def stem(token_words):
    '''
        Morphological normalization
    '''
    words_lematizer = []
    for word, tag in token_words:
        if tag.startswith('NN'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='n')  # n noun
        elif tag.startswith('VB'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='v')  # v verb
        elif tag.startswith('JJ'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='a')  # a Adjectives
        elif tag.startswith('R'):
            word_lematizer = wordnet_lematizer.lemmatize(word, pos='r')  # r Pronouns
        else:
            word_lematizer = wordnet_lematizer.lemmatize(word)
        words_lematizer.append(word_lematizer)
    return words_lematizer


sr = stopwords.words('english')


def delete_stopwords(token_words):
    '''
        delete stop words
    '''
    cleaned_words = [word for word in token_words if word not in sr]
    return cleaned_words


def is_number(s):
    '''
        Determine if a string is a number
    '''
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


characters = [' ', ',', '.', 'DBSCAN', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '-', '...',
              '^', '{', '}']


def delete_characters(token_words):
    '''
        Removal of special characters, numbers
    '''
    words_list = [word for word in token_words if word not in characters and not is_number(word)]
    return words_list


def to_lower(token_words):
    '''
        Uniform lower case
    '''
    words_lists = [x.lower() for x in token_words]
    return words_lists


def pre_process(text):
    '''
        Text pre-processing
    '''
    token_words = tokenize(text)
    token_words = stem(token_words)
    token_words = delete_stopwords(token_words)
    token_words = delete_characters(token_words)
    token_words = to_lower(token_words)
    return token_words


if __name__ == '__main__':
    text = 'This is an example of English sentence which contains some other characters !@#$%^&*()...'
    token_words = tokenize(text)
    token_words = stem(token_words)
    token_words = delete_stopwords(token_words)
    token_words = delete_characters(token_words)
    token_words = to_lower(token_words)
    print(token_words)
