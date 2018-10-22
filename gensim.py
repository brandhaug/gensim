"""
The task is to index and query book paragraphs. For example: a query “How taxes influence Economics?”
should (among the others intermediate results) should result in printing 3 the most relevant paragraphs
(up to 5 lines each) from the book according to LSI (over TF-IDF) model:
"""
import random
import codecs
from nltk.stem.porter import PorterStemmer
import string
import gensim
import time

start = time.time()

"""
1. Data loading and preprocessing
In this part we load and process (clean, tokenize and stem) data.
"""

random.seed(123)

file = codecs.open("assets/pg3300.txt", "r", "utf-8")
lines = file.readlines()

collection = []
tokenized_collection = []
paragraph = ''
remove_paragraph = False

translator = str.maketrans('', '', string.punctuation)
stemmer = PorterStemmer()

for i, line in enumerate(lines):
    line_stripped = line.strip()

    if not line_stripped and paragraph:  # New paragraph
        if remove_paragraph:  # Paragraph includes Gutenberg
            remove_paragraph = False
        else:  # Add paragraph to collection
            collection.append(paragraph)  # Keeps a copy of original paragraphs
            paragraph = paragraph.translate(translator)  # Remove punctuation
            paragraph = paragraph.lower()  # Make lowercase
            tokenized_collection.append(paragraph.split(' '))  # Splits paragraphs into list of words
        paragraph = ''  # Reset paragraph
    elif 'Gutenberg' in line:  # Header / footer
        remove_paragraph = True
    else:  # Add line to paragraph
        paragraph += line_stripped

for i, paragraph in enumerate(tokenized_collection):
    for j, word in enumerate(paragraph):
        tokenized_collection[i][j] = stemmer.stem(word)

"""
2. Dictionary building
In this part we filter (remove stopwords) and convert paragraphs into Bags-of-Words.
"""
dictionary = gensim.corpora.Dictionary('assets/common-english-words.txt')




end = time.time()
print(end - start)