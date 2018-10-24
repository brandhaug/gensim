"""
The task is to index and query book paragraphs. For example: a query “How taxes influence Economics?”
should (among the others intermediate results) should result in printing 3 the most relevant paragraphs
(up to 5 lines each) from the book according to LSI (over TF-IDF) model:
"""
import random
import codecs
from nltk.stem.porter import PorterStemmer
import string
from gensim import corpora, models, similarities
import time
import os.path

start = time.time()

"""
1. Data loading and preprocessing
In this part we load and process (clean, tokenize and stem) data.
"""
random.seed(123)
translator = str.maketrans('', '', string.punctuation)
stemmer = PorterStemmer()

file = codecs.open("assets/pg3300.txt", "r", "utf-8")
lines = file.readlines()

collection = []
tokenized_collection = []
paragraph = ''
remove_paragraph = False

for i, line in enumerate(lines):
    line_stripped = line.strip()

    if not line_stripped and paragraph:  # new paragraph
        if remove_paragraph:  # paragraph includes Gutenberg
            remove_paragraph = False
        else:  # add paragraph to collection
            collection.append(paragraph)  # keep a copy of original paragraphs
            paragraph = paragraph.translate(translator)  # remove punctuation
            tokenized_collection.append(paragraph.split(' '))  # split paragraphs into list of words
        paragraph = ''  # reset paragraph
    elif 'Gutenberg' in line:  # header / footer
        remove_paragraph = True
    else:  # add line to paragraph
        paragraph += line_stripped

for i, paragraph in enumerate(tokenized_collection):
    for j, word in enumerate(paragraph):
        tokenized_collection[i][j] = stemmer.stem(word)  # stem and make lowercase

dictionary = corpora.Dictionary(tokenized_collection)
dictionary.save('assets/dictionary.dict')

"""
2. Dictionary building
In this part we filter (remove stopwords) and convert paragraphs into Bags-of-Words.
https://radimrehurek.com/gensim/tut1.html
"""

file = codecs.open("assets/common-english-words.txt", "r", "utf-8")
lines = file.readlines()

stopwords = []

for line in lines:
    stopwords.extend(line.split(','))  # add stopwords in line to stopwords list

stopword_ids = []

for stopword in stopwords:
    if stopword in dictionary.token2id:
        stopword_ids.append(dictionary.token2id[stopword])  # add stopword_id to stopword id list

dictionary.filter_tokens(bad_ids=[stopword_id for stopword_id in stopword_ids])  # filter out stopwords

corpus = [dictionary.doc2bow(document) for document in tokenized_collection]

corpora.MmCorpus.serialize('assets/corpus.mm', corpus)

"""
3. Retrieval Models
In this part we convert Bags-of-Words into TF-IDF weights and then LSI(Latent Semantic Indexing)
weights.
https://radimrehurek.com/gensim/tut2.html
https://radimrehurek.com/gensim/tut3.html
"""
tfidf_model = models.TfidfModel(corpus)  # initialize a model
corpus_tfidf = tfidf_model[corpus]  # apply transformation to corpus
tfidf_model.save('assets/model.tfidf')
tfidf_index = similarities.MatrixSimilarity(corpus_tfidf)  # transform corpus to tfidf space and index it
tfidf_index.save('assets/tfidf.index')

lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary,
                            num_topics=100)  # initialize an LSI transformation (100 dimensions)
corpus_lsi = lsi_model[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
lsi_model.save('assets/model.lsi')
lsi_index = similarities.MatrixSimilarity(corpus_lsi)  # transform corpus to LSI space and index it
lsi_index.save('assets/lsi.index')

"""
4. Querying
In this part we query the models built in the previous part and report results.
"""

query = "What is the function of money?"
print('Query:', query)
query = query.strip()  # remove redundant spaces etc
query = query.translate(translator)  # remove punctuation
query = query.split(' ')  # split paragraphs into list of words

for i, word in enumerate(query):
    query[i] = stemmer.stem(word)  # stem and make lowercase

query = dictionary.doc2bow(query)  # bag of words representation
print('Preprocessed query', query)

tfidf_query = tfidf_model[query]  # apply transformation to query
print('\nTF-IDF Query', tfidf_query)

doc2similarity = enumerate(tfidf_index[tfidf_query])
tfidf_results = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]
print('TF-IDF Best Results', tfidf_results)

for i, result in enumerate(tfidf_results):
    paragraph_index = result[0]
    print('\nResult %d - [Paragraph %d]' % (i + 1, paragraph_index))
    print(collection[paragraph_index])

print('\n=================================')

lsi_query = lsi_model[tfidf_query]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
print('\nLSI Query', lsi_query)
lsi_results = sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:3]
print('\nLSI Worst Results', lsi_results)

for i, result in enumerate(lsi_results):
    paragraph_index = result[0]
    print('\nResult %d - [Paragraph %d]' % (i + 1, paragraph_index))
    print(collection[paragraph_index])

print('\n=================================')

doc2similarity = enumerate(lsi_index[lsi_query])
lsi_results2 = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]
print('\nLSI Best Results', lsi_results2)

for i, result in enumerate(lsi_results2):
    paragraph_index = result[0]
    print('\nResult %d - [Paragraph %d]' % (i + 1, paragraph_index))
    print(collection[paragraph_index])

end = time.time()
print('\nExecuted in %0.4fs' % (end - start))
