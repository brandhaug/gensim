"""
The task is to index and query book paragraphs. For example: a query “How taxes influence Economics?”
should (among the others intermediate results) should result in printing 3 the most relevant paragraphs
(up to 5 lines each) from the book according to LSI (over TF-IDF) model:
"""
import random
import codecs
import string
# from nltk.stem.porter import PorterStemmer

"""
1. Data loading and preprocessing
In this part we load and process (clean, tokenize and stem) data.
"""

# 1.0. Fix random numbers generator:
random.seed(123)

# 1.1 Open and load the file (it’s UTF-8 encoded) using codecs.
file = codecs.open("assets/pg3300.txt", "r", "utf-8")
lines = file.readlines()

collection = []

# 1.2. Partition file into separate paragraphs. Paragraphs are text chunks separated by empty line.
# 1.3. Remove (filter out) paragraphs containing the word “Gutenberg” (=headers and footers).
for i, line in enumerate(lines):
    paragraph = ''

    if line is '\n\n': # Breakpoint
        # TODO: Not Working
        print('Line %d is a breakpoint' % i)

        if paragraph is '':
            collection.append(paragraph)
        else:
            print('Paragraph is a empty')

    elif 'Gutenberg' in line: # Header / footer
        print('Line %d is a header/footer' % i)
    else: # Paragraph
        print('Line %d is a text line' % i)
        paragraph += line


print(collection)
# collection = file.split("\n\n")


# new_collection = filter(lambda k: 'Gutenberg' in k, collection)