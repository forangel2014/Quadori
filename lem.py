import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

wnl = WordNetLemmatizer()

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r', 'VBD':'v'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' 

word = 'better'#'participated'
tag = pos_tag([word])
print(tag)
print(wnl.lemmatize(word, penn2morphy(tag)))