# Import Dictionary
import gensim
import nltk
from gensim import corpora, models, similarities
import pandas as pd

def avg_sim(song, sims, dictionary, tf_idf):
    query_doc = [w.lower() for w in nltk.word_tokenize(song)]
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]
    similarities = sims[query_doc_tf_idf]
    avg_sim = sum(similarities)/len(similarities)
    return avg_sim

#lyrics: corpus of an artist's songs
#song: song that will be compared against an the corpus
def get_sim(lyrics, song, save_corpus=True):
    tok_corpus = [nltk.word_tokenize(str(s)) for s in lyrics]
    dictionary = corpora.Dictionary(tok_corpus)
    corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in tok_corpus]
    if save_corpus:
        print('saving corpus...')
        corpora.MmCorpus.serialize('bow_taylor.mm', corpus)
    tf_idf = gensim.models.TfidfModel(corpus)
    sims = gensim.similarities.Similarity('',tf_idf[corpus], num_features=len(dictionary))
    return avg_sim(song, sims, dictionary, tf_idf)


#TEST
#df = pd.read_csv('../data/taylor_swift.csv')
#lyrics = list(df.Lyrics)
#song = """Right these are the hands of fate you re my achilles heel this is the golden age of something good and right and real and i never saw you coming and i ll never be the same and i never my clear and and i never t never says a with i field beat got a ruthless as when made football ran yeah you knows down miss it mess came it motion yeah, way we was was t out ey i, get a wanted too fun better today she it hey! he’s so never hold loving baby dwarfs says next who why forever feel in in once, night it and goes but until him bright took when about him oh, enemies wonderland works wrapped younger t known up) let your high the goes hope i never m they then us forever best she fake let mother never how saddest would love we stand dreams made you re the one to been (again!)"""
#print(get_sim(lyrics, song, save_corpus=False))
    




