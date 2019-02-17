# Import Dictionary
import gensim
import nltk
from gensim import corpora, models, similarities
import pandas as pd

kanye = pd.read_excel('kanye_lyrics.xlsx')
kanye_lyrics = list(kanye.Lyrics)
tok_corpus = [nltk.word_tokenize(str(song)) for song in kanye_lyrics]

dictionary = corpora.Dictionary(tok_corpus)


print(dictionary)

print('dict')


corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in tok_corpus]
print(corpus)

print('saving corpus')
corpora.MmCorpus.serialize('bow_kanye.mm', corpus)



tf_idf = gensim.models.TfidfModel(corpus)
print(tf_idf)

s = 0
for i in corpus:
    s += len(i)
print(s)

sims = gensim.similarities.Similarity('',tf_idf[corpus],
                                      num_features=len(dictionary))
print(sims)
print(type(sims))

test = """Kanye, can I talk to you for a minute? Me and the other faculty members was wonderin' could you do a lil som... Somethin' beautiful, somethin' that the kids is gon' love when they hear it. Tha's gon make them start jumpin' up and down and sharin' candy an' stuff. Think you could probably do somethin' for the kids for graduation to sing?"""



query_doc = [w.lower() for w in nltk.word_tokenize(test)]
query_doc_bow = dictionary.doc2bow(query_doc)
query_doc_tf_idf = tf_idf[query_doc_bow]

similarities = sims[query_doc_tf_idf]
avg_sim = sum(similarities)/len(similarities)
print(avg_sim)