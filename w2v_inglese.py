#pip install --upgrade gensim #--> se non hai la libreria installata (installi da terminale)
#conda install -c conda-forge gensim --> se lavori in un ambiente CONDA usa il comando da terminale
# oppure (per conda) installa da conda navigator

import gensim
from gensim.models import Word2Vec # serve per continuare l'addestramento del modello o addestrare da zero
from gensim.models import KeyedVectors # serve per usare il modello pre-addestrato se salvato sul pc (file .bin o .kv)
import gensim.downloader as api # per scaricare e caricare in python il modello di w2vec preaddestrato

print(list(gensim.downloader.info()['models'].keys())) # restituisce la lista di modelli disponibili (tra cui anche glove)
# Esempio: ['word2vec-ruscorpora-300' --> w2vec sulla lingua russa,
# 'word2vec-google-news-300' --> w2vec originale
# 'glove-wiki-gigaword-300' --> glove addestrato con dati di wikipedia e gigaword
# 'glove-twitter-200' --> glove addestrato con dati di twitter (messaggi brevi emoticons e hashtag]
#

# Scarico dal repository il modello pre-addestrato e lo carico nello script di python
#questo modello è scaricato e salvato nella cache di gensim e fisicamente nella cartella dello script
model_w2vec = api.load("word2vec-google-news-300")
#model_glove = api.load('glove-wiki-gigaword-300') # --> per scaricare il modello di glove


# salvo il modello scaricato in formato gensim (pickle) --> per usarlo essenzialmente con la libreria gensim
model_w2vec.save("google_news_300.kv") # è un oggetto KeyedVectors: memorizza mappa parola→indice (vocabolario/metadata) e
                            # e i vettori. Crea un file .npy che è la matrice dei vettori salvata in formato numpy
# salvo nel formato word2vec binario (compatibile con molte librerie)
# nel caso mi servisse il modello completo (ad esempio continuare l'addestramento)
model_w2vec.save_word2vec_format("google_news_300.bin", binary=True)

#######################################################################

# Se il modello è salvato in locale su disco con nome "google_news_300.kv"
# ovviamente mettere il path completo se non si trova nella stessa directory dello script python
model_w2vec = KeyedVectors.load("google_news_300.kv", mmap='r')  # occupa meno RAM


# vettore embedded di una parola
# restituisce un vettore con 300 elementi, che è la dimensione dell'embedding
vec = model_w2vec['king']           # è array a (300-d)

# analogie --> re sta a uomo come donna sta a?
analogie_top10 = model_w2vec.most_similar(positive=['king', 'woman'], negative=['man'], topn=10)

#parole più simili a una parola target ("dog", nell'esempio) --> il comando che dovresti usare
top10 = model_w2vec.most_similar("dog", topn=10) # se si vogliono più o meno parole simili si cambia topn

# quanto due parole sono (semanticamente) simili (la metrica è il coseno dell'angolo tra i vettori delle due parole)
sim = model_w2vec.similarity('king', 'queen') # restituisce un numero tra 0 e 1 (parole molto simili)

# verificare se la parola è nel vocabolario (gensim >=4)
'parola_da_verificare' in model_w2vec.key_to_index   # La risposta è booleana: True/False

# ottenere dimensione del modello e il suo vocabolario
vec_size = model_w2vec.vector_size          # dimensione del modello (cioè dei vettori embedded)
vocab_size = len(model_w2vec.key_to_index) # dimensione del vocabolario
