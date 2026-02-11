#pip install --upgrade gensim #--> se non hai la libreria installata (installi da terminale)
#conda install -c conda-forge gensim --> se lavori in un ambiente CONDA usa il comando da terminale
# oppure (per conda) installa da conda navigator
from tqdm import tqdm 
import csv
import gensim
from gensim.models import Word2Vec # serve per continuare l'addestramento del modello o addestrare da zero
from gensim.models import KeyedVectors # serve per usare il modello pre-addestrato se salvato sul pc (file .bin o .kv)
import gensim.downloader as api # per scaricare e caricare in python il modello di w2vec preaddestrato

print(list(gensim.downloader.info()['models'].keys())) 

# model_w2vec = api.load("word2vec-google-news-300")
model_glove = api.load('glove-wiki-gigaword-300')

# # salvo il modello scaricato in formato gensim (pickle) --> per usarlo essenzialmente con la libreria gensim
# model_w2vec.save("word2vec-google-news-300.kv") 
model_glove.save("glove-wiki-gigaword-300.kv") # è un oggetto KeyedVectors: memorizza mappa parola→indice (vocabolario/metadata) e
#                             # e i vettori. Crea un file .npy che è la matrice dei vettori salvata in formato numpy
# # salvo nel formato word2vec binario (compatibile con molte librerie)
# # nel caso mi servisse il modello completo (ad esempio continuare l'addestramento)
# model_w2vec.save_word2vec_format("google_news_300.bin", binary=True)

# #######################################################################

# # Se il modello è salvato in locale su disco con nome "google_news_300.kv"
# # ovviamente mettere il path completo se non si trova nella stessa directory dello script python
# model_w2vec = KeyedVectors.load("word2vec-google-news-300.kv", mmap='r')
model_glove = KeyedVectors.load("glove-wiki-gigaword-300.kv", mmap='r')  # occupa meno RAM

categories = {
        "AD": ["Oxygen",
"Asthma",
"Malaria",
"East",
"Birthday",
"Divorce",
"Christmas",
"Orgasm",
"Atom",
"Rape",
"Price",
"Earthquake",
"Year",
"Suicide",
"Abortion",
"Death",
"Winter",
"Morning",
"Murderer",
"Shiver",
"Equator",
"Definition",
"Wedding",
"Funeral",
"Cancer",
"Cousin",
"Headache",
"Spouse",
"Autumn",
"West",
"Employment",
"Evening",
"Friar",
"Prey",
"Subtraction",
"Day",
"Recycling",
"Allergy",
"Hernia",
"Baptism",
"Password",
"Slaughter"],
        "AV": ["Life",
"Utopia",
"Beauty",
"Concept",
"Existence",
"Love",
"Belief",
"Ethics",
"Space",
"Idea",
"Fun",
"Dream",
"Power",
"Trauma",
"Fantasy",
"Personality",
"Identity",
"Humor",
"Pain",
"Paradise",
"Value",
"Religion",
"Faith",
"Drama",
"Fate",
"Spirit",
"Affection",
"Intellect",
"Rule",
"Crisis",
"Oblivion",
"Failure",
"Matter",
"Savior",
"Pressure",
"Excuse",
"Weakness",
"Mind",
"Success",
"Hardship",
"Wonder",
"Grief"],
        "CD": ["Artichoke",
"Ashtray",
"Banana",
"Basil",
"Cactus",
"Carrot",
"Cat",
"Cellphone",
"Cocoa",
"Dandruff",
"Dentist",
"Elbow",
"Elephant",
"Flamingo",
"Garlic",
"Hamburger",
"Hippopotamus",
"Kangaroo",
"Knee",
"Lemon",
"Onion",
"Penguin",
"Piano",
"Pigeon",
"Pineapple",
"Pizza",
"Rabbit",
"Rhinoceros",
"Saxophone",
"Spinach",
"Strawberry",
"Sunset",
"Tiger",
"Toothpaste",
"Tortoise",
"Tricycle",
"Vagina",
"Violin",
"Woodpecker",
"Yacht",
"Algae",
"Pomegranate"],
        "CV": ["Art",
"Mate",
"Treasure",
"Cob",
"Web",
"Home",
"Bottom",
"World",
"Ruler",
"Goods",
"Novel",
"Back",
"Gift",
"Peak",
"Light",
"Hooker",
"Slum",
"Lead",
"Family",
"Spark",
"Hail",
"Dirty",
"Trash",
"Group",
"Beast",
"Frame",
"Prize",
"Processor",
"Hardware",
"Tie",
"Game",
"Blaze",
"Piece",
"Garbage",
"Sling",
"Sex",
"Filth",
"Board",
"Material",
"Machine",
"Tip",
"Ground"]
    }

print('starting generation')
with open('similar_words_glove.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Category', 'Word', 'Association', 'Similarity'])

    # 3. Process and Write Rows
    for cat, words in categories.items():
        for word in tqdm(words):
            word_lower = word.lower()

            if word_lower in model_glove:
                top20000 = model_glove.most_similar(word_lower, topn=20000)
                for association, similarity in tqdm(top20000):
                        writer.writerow([cat, word_lower, association, similarity])


# # vettore embedded di una parola
# # restituisce un vettore con 300 elementi, che è la dimensione dell'embedding
# vec = model_w2vec['king']    
# print(vec)       # è array a (300-d)

# #parole più simili a una parola target ("dog", nell'esempio) --> il comando che dovresti usare
# top10 = model_w2vec.most_similar("dog", topn=10) # se si vogliono più o meno parole simili si cambia topn

# # quanto due parole sono (semanticamente) simili (la metrica è il coseno dell'angolo tra i vettori delle due parole)
# sim = model_w2vec.similarity('king', 'queen') # restituisce un numero tra 0 e 1 (parole molto simili)

# # verificare se la parola è nel vocabolario (gensim >=4)
# 'parola_da_verificare' in model_w2vec.key_to_index   # La risposta è booleana: True/False

# # ottenere dimensione del modello e il suo vocabolario
# vec_size = model_w2vec.vector_size          # dimensione del modello (cioè dei vettori embedded)
# vocab_size = len(model_w2vec.key_to_index) # dimensione del vocabolario





    