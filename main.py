# SRI AHMAD TSAQIF
# AKIP TSAQIF

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('expand_frame_repr', False)

# Example words
d1 = 'Shipment of gold damaged in a fire'
d2 = 'Delivery of silver arrived in a silver truck'
d3 = 'Shipment of gold arrived in a truck'
d4 = 'gold silver truck'
d5 = 'gold truck'

stop = stopwords.words('english')

# Calculate TF-IDF using TfidVectorizer from SkLearn
vectorizer = TfidfVectorizer(stop_words=stop)
vectors = vectorizer.fit_transform([d1, d2, d3, d4, d5])

names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()

df2 = pd.DataFrame(denselist, columns=names)
df2.index = ['D1', 'D2', 'D3', 'D4', 'D5']

print()
print(df2)

print()
print('\tBased on the data above, we can calculate the similarity scores')
print('\tFor (1) "gold silver truck", its similarity scores is:')

goldQ = df2.at['D4', 'gold']
goldD = df2.at['D1', 'gold']
silverQ = df2.at['D4', 'silver']
silverD = df2.at['D2', 'silver']
truckQ = df2.at['D4', 'truck']
truckD = df2.at['D3', 'truck']

x = goldQ * goldD
y = silverQ * silverD + truckQ * truckD
z = truckQ * truckD + goldQ * goldD

skor1 = []
skor1.extend((x, y, z))

dfskor = pd.DataFrame(skor1)
dfskor.columns = ['Score']
dfskor.index = ['D1', 'D2', 'D3']

print(dfskor)
print('\tSo, their rank is:')
print(dfskor.sort_values(by = 'Score', ascending=False))

print()
print('\tAnd for (2) "gold truck", its similarity scores is:')

goldQ2 = df2.at['D5', 'gold']
goldD2 = df2.at['D1', 'gold']
truckQ2 = df2.at['D5', 'truck']
truckD2 = df2.at['D3', 'truck']

x2 = goldQ2 * goldD2
y2 = truckQ2 * truckD2
z2 = x2 + y2

skor2 = []
skor2.extend((x2, y2, z2))

dfskor2 = pd.DataFrame(skor2)
dfskor2.columns = ['Score']
dfskor2.index = ['D1', 'D2', 'D3']

print(dfskor2)
print('\tSo, their rank is:')
print(dfskor2.sort_values(by = 'Score', ascending=False))



# ======================================================
# From here, these are the manual calculations of TF-IDF
# ======================================================
d1v = d1.split(' ')
d2v = d2.split(' ')
d3v = d3.split(' ')

unik = set(d1v).union(set(d2v).union(set(d3v)))
jumlah_d1v = dict.fromkeys(unik, 0)
jumlah_d2v = dict.fromkeys(unik, 0)
jumlah_d3v = dict.fromkeys(unik, 0)

for kata in d1v:
    jumlah_d1v[kata] += 1

for kata in d2v:
    jumlah_d2v[kata] += 1

for kata in d3v:
    jumlah_d3v[kata] += 1

# Find TF per words
def TF(kamus, kumpulan):
    kamusTF = {}
    panjang = len(kumpulan)
    for kata, total in kamus.items():
        kamusTF[kata] = total / float(panjang)
    return kamusTF

# Find TF from each documents
tfd1 = TF(jumlah_d1v, d1v)
tfd2 = TF(jumlah_d2v, d2v)
tfd3 = TF(jumlah_d3v, d3v)

# Find IDF per words
def IDF(dokumen):
    import math
    pan = len(dokumen)

    kamusIDF = dict.fromkeys(dokumen[0].keys(), 0)
    for dok in dokumen:
        for kata, nilai in dok.items():
            if nilai > 0:
                kamusIDF[kata] += 1

    for kata, nilai in kamusIDF.items():
        kamusIDF[kata] = math.log(pan / float(nilai))
    return kamusIDF

# Find IDF in those documents
idf = IDF([jumlah_d1v, jumlah_d2v, jumlah_d3v])

# Calculate TF-IDF
def hitungTFIDF(TF, IDF):
    tfidf = {}
    for kata, nilai in TF.items():
        tfidf[kata] = nilai * IDF[kata]
    return tfidf

# Finally lets see the total TF-IDF from those documents
tfidfd1 = hitungTFIDF(tfd1, idf)
tfidfd2 = hitungTFIDF(tfd2, idf)
tfidfd3 = hitungTFIDF(tfd3, idf)

df = pd.DataFrame([tfidfd1, tfidfd2, tfidfd3])
# print(df)
