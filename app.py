import streamlit as st
import numpy as np
import pandas as pd
import re
import string
import pickle
import nltk
from nltk.corpus import stopwords
import Sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.write("""
# Klaster Anggota App
Aplikasi ini merupakan implementasi sederhana _machine learning_. Pembuatan model berdasarkan data alasan masuk anggota pada tahun 2017 dan 2020. Berdasarkan model terdapat 10 klaster, yaitu:  

1. `Belajar Usaha`  
2. `Belajar Berwirausaha`  
3. `Belajar Keuangan`  
4. `Belajar Interaksi dengan Orang`  
5. `Belajar Bisnis`  
6. `Menambah Pengalaman`  
7. `Mengembangkan Jiwa Kewirausahaan`  
8. `Belajar Terkait Koperasi`  
9. `Tertarik Internal Kopma`  
10. `Mencari Pengalaman`  
Aplikasi akan mengolah input berupa alasan yang diberikan untuk memprediksi klaster yang sesuai berdasarkan model yang telah diperoleh.
""")

stp_all = pickle.load(open('stopwords.pkl', 'rb'))

def remove_stopwords(x):    
    x = x.split(' ')    
    for word in x:
        if word in stp_all:
            x.remove(word)    
    x = ' '.join(x)
    return x

def bersihin_teks(teks):  
    teks = teks.lower().strip() # case folding, trim whitespace
    teks = re.sub(r'\d+', ' ', teks) # menghapus angka
    teks = re.sub('-', ' ', teks) # memisahkan kata ulang  
    teks = teks.translate(str.maketrans(' ',' ',string.punctuation)) # menghapus tanda baca
    return teks

stemmer = StemmerFactory().create_stemmer()

kat_cluster = {0:'Belajar Usaha',
               1:'Belajar Berwirausaha',
               2:'Belajar Keuangan',
               3:'Belajar Interaksi dengan Orang',
               4:'Belajar Bisnis',
               5:'Menambah Pengalaman',
               6:'Mengembangkan Jiwa Kewirausahaan',
               7:'Belajar Terkait Koperasi',
               8:'Tertarik Internal Kopma',
               9:'Mencari Pengalaman'}

with st.form('Form Alasan Masuk'):
	st.markdown('### Silakan masukkan alasan kamu')
	user_input = st.text_area(' ', ' ')
	
	submitted = st.form_submit_button('Cek Prediksi Cluster')	
	
	if submitted:
		sample_input = pd.DataFrame({'alasan': [user_input]})

		sample_input['alasan'] = sample_input['alasan'].apply(lambda x:bersihin_teks(x))
		sample_input['alasan'] = sample_input['alasan'].apply(lambda x:remove_stopwords(x))
		sample_input['alasan'] = sample_input['alasan'].apply(lambda x:remove_stopwords(x))
		sample_input['alasan'] = sample_input['alasan'].apply(lambda x:stemmer.stem(x))

		fitur = pickle.load(open('fitur.pkl', 'rb'))
		fitur = {k: v for k, v in sorted(fitur.items(), key=lambda item: item[1])}
		tes_alasan_tf_idf = TfidfVectorizer(max_features=50, vocabulary=fitur)
		tes_alasan_tf_idf_matriks = tes_alasan_tf_idf.fit_transform(sample_input['alasan']) #pickle-in
		tes_alasan_tf_idf_df = pd.DataFrame(tes_alasan_tf_idf_matriks.toarray(), columns=fitur)
		tes_alasan_tf_idf_df.index = sample_input.index

		load_kmeans_alasan = pickle.load(open('kmeans_alasan.pkl', 'rb'))
		cluster = load_kmeans_alasan.predict(tes_alasan_tf_idf_df)# Add to the dataframe and show the result
		sample_input['cluster'] = cluster
		sample_input['cluster'] = sample_input['cluster'].replace(kat_cluster)
		
		st.markdown('#### Cluster Kamu:')
		st.info(sample_input.iloc[0]['cluster'])
