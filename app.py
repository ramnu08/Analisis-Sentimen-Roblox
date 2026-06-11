import streamlit as st
import re
import pickle
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

from preprocessing import preprocess_text

# ======================
# CUSTOM CSS
# ======================
st.markdown("""
<style>

/* Tombol sidebar */
section[data-testid="stSidebar"] .stButton > button {

    width: 100%;
    height: 55px;

    border: 1px solid #1E1E1E;
    border-radius: 15px;

    background-color: white;
    color: black;

    font-size: 20px;
    font-weight: bold;

    margin-top: 10px;
}

/* Hover effect */
section[data-testid="stSidebar"] .stButton > button:hover {

    background-color: #4CAF50;
    color: white;

    border: 3px solid white;
}

</style>
""", unsafe_allow_html=True)



# State awal
if "halaman" not in st.session_state:
    st.session_state.halaman = "home"

# Tombol sidebar
if st.sidebar.button("Home"):
    st.session_state.halaman = "home"

if st.sidebar.button("Analisis"):
    st.session_state.halaman = "galeri"



# Load model
with open("modelsvm_sentimen.pkl", "rb") as file:
    model = pickle.load(file)

with open("modelsvm_aspek.pkl", "rb") as file:
    model2 = pickle.load(file)

# Load tfidf
with open("modeltfidf_sentimen.pkl", "rb") as file:
    tfidf = pickle.load(file)

with open("modeltfidf_aspek.pkl", "rb") as file:
    tfidf2 = pickle.load(file)

# -------------------------------
# UI sederhana
# -------------------------------

if st.session_state.halaman == "home":
    
    st.title("Klasifikasi Sentimen Komentar")

    text_input = st.text_input("Input data text")




    if st.button("Proses"):
        if text_input.strip() == "":
            st.warning("Masukkan teks terlebih dahulu")
        else:
            #hasil = predict_sentiment(text_input)
            cleaned = preprocess_text(text_input)

            # tfidf transform
            vector = tfidf.transform([cleaned])
            vector2 = tfidf2.transform([cleaned])

            # prediksi
            hasil = model.predict(vector)
            hasil2 = model2.predict(vector2)

            st.write("Prediksi:", hasil[0], "-", hasil2[0])
            #st.write("Hasil:", hasil)

            st.header("Sentimen")
            if hasil[0] == "positif":
                st.subheader("Positif")
                st.write("Sentimen Positif berasal dari dampak " \
                "positif yang didapatkan atau dialami oleh anak " \
                "yang mempengaruhi psikis dan perkembangan anak. " \
                "Seperti rajin belajar, makin pintar, saling membantu, " \
                "mendapat kawan baru, belajar berkomunikasi, dan lainnya.")

            elif hasil[0] == "netral":
                st.subheader("Netral")
                st.write("Sentimen Netral merupakan penanda bahwa sebuah teks " \
                "tidak memberatkan sentimen positif ataupun negatif, netral bisa juga berisi " \
                "suatu informasi atau teks yang tidak ada sangkut pautnya dalam dampak pada anak")

            elif hasil[0] == "negatif":
                st.subheader("Negatif")
                st.write("Negatif merupakan suatu dampak yang dialami oleh anak yang "
                "dapat mengakibatkan adanya perubahan yang buruk pada anak seperti anak yang "
                "menjadi emosional, penurunan nilai, kurang bersosialisasi, dan lainnya.")


            st.header("Aspek")
            
            if hasil2[0] == "sosial":
                st.subheader("Sosial")
                st.write("Aspek sosial adalah sebuah kategori teks yang memuat tentang dampak " \
                "yang berhubungan dengan pengaruh pada anak dalam bersosialisasi dan berintraksi " \
                "dengan player lain atau dengan teman di lingkungannya.")

            elif hasil2[0] == "keamanan":
                st.subheader("Keamanan")
                st.write("Aspek Keamanan merupakan suatu kategori teks yang membahas tentang suatu konten dalam " \
                "game yang memiliki potensi untuk merusak psikis anak yang bisa bersumber dari isi konten dalam game " \
                "atau juga intraksi anak terhadap user lain yang dapat mempengaruhi anak.")

            elif hasil2[0] == "kognitif":
                st.subheader("Kognitif")
                st.write("Aspek Kognitif merupakan suatu aspek yang menunjukkan perubahan pola pikir atau kecerdasan "
                "anak yang di sebabkan saat bermain bermain game online yang bisa dilihat dari sisi pengetahuan atapun "
                "perilaku anak.")


    

           

if st.session_state.halaman == "galeri":
    st.header("Persentase Sentiment")
    st.image("persentaseSentimen.png")

    # Memberi jarak ke bawah
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.header("Persentase Aspek")
    st.image("persentaseAspek.png")
