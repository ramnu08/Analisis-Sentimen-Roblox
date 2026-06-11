import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

# -------------------------------
# 1. Setup Stemmer & Stopwords
# -------------------------------
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))
stop_words.update(["salah_satu", "woi", "sumpah", "nya", "bro", "buat", "gituloh", "huhuhu", "saji", 
                   "aduh", "dah", "wkwk", "sih", "miin", "ne", "tinggal", "eeh", "bener", "kayak"]) #menambah kata untuk di hapus
for kata in ["dua", "banyak", "seharusnya", "perhatikan", "pengguna", "selain", "lain", "perlu", 
             "lebih", "kurang", "lama", "panjang", "kata", "bisa"]: #menghapus kata agar tidak dihapus
    if kata in stop_words:
        stop_words.remove(kata)

# -------------------------------
# 2. Kamus slang & typo correction
# -------------------------------
slang_dict = {
    "gk": "gak",
    "ga": "gak",
    "ngga": "gak",
    "nggak": "gak",
    "tidak": "gak",
    "bgt": "banget",
    "tp": "tapi",
    "yg": "yang",
    "dlm": "dalam",
    "gamnya": "gamenya",
    "toxickasar": "toxic kasar",
    "cuman" : "cuma",
    "gw" : "aku",
    "gua": "aku",
    "doang" : "aja",
    "gaje" : "tidak_jelas",
    "kedua": "dua",
    "bet": "banget",
    "smasederajat": "sma sederajat",
    "buanyaaakkk": "banyak",
    "diceritain" : "diceritakan",
    "aja": "saja",
    "offer": "over",
    "dissapointing": "disappointing",
    "cutsene": "cut scene",
    "gamplaynya": "gameplaynya",
    "kureng": "kurang",
    "ggwp" : "good game, well played",
    "geloo": "gila",
    "gane": "game",
    "larakternya": "karakternya",
    "sesua": "sesuai",
    "kerenn": "keren",
    "mantul": "mantap_betul",
    "kuranag": "kurang",
    "mainin": "mainkan",
    "banyakny": "banyaknya",
    "mc": "main character",
    "fck": "fuck"
}

# -------------------------------
# 3. Exception untuk stemming
# -------------------------------
exception_dict = {
    "berasa": "rasa",   # jangan jadi "asa"
    "ngegas": "gas",
    "menarik": "menarik",
    "gini" : "begini",
    "seharusnya": "seharusnya",
    "diperhatikan": "perhatikan",
    "penggunaannya": "pengguna",
    "terkesan": "kesan",
    "memperbaiki": "perbaiki",
    "sekedar" : "sekedar",
    "berlebihan": "berlebihan",
    "terdengar": "terdengar",
    "pertarungan": "pertarungan",
    "diikuti": "diikuti",
    "pengisi": "pengisi",
    "pergerakan": "gerak",
    "setuju": "setuju",
    "semoga": "semoga",
    "diperbaiki": "perbaiki",
    "pentolan": "pentolan"
}

# -------------------------------
# 4. Kata sentimen (lexicon)
# -------------------------------
positive_words = ["bagus", "baik", "keren", "indah", "seru", "tepat", "pas", "banyak", "jelas", "mau"]
negative_words = ["jelek", "buruk", "parah", "cringe", "letoy", "sedikit"]

# -------------------------------
# 5. Normalisasi slang
# -------------------------------
def normalize_slang(tokens):
    return [slang_dict.get(word, word) for word in tokens]

def protect_phrases(text):
    text = text.replace("salah satu", "salah_satu")
    return text

# -------------------------------
# 5b. Normalisasi kata asing dengan sufiks (game, grafik, level, dll)
# -------------------------------
def normalize_foreign_suffix(tokens):
    new_tokens = []
    for word in tokens:
        if word.endswith(("nya","ku","mu")):
            if word.startswith(("game","grafik","level","control", "story", "minigame", "dubbing", 
                                "script", "system", "gameplay", "action", "bug")):
                # buang sufiks
                if word.endswith("nya"):
                    new_tokens.append(word[:-3])
                elif word.endswith("ku") or word.endswith("mu"):
                    new_tokens.append(word[:-2])
                else:
                    new_tokens.append(word)
            else:
                new_tokens.append(word)
        else:
            new_tokens.append(word)
    return new_tokens

# -------------------------------
# 6. Rule-based negasi
# -------------------------------
def handle_negation(tokens):
    new_tokens = []
    skip = False
    for i in range(len(tokens)):
        if skip:
            skip = False
            continue
        if tokens[i] in ["tidak", "gak"]:
            if i+1 < len(tokens) and (tokens[i+1] in positive_words or tokens[i+1] in negative_words):
                new_tokens.append("tidak_" + tokens[i+1])
                skip = True
            else:
                new_tokens.append(tokens[i])
        else:
            if tokens[i] == "kurang" and i+1 < len(tokens):
                next_word = tokens[i+1]
                if next_word in positive_words:
                    new_tokens.append("kurang_" + next_word)  # negatif
                    skip = True
                elif next_word in negative_words:
                    new_tokens.append("kurang_" + next_word)  # bisa ditandai netral/positif
                    skip = True
                else:
                    new_tokens.append(tokens[i])  
            else:
                new_tokens.append(tokens[i])          
    return new_tokens

# -------------------------------
# 7. Custom stemming
# -------------------------------
def custom_stemming(tokens):
    result = []
    for word in tokens:
        if word in exception_dict:
            result.append(exception_dict[word])
        else:
            result.append(stemmer.stem(word))
    return result

# -------------------------------
# 8. Preprocessing pipeline
# -------------------------------
def preprocess_text(text):
    # Lowercase
    text = text.lower()

    text = text.replace("-", " ")
    text = text.replace(",", " ")
    text = text.replace("/", " ")
    # Hapus angka & simbol
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    text = protect_phrases(text)
    # Tokenisasi
    tokens = text.split()
    # Normalisasi slang
    tokens = normalize_slang(tokens)
    # Normalisasi kata asing dengan sufiks
    tokens = normalize_foreign_suffix(tokens)
    # Rule-based negasi (sebelum stopword & stemming)
    tokens = handle_negation(tokens)
    # Hapus stopwords (jangan buang 'tidak' karena penting)
    tokens = [word for word in tokens if word not in stop_words or word.startswith("tidak_")]
    # Custom stemming
    tokens = custom_stemming(tokens)
    return " ".join(tokens)

