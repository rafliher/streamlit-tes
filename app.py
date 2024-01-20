import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model('my_model.h5')

# Load Tokenizer
tokenizer = Tokenizer(num_words=5000, oov_token='<oov>')
tokenizer.fit_on_texts([])  # Tidak perlu fit_on_texts pada streamlit

def predict_sentiment(text):
    # Lakukan pra-pemrosesan
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=20, truncating='post')

    # Lakukan prediksi
    result = model.predict(padded)

    # Tentukan threshold dan interpretasikan hasil prediksi
    threshold = 0.5
    sentiment = "Positif" if result[0][0] > threshold else "Negatif"

    return sentiment

def main():
    st.title("Analisis Sentimen dengan Streamlit")

    # Tambahkan input teks dari pengguna
    user_input = st.text_area("Masukkan teks untuk analisis sentimen:")

    if st.button("Analisis"):
        # Panggil fungsi prediksi
        result = predict_sentiment(user_input)

        # Tampilkan hasil
        st.write("Sentimen Prediksi:", result)

if __name__ == "__main__":
    main()
