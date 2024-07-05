import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize

# Descargar los datos necesarios para tokenizar oraciones
nltk.download('punkt')

# Función para extraer texto de un PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

# Función para dividir texto en oraciones
def split_into_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

# Función para clasificar el texto usando BERTa
def classify_text(text):
    classifier = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment')
    sentences = split_into_sentences(text)
    results = []
    for sentence in sentences:
        short_sentence = sentence[:512]
        result = classifier(short_sentence)[0]
        results.append({
            'Oración': sentence,
            'Etiqueta': result['label'],
            'Confianza': result['score']
        })
    df = pd.DataFrame(results)
    etiquetas_traducidas = {
        'LABEL_0': 'negativo',
        'LABEL_1': 'neutro',
        'LABEL_2': 'positivo'
    }
    df['Sentimiento'] = df['Etiqueta'].map(etiquetas_traducidas)
    return df

# Cargar archivos de resultados
def load_results(file_path):
    return pd.read_excel(file_path)

# Mostrar análisis estadístico de resultados previos
def show_previous_statistics(df):
    st.write(df)
    sentiment_counts = df.groupby('Archivo').sum().reset_index()
    st.write(sentiment_counts)

    for index, row in sentiment_counts.iterrows():
        fig, ax = plt.subplots()
        data = row[['positivo', 'neutral', 'negativo']]
        labels = ['Positivo', 'Neutral', 'Negativo']
        colors = ['green', 'gray', 'red']
        ax.bar(labels, data, color=colors)
        ax.set_ylabel('Número de oraciones')
        ax.set_title(f"Análisis de {row['Archivo']}")
        st.write(f"Análisis de {row['Archivo']}")
        st.pyplot(fig)

# Mostrar análisis estadístico de nuevo análisis
def show_new_statistics(df):
    st.write(df)
    sentiment_counts = df['Sentimiento'].value_counts()
    st.write(sentiment_counts)

    fig, ax = plt.subplots()
    data = sentiment_counts.values
    labels = sentiment_counts.index
    colors = ['red' if label == 'negativo' else 'green' if label == 'positivo' else 'gray' for label in labels]
    ax.bar(labels, data, color=colors)
    ax.set_ylabel('Número de oraciones')
    ax.set_title(f"Análisis del nuevo archivo PDF")
    st.pyplot(fig)

# Título de la aplicación
st.title("Análisis Estadístico de Sentimientos de Textos Jurídicos")

# Mostrar análisis estadístico de archivos cargados
st.header("Análisis Estadístico de Modelos Previamente Ejecutados")
uploaded_file = st.file_uploader("Subir archivo de resultados (.xlsx)", type="xlsx")
if uploaded_file is not None:
    df_results = load_results(uploaded_file)
    show_previous_statistics(df_results)

# Subir y analizar un nuevo archivo PDF
st.header("Análisis de Nuevo Archivo PDF")
uploaded_pdf = st.file_uploader("Subir archivo PDF", type="pdf")
if uploaded_pdf is not None:
    text = extract_text_from_pdf(uploaded_pdf)
    df_new_analysis = classify_text(text)
    show_new_statistics(df_new_analysis)
