import os
import PyPDF2
import pandas as pd
from transformers import  BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from nltk.tokenize import sent_tokenize


# Inicializa y entrena tu modelo de BERT
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Después de entrenar el modelo, guarda el modelo y el tokenizador
model.save_pretrained('modelo/bert_model')
tokenizer.save_pretrained('modelo/bert_model')

# Cargar el modelo BERT entrenado y el tokenizador correspondiente
tokenizer = BertTokenizer.from_pretrained('modelo/bert_model')
model = TFBertForSequenceClassification.from_pretrained('modelo/bert_model')

def extract_text_from_pdf(pdf_path):
    """
    Extrae el texto de un archivo PDF.
    """
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text

def split_into_sentences(text):
    """
    Divide el texto en oraciones.
    """
    sentences = sent_tokenize(text)
    return sentences

def classify_text(model, tokenizer, text):
    """
    Clasifica un texto en positivo, neutro o negativo usando el modelo BERT.
    """
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)
    outputs = model(inputs)
    predictions = tf.nn.softmax(outputs.logits, axis=-1)
    label = tf.argmax(predictions, axis=1).numpy()[0]
    score = predictions[0][label].numpy()
    return {'label': f'LABEL_{label}', 'score': score}

# Carpeta donde se encuentran los archivos PDF
pdf_folder = 'textos_juridicos'
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

# Lista para almacenar los resultados de la clasificación
results_list = []

# Procesar cada archivo PDF en la carpeta
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    print(f'Procesando archivo: {pdf_file}')
    
    text = extract_text_from_pdf(pdf_path)
    sentences = split_into_sentences(text)
    
    # Clasificar cada oración en el PDF
    for sentence in sentences:
        short_sentence = sentence[:512]
        result = classify_text(model, tokenizer, short_sentence)
        results_list.append({
            'Archivo': pdf_file,
            'Oración': sentence,
            'Etiqueta': result['label'],
            'Confianza': result['score']
        })

# Crear un DataFrame con los resultados y traducir las etiquetas
df = pd.DataFrame(results_list)
etiquetas_traducidas = {
    'LABEL_0': 'positivo',
    'LABEL_1': 'neutro',
    'LABEL_2': 'negativo'
}
df['Sentimiento'] = df['Etiqueta'].map(etiquetas_traducidas)

# Guardar los resultados en un archivo Excel
output_xlsx = 'resultados_clasificacion.xlsx'
df.to_excel(output_xlsx, index=False)

# Crear un resumen de la clasificación
summary_list = []
for pdf_file in pdf_files:
    file_df = df[df['Archivo'] == pdf_file]
    label_counts = file_df['Etiqueta'].value_counts()
    total_labels = label_counts.sum()
    label_0_count = label_counts.get('LABEL_0', 0)
    label_1_count = label_counts.get('LABEL_1', 0)
    label_2_count = label_counts.get('LABEL_2', 0)
    label_0_percentage = (label_0_count / total_labels) * 100 if total_labels > 0 else 0
    label_1_percentage = (label_1_count / total_labels) * 100 if total_labels > 0 else 0
    label_2_percentage = (label_2_count / total_labels) * 100 if total_labels > 0 else 0
    
    summary_list.append({
        'Archivo': pdf_file,
        'positivo': label_0_count,
        'neutral': label_1_count,
        'negativo': label_2_count,
        'Total etiquetas': total_labels,
        'Porcentaje positivo': label_0_percentage,
        'Porcentaje neutral': label_1_percentage,
        'Porcentaje negativo': label_2_percentage,
    })

# Guardar el resumen en un archivo Excel
summary_df = pd.DataFrame(summary_list)
summary_excel = 'resumen_clasificacion.xlsx'
summary_df.to_excel(summary_excel, index=False)

print(f"Clasificación completada. Los resultados se han guardado en {output_xlsx}.")
print(f"Resumen completado. El resumen se ha guardado en {summary_excel}.")
