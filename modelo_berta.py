import os
import PyPDF2
import pandas as pd
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

# Descargar los datos necesarios para tokenizar oraciones
nltk.download('punkt')

# Definir la ruta a la carpeta que contiene los archivos PDF
pdf_folder = 'textos_juridicos'

# Crear un pipeline de análisis de sentimientos con el modelo específico para sentimientos (positivo, negativo, neutro)
classifier = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment')

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

# Obtener una lista de todos los archivos PDF en la carpeta
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

# Crear una lista para almacenar los resultados
results_list = []

# Leer y clasificar cada archivo PDF
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    print(f'Procesando archivo: {pdf_file}')
    
    # Extraer texto del PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Dividir el texto en oraciones
    sentences = split_into_sentences(text)
    
    # Clasificar cada oración
    for sentence in sentences:
        # Limitar la oración a los primeros 512 caracteres para la clasificación
        short_sentence = sentence[:512]
        result = classifier(short_sentence)[0]
        
        # Añadir el resultado a la lista
        results_list.append({
            'Archivo': pdf_file,
            'Oración': short_sentence,
            'Etiqueta': result['label'],
            'Confianza': result['score']
        })

# Crear un DataFrame de pandas con los resultados
df = pd.DataFrame(results_list)

# Guardar el DataFrame en un archivo CSV
output_csv_path = 'resultados_clasificacion.csv'
df.to_csv(output_csv_path, index=False)

# Crear un nuevo DataFrame para las estadísticas por archivo
summary_list = []

for pdf_file in pdf_files:
    file_df = df[df['Archivo'] == pdf_file]
    label_counts = file_df['Etiqueta'].value_counts()
    total_labels = label_counts.sum()
    label_0_count = label_counts.get('LABEL_0', 0)
    label_1_count = label_counts.get('LABEL_1', 0)
    label_2_count = label_counts.get('LABEL_2', 0)
    label_1_percentage = (label_1_count / total_labels) * 100 if total_labels > 0 else 0
    
    summary_list.append({
        'Archivo': pdf_file,
        'Label_0': label_0_count,
        'Label_1': label_1_count,
        'Label_2': label_2_count,
        'Porcentaje_Label_1': label_1_percentage
    })

# Crear un DataFrame de pandas con el resumen
summary_df = pd.DataFrame(summary_list)

# Guardar el DataFrame de resumen en un archivo CSV
summary_csv_path = 'resumen_clasificacion.csv'
summary_df.to_csv(summary_csv_path, index=False)

print(f"Clasificación completada. Los resultados se han guardado en {output_csv_path}.")
print(f"Resumen completado. El resumen se ha guardado en {summary_csv_path}.")
