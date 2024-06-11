import os
import PyPDF2
import pandas as pd
from transformers import pipeline

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

def split_into_paragraphs(text):
    """
    Divide el texto en párrafos.
    """
    paragraphs = text.split('\n')
    paragraphs = [p for p in paragraphs if p.strip() != '']
    return paragraphs

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
    
    # Dividir el texto en párrafos
    paragraphs = split_into_paragraphs(text)
    
    # Clasificar cada párrafo
    for paragraph in paragraphs:
        # Limitar el párrafo a los primeros 512 caracteres para la clasificación
        short_paragraph = paragraph[:]
        result = classifier(short_paragraph)[0]
        
        # Añadir el resultado a la lista
        results_list.append({
            'Archivo': pdf_file,
            'Párrafo': short_paragraph,
            'Etiqueta': result['label'],
            'Confianza': result['score']
        })

# Crear un DataFrame de pandas con los resultados
df = pd.DataFrame(results_list)

# Guardar el DataFrame en un archivo CSV
output_csv_path = 'resultados_clasificacion.csv'
df.to_csv(output_csv_path, index=False)

print(f"Clasificación completada. Los resultados se han guardado en {output_csv_path}.")



# import fitz  # PyMuPDF

# # Abre el documento PDF
# import os

# # Define la ruta del documento PDF
# carpeta = 'textos_juridicos'
# archivo = '128-2023 (MEcheverria) TIE Hamilton Sumarán y José Juypa.pdf'
# ruta_pdf = os.path.join(carpeta, archivo)

# document = fitz.open(ruta_pdf)

# # Extrae texto de cada página
# for page_num in range(document.page_count):
#     page = document.load_page(page_num)
#     text = page.get_text('text')
#     print(f"Texto de la página {page_num + 1}:\n{text}\n")

# # Cierra el documento
# document.close()