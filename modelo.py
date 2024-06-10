import os
import PyPDF2
from transformers import pipeline

# Definir la ruta a la carpeta que contiene los archivos PDF
pdf_folder = 'textos_juridicos'

# Crear un pipeline de análisis de sentimientos
classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

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

# Obtener una lista de todos los archivos PDF en la carpeta
pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

# Leer y clasificar cada archivo PDF
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_folder, pdf_file)
    print(f'Procesando archivo: {pdf_file}')
    
    # Extraer texto del PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Clasificar el texto
    results = classifier(text[:512])  # Limitar el texto a los primeros 512 caracteres para la clasificación
    
    # Mostrar resultados
    for result in results:
        print(f'Texto: {text[:100]}...')  # Mostrar solo los primeros 100 caracteres del texto
        print(f'Predicción: {result}\n')

print("Clasificación completada.")



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