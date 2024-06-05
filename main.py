import streamlit as st
import pandas as pd
import requests
from google.cloud import vision
import io
import openai

# Títulos y descripciones de la aplicación
st.title("Verificación Aleatoria de Casillas Electorales")
st.write("Esta aplicación realiza una verificación aleatoria de casillas electorales en México para asegurar la integridad de los resultados electorales de las elecciones presidenciales de 2024.")

# Ingreso de claves API
google_api_key = st.text_input("Ingresa tu clave API de Google Cloud Vision", type="password")
openai_api_key = st.text_input("Ingresa tu clave API de OpenAI", type="password")

# Cargar archivos CSV
st.write("Cargando datos de los archivos CSV...")
file_path_pres = 'PRES_2024.csv_exported.csv'
file_path_candidaturas = 'PRES_CANDIDATURAS_2024.csv'

@st.cache_data
def cargar_datos():
    pres_data = pd.read_csv(file_path_pres)
    candidaturas_data = pd.read_csv(file_path_candidaturas)
    return pres_data, candidaturas_data

pres_data, candidaturas_data = cargar_datos()
st.write("Datos cargados exitosamente.")

# Mostrar las primeras filas de los datos
st.write("Primeras filas de PRES_2024.csv_exported.csv:")
st.dataframe(pres_data.head())

st.write("Primeras filas de PRES_CANDIDATURAS_2024.csv:")
st.dataframe(candidaturas_data.head())

# Selección aleatoria de casillas
st.write("Seleccionando 400 casillas aleatorias...")
def seleccionar_casillas_aleatorias(df, n=400):
    return df.sample(n=n, random_state=1)

casillas_seleccionadas = seleccionar_casillas_aleatorias(pres_data)
st.write("Casillas seleccionadas:")
st.dataframe(casillas_seleccionadas)

# Función para obtener la URL del acta
def obtener_url_acta(entidad, seccion):
    url = f"https://prep.tec.mx/publicacion/nacional/assets/presidencia/entidad/{entidad}_{seccion}.json"
    response = requests.get(url)
    data = response.json()
    return data['casillas'][0]['url'] if data['casillas'] else None

# Configurar Google Cloud Vision y OpenAI
if google_api_key and openai_api_key:
    st.write("Claves API ingresadas correctamente.")
    client = vision.ImageAnnotatorClient()
    openai.api_key = openai_api_key

    # Procesar actas
    st.write("Procesando actas...")

    def procesar_acta(url):
        response = requests.get(url)
        image_content = response.content
        image = vision.Image(content=image_content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        return texts

    def parsear_informacion(texts):
        response = openai.Completion.create(
          model="gpt-3.5-turbo",
          prompt=f"Parsea la siguiente información de acta electoral:\n{texts[0].description}",
          temperature=0.7,
          max_tokens=1500
        )
        return response.choices[0].text

    # Procesar las actas de las casillas seleccionadas
    resultados = []
    for _, row in casillas_seleccionadas.iterrows():
        entidad = int(row['ID_ENTIDAD'])
        seccion = int(row['SECCION'])
        url_acta = obtener_url_acta(entidad, seccion)
        if url_acta:
            texts = procesar_acta(url_acta)
            informacion_structurada = parsear_informacion(texts)
            resultados.append({
                "Entidad": entidad,
                "Sección": seccion,
                "URL Acta": url_acta,
                "Información Estructurada": informacion_structurada
            })

    st.write("Resultados de la verificación:")
    resultados_df = pd.DataFrame(resultados)
    st.dataframe(resultados_df)

    # Guardar los resultados en un archivo CSV
    resultados_df.to_csv('resultados_verificacion.csv', index=False)
    st.write("Resultados guardados en el archivo 'resultados_verificacion.csv'.")

else:
    st.error("Por favor, ingresa ambas claves API para continuar.")
