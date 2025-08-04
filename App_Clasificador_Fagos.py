# App para clasificar fagos en su correspondiente especie

# -----------------------
# IMPORTACIONES
# -----------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import mysql.connector
from config_db import DB_CONFIG  # La configuración de conexión a base de datos se obtiene desde otro archivo .py

# -----------------------
# CONFIGURACIÓN DE PÁGINA Y ESTILO
# -----------------------
st.set_page_config(page_title="Clasificador de Fagos", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #dfe9f3, #ffffff);
    }
    .stButton>button, .stDownloadButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stTextArea textarea {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# TÍTULO DE LA APLICACIÓN
# -----------------------
st.title("🧬 Clasificador de Secuencias de ADN de Fagos")

# -----------------------
# CONEXIÓN A LA BASE DE DATOS
# -----------------------
def conectar_db():
    """Función que conecta a MySQL usando configuración externa"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        st.error(f"❌ Error de conexión: {err}")
        return None

# -----------------------
# CARGA EL MODELO Y LOS OBJETOS ENTRENADOS GENERADOS EN LA EJECUCIÓN DEL MODELO CON DATOS DE ENTRENAMIENTO Y VALIDACIÓN
# -----------------------
modelo = load_model("modelo_fagos.h5")
vectorizador_tfidf = joblib.load("vectorizador_tfidf.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -----------------------
# INTERFAZ PRINCIPAL: MODO DE INGRESO
# -----------------------
modo = st.radio("Selecciona el modo de predicción:", ["🧪 Ingresar secuencia manual", "📁 Subir archivo CSV"])

# -----------------------
# MODO 1: INGRESAR SECUENCIA MANUAL
# -----------------------
if modo == "🧪 Ingresar secuencia manual":
    st.markdown("Ingrese una secuencia de ADN para clasificarla:")
    secuencia = st.text_area("🔤 Secuencia de ADN")

    if st.button("🔍 Clasificar"):
        if len(secuencia.strip()) < 10:
            st.warning("⚠️ La secuencia debe tener al menos 10 caracteres.") # Valida que la secuencia contenga mínimo 10 caracteres
        else:
            conn = conectar_db()
            if conn:
                cursor = conn.cursor(dictionary=True)

                # Verifica si la secuencia de ADN ingresada ya existe en la tabla de bases de datos, es decir, ya ha sido clasificada
                cursor.execute("SELECT * FROM dbclasificador_fagos WHERE secuencia_adn = %s", (secuencia,))
                resultado = cursor.fetchone()

                if resultado:
                    especie = resultado['Host']
                    st.warning(f"⚠️ La secuencia ya ha sido clasificada previamente como **{especie}**.")
                else:
                    X = vectorizador_tfidf.transform([secuencia]).toarray()
                    pred = modelo.predict(X)
                    clase = label_encoder.inverse_transform([np.argmax(pred)])[0]

                    st.success(f"✅ Predicción: **{clase}**")

                    # Insertar en la base de datos
                    cursor.execute("""
                        INSERT INTO dbclasificador_fagos (Phage_ID, secuencia_adn, Host)
                        VALUES (%s, %s, %s)
                    """, (None, secuencia, clase))
                    conn.commit()
                    st.success("🗃️ Secuencia almacenada en la base de datos.")

                cursor.close()
                conn.close()

# -----------------------
# MODO 2: SUBIR ARCHIVO CSV
# -----------------------
elif modo == "📁 Subir archivo CSV":
    archivo = st.file_uploader("📄 Carga un archivo '.csv' con la columna 'secuencia_adn'", type="csv")

    if archivo:
        df = pd.read_csv(archivo)
        if 'secuencia_adn' not in df.columns:
            st.error("❌ El archivo debe tener la columna 'secuencia_adn'.")
        else:
            conn = conectar_db()
            if conn:
                cursor = conn.cursor(dictionary=True)
                nuevas = []
                duplicadas = []

                for _, fila in df.iterrows():
                    secuencia = fila['secuencia_adn']

                    cursor.execute("SELECT * FROM dbclasificador_fagos WHERE secuencia_adn = %s", (secuencia,))
                    existente = cursor.fetchone()

                    if existente:
                        duplicadas.append({
                            "secuencia_adn": secuencia,
                            "Host": existente['Host']
                        })
                    else:
                        nuevas.append((fila.get('Phage_ID', None), secuencia))

                # Muestra las secuencias duplicadas
                if duplicadas:
                    df_duplicadas = pd.DataFrame(duplicadas)
                    st.warning("⚠️ Las siguientes secuencias ya fueron clasificadas:")
                    st.dataframe(df_duplicadas)

                # Clasifica nuevas secuencias
                if nuevas:
                    secuencias_solas = [s[1] for s in nuevas]
                    X = vectorizador_tfidf.transform(secuencias_solas).toarray()
                    pred = modelo.predict(X)
                    clases_pred = label_encoder.inverse_transform(np.argmax(pred, axis=1))

                    df_nuevas = pd.DataFrame(nuevas, columns=["Phage_ID", "secuencia_adn"])
                    df_nuevas['Host'] = clases_pred

                    st.success("✅ Clasificación de nuevas secuencias realizada:")
                    st.dataframe(df_nuevas)

                    # Insertar a la base de datos y guardar en CSV
                    for _, fila in df_nuevas.iterrows():
                        cursor.execute("""
                            INSERT INTO dbclasificador_fagos (Phage_ID, secuencia_adn, Host)
                            VALUES (%s, %s, %s)
                        """, (fila['Phage_ID'], fila['secuencia_adn'], fila['Host']))
                    conn.commit()

                    df_nuevas.to_csv("dbclasificador_fagos.csv", index=False)
                    st.download_button("📥 Descargar resultados nuevos", df_nuevas.to_csv(index=False), file_name="dbclasificador_fagos.csv", mime="text/csv")

                else:
                    st.info("📄 No se encontraron nuevas secuencias para clasificar.")

                cursor.close()
                conn.close()

# -----------------------
# FINALIZAR APLICACIÓN
# -----------------------
if st.button("🛑 Finalizar aplicación"):
    st.write("🔌 Aplicación finalizada. Puedes cerrar la ventana o terminal.")
    st.stop()
