import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import mysql.connector
from config_db import DB_CONFIG  # Configuración externa

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

st.title("🧬 Clasificador de Secuencias de ADN de Fagos")

# -----------------------
# CONEXIÓN A BASE DE DATOS
# -----------------------
def conectar_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        st.error(f"Error de conexión: {err}")
        return None

# -----------------------
# Cargar modelo y objetos entrenados
# -----------------------
modelo = load_model("modelo_fagos.h5")
vectorizador_tfidf = joblib.load("vectorizador_tfidf.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# -----------------------
# Interfaz de usuario
# -----------------------
modo = st.radio("Selecciona el modo de predicción:", ["🧪 Ingresar secuencia manual", "📁 Subir archivo CSV"])

# -----------------------
# Modo 1: Secuencia manual
# -----------------------
if modo == "🧪 Ingresar secuencia manual":
    st.markdown("Ingresa una secuencia de ADN para clasificarla:")
    secuencia = st.text_area("🔤 Secuencia de ADN")

    if st.button("🔍 Predecir"):
        if len(secuencia.strip()) < 10:
            st.warning("⚠️ La secuencia es muy corta.")
        else:
            conn = conectar_db()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM dbclasificador_fagos WHERE secuencia_adn = %s", (secuencia,))
                existe = cursor.fetchone()[0] > 0
                if existe:
                    st.warning("⚠️ La secuencia ya ha sido clasificada anteriormente.")
                else:
                    X = vectorizador_tfidf.transform([secuencia]).toarray()
                    pred = modelo.predict(X)
                    clase = label_encoder.inverse_transform([np.argmax(pred)])
                    st.success(f"✅ Predicción: **{clase[0]}**")
                    cursor.execute("""
                        INSERT INTO dbclasificador_fagos (Phage_ID, secuencia_adn, Host)
                        VALUES (%s, %s, %s)
                    """, (None, secuencia, clase[0]))
                    conn.commit()
                    st.success("✅ Secuencia almacenada en base de datos.")
                cursor.close()
                conn.close()

# -----------------------
# Modo 2: Archivo CSV
# -----------------------
elif modo == "📁 Subir archivo CSV":
    archivo = st.file_uploader("📄 Carga un archivo `.csv` con una columna llamada `secuencia_adn`", type="csv")

    if archivo:
        df = pd.read_csv(archivo)
        if 'secuencia_adn' not in df.columns:
            st.error("❌ El archivo debe contener una columna llamada `secuencia_adn`.")
        else:
            st.success(f"✅ {len(df)} secuencias cargadas correctamente.")

            conn = conectar_db()
            if conn:
                cursor = conn.cursor()
                secuencias_nuevas = []
                secuencias_existentes = []

                for i, fila in df.iterrows():
                    cursor.execute("SELECT COUNT(*) FROM dbclasificador_fagos WHERE secuencia_adn = %s", (fila['secuencia_adn'],))
                    if cursor.fetchone()[0] == 0:
                        secuencias_nuevas.append(fila['secuencia_adn'])
                    else:
                        secuencias_existentes.append(fila['secuencia_adn'])

                if not secuencias_nuevas:
                    st.warning("⚠️ Todas las secuencias del archivo ya están clasificadas.")
                else:
                    st.info(f"📢 Se omitieron {len(secuencias_existentes)} secuencias ya clasificadas.")
                    X = vectorizador_tfidf.transform(secuencias_nuevas).toarray()
                    pred = modelo.predict(X)
                    y_pred = np.argmax(pred, axis=1)
                    clases_pred = label_encoder.inverse_transform(y_pred)
                    df_nuevo = pd.DataFrame({
                        'secuencia_adn': secuencias_nuevas,
                        'prediccion': clases_pred
                    })
                    st.dataframe(df_nuevo)
                    for index, row in df_nuevo.iterrows():
                        cursor.execute("""
                            INSERT INTO dbclasificador_fagos (Phage_ID, secuencia_adn, Host)
                            VALUES (%s, %s, %s)
                        """, (None, row['secuencia_adn'], row['prediccion']))
                    conn.commit()
                    df_nuevo.to_csv("dbclasificador_fagos.csv", index=False)
                    st.success("✅ Clasificación realizada y datos almacenados en base de datos y archivo CSV.")

                cursor.close()
                conn.close()

# -----------------------
# Botón para cerrar aplicación
# -----------------------
if st.button("🛑 Finalizar aplicación"):
    st.write("🔌 Aplicación finalizada. Puedes cerrar la ventana o terminal.")
    st.stop()
