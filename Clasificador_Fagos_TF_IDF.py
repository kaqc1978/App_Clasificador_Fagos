# 1) IMPORTACIONES Y CONFIGURACIÓN INICIAL
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mysql.connector
from config_db import DB_CONFIG
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2) CARGAR DATASETS
print("Cargando datasets...")
df_train = pd.read_csv("Fagos_entrenamiento.csv")
df_test = pd.read_csv("Fagos_validacion.csv")

# 3) PREPARAR DATOS
X_train_raw = df_train['secuencia_adn']
y_train_raw = df_train['Host']
X_test_raw = df_test['secuencia_adn']
y_test_raw = df_test['Host']

# Codificar etiquetas
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train_raw)
y_test = label_encoder.transform(y_test_raw)

# Vectorización TF-IDF
print("Vectorizando secuencias con TF-IDF...")
tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(5, 7), max_features=100000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw)
X_test_tfidf = tfidf_vectorizer.transform(X_test_raw)

# 4) CONSTRUCCIÓN DEL MODELO
print("Construyendo modelo de red neuronal...")
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_tfidf.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(len(np.unique(y_train)), activation='softmax')
])
model.compile(optimizer=Adam(learning_rate=0.0003), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5) ENTRENAMIENTO
print("Entrenando modelo...")
history = model.fit(X_train_tfidf.toarray(), y_train,
                    validation_data=(X_test_tfidf.toarray(), y_test),
                    epochs=100, batch_size=128, verbose=1)

# 6) EVALUACIÓN
print("Evaluando modelo...")
y_pred = np.argmax(model.predict(X_test_tfidf.toarray()), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy:.4f}')
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.tight_layout()
plt.show()

# 7) GRAFICAR HISTORIA DE ENTRENAMIENTO
def plot_training_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 12))
    ax[0].plot(history.history['loss'], label='Loss')
    ax[0].plot(history.history['val_loss'], label='Validation Loss')
    ax[0].set_title('Pérdida durante el Entrenamiento')
    ax[0].legend()

    ax[1].plot(history.history['accuracy'], label='Accuracy')
    ax[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[1].set_title('Precisión durante el Entrenamiento')
    ax[1].legend()
    plt.show()

plot_training_history(history)

# 8) GUARDAR MODELO Y OBJETOS
model.save("modelo_fagos.h5")
print("\n✅ Modelo guardado como modelo_fagos.h5")
joblib.dump(tfidf_vectorizer, "vectorizador_tfidf.pkl")
print("✅ Vectorizador TF-IDF guardado como vectorizador_tfidf.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("✅ Codificador de etiquetas guardado como label_encoder.pkl")

# 9) GUARDAR DATOS EN BASE DE DATOS Y CSV
print("\nConectando a base de datos...")
df_total = pd.concat([df_train, df_test], ignore_index=True)

conn = mysql.connector.connect(
    host=DB_CONFIG['host'],
    user=DB_CONFIG['user'],
    password=DB_CONFIG['password'],
    database=DB_CONFIG['database']
)
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS dbclasificador_fagos (
        id INT AUTO_INCREMENT PRIMARY KEY,
        Phage_ID VARCHAR(100),
        secuencia_adn LONGTEXT,
        Host VARCHAR(255)
    )
""")
conn.commit()

print("Insertando registros en la tabla dbclasificador_fagos...")
for _, row in df_total.iterrows():
    cursor.execute("""
        INSERT INTO dbclasificador_fagos (Phage_ID, secuencia_adn, Host)
        VALUES (%s, %s, %s)
    """, (row['Phage_ID'], row['secuencia_adn'], row['Host']))
conn.commit()

cursor.execute("SELECT Phage_ID, secuencia_adn, Host FROM dbclasificador_fagos")
rows = cursor.fetchall()
df_export = pd.DataFrame(rows, columns=["Phage_ID", "secuencia_adn", "Host"])
df_export.to_csv("dbclasificador_fagos.csv", index=False)

cursor.close()
conn.close()
print("✅ Registros guardados en base de datos y exportados como dbclasificador_fagos.csv")
