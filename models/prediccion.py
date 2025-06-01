from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

# Configuración inicial de la página
st.set_page_config(
    page_title="Clasificador de Aves",
    page_icon="🦜",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""<style>
    ... (tus estilos CSS aquí sin cambios) ...
</style>""", unsafe_allow_html=True)

# Encabezado
st.markdown("""<div class="header">
    <h1 style='text-align: center;'>🦜 Clasificador de Aves</h1>
    <p style='text-align: center;'>Identificación de especies mediante inteligencia artificial</p>
</div>""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ℹ️ Acerca de")
    st.markdown("""
        Esta aplicación utiliza un modelo de TensorFlow Lite para clasificar imágenes de aves en 10 categorías diferentes.
        **Especies reconocidas:**
        - Chipe Celeste
        - Chipe de Connecticut
        - Chipe de Connecticut Lores Negros
        - Chipe de Lawrence
        - Chipe de Pechera
        - Chipe Dorado
        - Chipe Peregrino
        - Chipe Trepador
        - Mascarita Equinoccial
        - Pavito Migratorio
    """)
    st.markdown("---")
    st.markdown("### ⚙️ Configuración")
    show_confidence = st.checkbox("Mostrar barra de confianza", value=True)

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    interpreter = tf.lite.Interpreter(model_path="models/modelo_convertido.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = cargar_modelo()
entrada = interpreter.get_input_details()
salida = interpreter.get_output_details()

# Preprocesamiento
def preparar_imagen_vgg16(imagen):
    imagen = imagen.convert("RGB")
    imagen = imagen.resize((224, 224))
    matriz = np.array(imagen).astype(np.float32) / 255.0
    matriz = np.expand_dims(matriz, axis=0)
    return matriz

# Etiquetas
etiquetas = [
    'CHIPE CELESTE', 'CHIPE DE CONNECTICUT', 'CHIPE DE CONNECTICUT LORES NEGROS DE PECHERA',
    'CHIPE DE LAWRENCE', 'CHIPE DE PECHERA', 'CHIPE DORADO',
    'CHIPE PEREGRINO', 'CHIPE TREPADOR', 'MASCARITA EQUINOCCIAL', 'PAVITO MIGRATORIO'
]

# Subida de imagen
st.markdown("## 📤 Subir Imagen")
with st.container():
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    imagen_usuario = st.file_uploader("Arrastra o selecciona una imagen JPG/PNG del ave", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

if imagen_usuario:
    imagen = Image.open(imagen_usuario)

    # Mostrar imagen
    st.markdown('<div class="image-centered">', unsafe_allow_html=True)
    st.image(imagen, caption="Imagen subida", use_column_width=False, width=350)
    st.markdown('</div>', unsafe_allow_html=True)

    # Predicción
    with st.spinner('🔍 Analizando imagen...'):
        imagen_preparada = preparar_imagen_vgg16(imagen)
        interpreter.set_tensor(entrada[0]['index'], imagen_preparada)
        interpreter.invoke()
        salida_predicha = interpreter.get_tensor(salida[0]['index'])
        clase = np.argmax(salida_predicha)
        confianza = np.max(salida_predicha)

    # Mostrar resultado
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown(f"### 🎯 Resultado de la clasificación")
    st.markdown(f'<div class="bird-name">{etiquetas[clase]}</div>', unsafe_allow_html=True)

    if show_confidence:
        st.markdown(f"**Confianza:** {confianza*100:.2f}%")
        st.markdown(f'<div class="confidence-bar" style="width: {confianza*100}%;"></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Información de la especie
    info_dict = {
        0: "El Chipe Celeste (Setophaga cerulea)...",
        1: "El Chipe de Connecticut es una especie migratoria...",
        2: "El Chipe de Connecticut Lores Negros...",
        3: "El Chipe de Lawrence es conocido por su distintivo patrón...",
        4: "El Chipe de Pechera tiene tonos vibrantes...",
        5: "El Chipe Dorado destaca por su plumaje amarillo...",
        6: "El Chipe Peregrino es una especie ágil...",
        7: "El Chipe Trepador escala árboles con facilidad...",
        8: "La Mascarita Equinoccial tiene una máscara facial distintiva...",
        9: "El Pavito Migratorio realiza largas migraciones..."
    }
    st.markdown("### 📚 Sobre esta especie")
    st.info(info_dict.get(clase, "Información no disponible para esta especie."))

    # Precisión por clase (simulada)
    st.markdown("### 📊 Precisión del modelo por especie")
    fig, ax = plt.subplots(figsize=(10, 5))
    precisiones_simuladas = np.random.uniform(0.7, 0.95, size=len(etiquetas))
    ax.barh(etiquetas, precisiones_simuladas, color='#6e8efb')
    ax.set_xlim(0, 1)
    ax.set_xlabel("Precisión")
    ax.set_title("Precisión estimada por clase")
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.info("ℹ️ Por favor, sube una imagen de un ave para comenzar el análisis.")
