import streamlit as st

# Título de la aplicación
st.title("PREDICCIÓN DE EMBARAZO Y PLOIDÍA")

# Selección de archivos
st.subheader("Seleccionar la señal:")
uploaded_files = st.file_uploader(
    "Cargue uno o más archivos desde la carpeta de su computadora",
    accept_multiple_files=True,
)

if st.button("Confirmar selección"):
    if uploaded_files:
        file_names = [file.name for file in uploaded_files]
        st.success(f"Se seleccionaron los siguientes archivos: {', '.join(file_names)}")
    else:
        st.error("Por favor, cargue al menos un archivo.")

# Formulario para ingresar variables clínicas
st.subheader("Indique las variables clínicas del embrión:")

# Input de texto para Edad Ovocitos
edad_ovocitos = st.text_input("Edad Ovocitos")

# Dropdowns para las demás variables clínicas
procedencia_ovocitos = st.selectbox("Procedencia Ovocitos", ["ÓVULO PROPIO", "ÓVULO DONADO"])
procedencia_semen = st.selectbox("Procedencia Semen", ["SEMEN PROPIO", "SEMEN DONADO"])
estado_semen = st.selectbox("Estado Semen", ["FRESCO", "CONGELADO   "])
dia_embrion = st.selectbox("Día del embrión", [5, 6])
destino = st.selectbox("Destino", ["FRESCO", "CONGELADO"])
grado_expansion = st.selectbox("Grado de Expansión", [1, 2, 3, 4, 5, 6])
mci = st.selectbox("MCI", ["A", "B", "C/D"])
trofodermo = st.selectbox("Trofodermo", ["A", "B", "C"])

# Botón para predecir
if st.button("Predecir"):
    if uploaded_files:
        file_names = [file.name for file in uploaded_files]
        st.success(f"Se procesaron los siguientes archivos: {', '.join(file_names)}")
    else:
        st.error("Por favor, cargue archivos para continuar.")
