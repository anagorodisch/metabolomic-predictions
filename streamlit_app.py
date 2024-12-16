import streamlit as st
import pandas as pd
import numpy as np

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

ID_embrion = st.text_input("ID embrión")

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

# ARMO EL DATAFRAME
def buscar_extremos(ftir):
    """Encuentra los índices de inicio y fin donde el valor no es cero."""
    L = len(ftir)
    inicio, fin = 0, L - 1
    for i in range(L):
        if ftir[i] != 0:
            inicio = i
            break
    for i in range(L):
        if ftir[L - 1 - i] != 0:
            fin = L - 1 - i
            break
    return inicio, fin

def transformacion(x):
    return np.log10(100 / x)

if uploaded_files:
    espectros_recortados = []
    nombres_archivos = []  # Lista para almacenar los nombres de los archivos

    for archivo in uploaded_files:
        # Verificar que sea un archivo .CSV
        if archivo.name.endswith('.CSV'):
            try:
                # Leer el archivo en un DataFrame
                csv_df = pd.read_csv(archivo, delimiter=';', decimal=',', header=None)

                # Verificar que tenga al menos dos columnas
                if csv_df.shape[1] < 2:
                    st.warning(f"El archivo {archivo.name} no tiene el formato esperado (mínimo 2 columnas).")
                    continue

                # Procesar el archivo
                inicio, fin = buscar_extremos(csv_df[1])  # Segunda columna (columna 1)
                espectro_recortado = csv_df.iloc[inicio:fin + 1]  # Recortar el espectro
                espectros_recortados.append(espectro_recortado[1].reset_index(drop=True))  # Guardar
                nombres_archivos.append(archivo.name)  # Agregar el nombre del archivo

            except Exception as e:
                st.error(f"Error al procesar el archivo {archivo.name}: {e}")
        else:
            st.warning(f"El archivo {archivo.name} no es un archivo .CSV.")

    # Concatenar los espectros recortados si hay datos válidos
    if espectros_recortados:
        try:
            df_espectros_recortados = pd.concat(espectros_recortados, axis=1, ignore_index=True).T
            # Asignar encabezados al nuevo DataFrame
            df_espectros_recortados.columns = csv_df.iloc[inicio:fin + 1][0].values
            
            # Agregar columna REP_ID con los nombres de los archivos
            df_espectros_recortados.insert(0, 'REP_ID', nombres_archivos)
            
            # Aplicar transformaciones sin afectar la columna REP_ID
            espectros_abs = df_espectros_recortados.iloc[:, 1:].applymap(transformacion)
            espectros_sindrift = espectros_abs.apply(lambda row: row - np.min(row), axis=1)

            # Agregar nuevamente REP_ID al DataFrame transformado
            espectros_sindrift.insert(0, 'REP_ID', nombres_archivos)

            espectros_sindrift.insert(0, 'ID', ID_embrion)  # Primera columna
            espectros_sindrift.insert(2, 'EDAD PTE OVOCITOS', edad_ovocitos)  # Después de REP_ID
            espectros_sindrift.insert(3, 'PROCEDENCIA OVOCITOS', procedencia_ovocitos)
            espectros_sindrift.insert(4, 'PROCEDENCIA SEMEN', procedencia_semen)
            espectros_sindrift.insert(5, 'DIA EMBRION', dia_embrion)
            espectros_sindrift.insert(6, 'GRADO EXPANSIÓN', grado_expansion)
            espectros_sindrift.insert(7, 'MCI', mci)
            espectros_sindrift.insert(8, 'TROFODERMO', trofodermo)
            espectros_sindrift.insert(9, 'DESTINO', destino)

        except Exception as e:
            st.error(f"Error al concatenar los espectros recortados: {e}")
    else:
        st.info("No se encontraron espectros válidos para procesar.")


# GENERO LAS PREDICCIONES 


# Botón para predecir
if st.button("Predecir"):
    if espectros_recortados:
        st.write("Espectros:")
        st.dataframe(espectros_sindrift)
    else:
        st.error("Por favor, cargue archivos para continuar.")