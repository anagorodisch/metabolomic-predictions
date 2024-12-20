import streamlit as st
import pandas as pd
import numpy as np

import joblib
import lightgbm as lgb
from joblib import load
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from collections import Counter
from tensorflow.keras.models import load_model
from scipy.signal import savgol_filter
import plotly.graph_objs as go
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode

# Título de la aplicación
st.image("embryoxite.png", width=250)
st.title("PREDICCIÓN DE EMBARAZO Y PLOIDÍA")

# Importo los datos
datos = pd.read_csv('df_para_app_final.csv')
print(datos.columns)
datos = datos.drop(columns=['Unnamed: 0'])

columnas_deseadas = ["ID", 'EDAD PTE OVOCITOS', "PROCEDENCIA OVOCITOS", "PROCEDENCIA SEMEN", "ESTADO SEMEN", "DIA EMBRION", "GRADO EXPANSIÓN", "MCI", "TROFODERMO", "DESTINO"]
embriones = datos[columnas_deseadas]
embriones = embriones.drop_duplicates(subset='ID')

gb = GridOptionsBuilder.from_dataframe(embriones)
gb.configure_selection(selection_mode="single", use_checkbox=True)  # Selección única con checkbox
grid_options = gb.build()

# Mostrar el DataFrame interactivo
response = AgGrid(
    embriones,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
    theme="streamlit",  # Cambiar a "dark", "blue", "alpine" según tu preferencia
)

if not response['selected_rows'].empty:  # Verificar si no está vacío
    fila_seleccionada = response['selected_rows'].iloc[0]  # Obtener la primera fila seleccionada
    st.write("Detalles de la fila seleccionada:")
    st.write(fila_seleccionada.to_dict())
    
    # Guardar el valor del ID seleccionado
    id_seleccionado = fila_seleccionada['ID']
    st.write(f"ID seleccionado: {id_seleccionado}")
else:
    st.write("No se ha seleccionado ninguna fila.")

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
estado_semen = st.selectbox("Estado Semen", ["FRESCO", "CRIOPRESERVADO"])
dia_embrion = st.selectbox("Día del embrión", [5, 6])
destino = st.selectbox("Destino", ["FRESCO", "CONGELADO"])
grado_expansion = st.selectbox("Grado de Expansión", [3, 4, 5])
mci = st.selectbox("MCI", ["A", "B"])
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
            espectros_sindrift.insert(5, 'ESTADO SEMEN', estado_semen)
            espectros_sindrift.insert(6, 'DIA EMBRION', dia_embrion)
            espectros_sindrift.insert(7, 'GRADO EXPANSIÓN', grado_expansion)
            espectros_sindrift.insert(8, 'MCI', mci)
            espectros_sindrift.insert(9, 'TROFODERMO', trofodermo)
            espectros_sindrift.insert(10,'DESTINO', destino)

        except Exception as e:
            st.error(f"Error al concatenar los espectros recortados: {e}")
    else:
        st.info("No se encontraron espectros válidos para procesar.")

# GENERO LAS PREDICCIONES 
def tratamiento_señal(espectro_completo,model):
    def integrales(X):
        x = X.columns.astype(float)

        def calcular_integral(fila):
            return np.trapz(fila, x)

        integrales = X.apply(calcular_integral, axis=1)
        integrales_df = pd.DataFrame(integrales, columns=['Integral'])
        return integrales_df

    def segmentar(X, indicador):
        columnas = [ (799, 1165),  (1165, 1481),  (1481, 1772), (2818, 2999), (2999, 3699), (3699, 4000)]

        nombres = ['crudo_2', 'crudo_3', 'crudo_4', 'crudo_5', 'crudo_6', 'crudo_7']

        if indicador == 'der_1':
            nombres = ['d1_2', 'd1_3', 'd1_4', 'd1_5', 'd1_6', 'd1_7']

        integrales_list = []
        for inicio, fin in columnas:
            columnas_segmento = X.columns[(X.columns.astype(float) >= inicio) & (X.columns.astype(float) < fin)]
            X_segmento = X[columnas_segmento]
            integrales_segmento = integrales(X_segmento)
            integrales_list.append(integrales_segmento)

        df_segmentado = pd.concat(integrales_list, axis=1)
        df_segmentado.columns = nombres
        return df_segmentado

    # Eliminación de columnas no deseadas
    solo_espectros = espectro_completo.drop(columns=["ID", "REP_ID", 'EDAD PTE OVOCITOS', "PROCEDENCIA OVOCITOS", "PROCEDENCIA SEMEN", "ESTADO SEMEN", "DIA EMBRION", "GRADO EXPANSIÓN", "MCI", "TROFODERMO", "DESTINO"])

    # Especificar los intervalos de longitudes de onda a eliminar
    intervalos_a_eliminar = [(674, 799), (1772, 2818)]
    columnas_a_eliminar = []
    for inicio, fin in intervalos_a_eliminar:
        columnas_a_eliminar.extend(solo_espectros.columns[(solo_espectros.columns.astype(float) >= inicio) & (solo_espectros.columns.astype(float) <= fin)])

    espectros_filtrados = solo_espectros.drop(columns=columnas_a_eliminar)

    # Aplicar segmentación y cálculo de integrales
    if model == "lgb":
      df_derivative1 = espectros_filtrados.diff(axis=1).fillna(0)
    else:
      df_derivative1 = pd.DataFrame(savgol_filter(espectros_filtrados, 7, 3, deriv=1, axis=1), columns=espectros_filtrados.columns, index=espectros_filtrados.index )

    crudo = segmentar(espectros_filtrados, 'crudo')
    der_1 = segmentar(df_derivative1, 'der_1')

    df_resultado = pd.concat([crudo, der_1], axis=1)

    # Seleccionar y reorganizar columnas
    columns_to_add = [  "ID","REP_ID",'EDAD PTE OVOCITOS', "PROCEDENCIA OVOCITOS", "PROCEDENCIA SEMEN", "ESTADO SEMEN", "DIA EMBRION", "GRADO EXPANSIÓN", "MCI", "TROFODERMO", "DESTINO"]
    nuevo_dataset = espectro_completo[columns_to_add].join(df_resultado)
    return nuevo_dataset

modelo_lgb_ploidy= "Modelos_scalers_encoders/model_lgb_ploidy.txt" 
modelo_lgb_pregnancy= "Modelos_scalers_encoders/model_lgb_pregnancy.txt"
model_dnn_ploidy= "Modelos_scalers_encoders/model_dnn_ploidy.keras" 
model_dnn_pregnancy= "Modelos_scalers_encoders/model_dnn_pregnancy.keras" 

scalers_pregnancy= "Modelos_scalers_encoders/scaler_pregnancy_lgb.pkl"
scalers_ploidy= "Modelos_scalers_encoders/scaler_ploidy_lgb.pkl"

encoder_pregnancy= "Modelos_scalers_encoders/encoder_pregnancy.pkl"
encoder_ploidy= "Modelos_scalers_encoders/encoder_ploidy.pkl"

label_encoders_lgb = joblib.load('Modelos_scalers_encoders/label_encoders_lgb.pkl')

def generar_predicciones(clinica_y_espectros):

  df_completo_lgb = tratamiento_señal(clinica_y_espectros,"lgb")
  df_completo_dnn = tratamiento_señal(clinica_y_espectros,"dnn")

#####Bloque de prediccion lgb ploidia
  df_ploidia = df_completo_lgb.copy()
  columns_to_transform = ['PROCEDENCIA SEMEN', 'ESTADO SEMEN', 'DIA EMBRION', 'GRADO EXPANSIÓN', 'MCI', 'TROFODERMO', 'DESTINO','PROCEDENCIA OVOCITOS']

  for column in columns_to_transform:
      df_ploidia[column] = label_encoders_lgb[column].transform(df_ploidia[column])

  df_ploidia['EDAD PTE OVOCITOS'] = df_ploidia['EDAD PTE OVOCITOS'].astype(int)

  cols_no_escalar = ['ID', 'REP_ID']
  cols_escalares = ['crudo_2','crudo_3','crudo_4', 'crudo_5', 'crudo_6', 'crudo_7','d1_2','d1_3','d1_4','d1_5','d1_6','d1_7']

  df_ploidia = df_ploidia.drop(columns=cols_no_escalar)

  scaler = StandardScaler()
  scaler = load(scalers_ploidy)

  #transformacion de testeo
  X_test_n = df_ploidia[cols_escalares]
  X_test_n_scaled = scaler.transform(X_test_n)
  X_test_scaled_df = pd.DataFrame(X_test_n_scaled, columns=X_test_n.columns, index=df_ploidia.index)
  X_test_no_escalar = df_ploidia.drop(columns=cols_escalares)
  X_test = pd.concat([X_test_no_escalar, X_test_scaled_df], axis=1)

  model = lgb.Booster(model_file=modelo_lgb_ploidy)
  y_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)
  y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]

  #crear dataframe con todos los espectros y otro con una votacion mayoritaria
  df_pred_ploidia_lgb = df_completo_lgb[['REP_ID']].copy()
  df_pred_ploidia_lgb['Prediccion'] = y_pred

  pred_ploidia_lgb = Counter(df_pred_ploidia_lgb['Prediccion']).most_common(1)[0][0]

  if pred_ploidia_lgb == 1:
      pred_ploidia_lgb = 'EUPLOIDE'
  else:
      pred_ploidia_lgb = 'ANEUPLOIDE'

######Prediccion lgb embarazo
# Cargar los LabelEncoders guardados
  df_embarazo = df_completo_lgb.copy()
  columns_to_transform = ['PROCEDENCIA SEMEN', 'ESTADO SEMEN', 'DIA EMBRION', 'GRADO EXPANSIÓN', 'MCI', 'TROFODERMO', 'DESTINO','PROCEDENCIA OVOCITOS']

  for column in columns_to_transform:
      df_embarazo[column] = label_encoders_lgb[column].transform(df_embarazo[column])
  df_embarazo['EDAD PTE OVOCITOS'] = df_embarazo['EDAD PTE OVOCITOS'].astype(int)

  df_embarazo=df_embarazo.drop(columns=["ESTADO SEMEN"])

  cols_no_escalar = ['ID', 'REP_ID']
  cols_escalares = ['crudo_2','crudo_3','crudo_4', 'crudo_5', 'crudo_6', 'crudo_7','d1_2','d1_3','d1_4','d1_5','d1_6','d1_7',]

  df_embarazo = df_embarazo.drop(columns=cols_no_escalar)

  scaler = StandardScaler()
  scaler = load(scalers_pregnancy)

  #transformacion de testeo
  X_test_n = df_embarazo[cols_escalares]
  X_test_n_scaled = scaler.transform(X_test_n)
  X_test_scaled_df = pd.DataFrame(X_test_n_scaled, columns=X_test_n.columns, index=df_embarazo.index)
  X_test_no_escalar = df_embarazo.drop(columns=cols_escalares)
  X_test = pd.concat([X_test_no_escalar, X_test_scaled_df], axis=1)

  #cargamos modelo
  model = lgb.Booster(model_file=modelo_lgb_pregnancy)

  y_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)
  y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]
  #crear dataframe con todos los espectros y otro con una votacion mayoritaria
  df_pred_embarazo_lgb = df_completo_lgb[['REP_ID']].copy()
  df_pred_embarazo_lgb['Prediccion'] = y_pred

  pred_embarazo_lgb = Counter(df_pred_embarazo_lgb['Prediccion']).most_common(1)[0][0]

  if pred_embarazo_lgb == 1:
      pred_embarazo_lgb = 'EMBARAZO'
  else:
      pred_embarazo_lgb = 'NO EMBARAZO'

#######prediccion ploidia dnn

  df_ploidia = df_completo_dnn
  nuevo_orden = ['ID', 'REP_ID','crudo_2', 'crudo_3', 'crudo_4', 'crudo_5', 'crudo_6', 'crudo_7', 'd1_2', 'd1_3', 'd1_4', 'd1_5', 'd1_6', 'd1_7','EDAD PTE OVOCITOS', 'PROCEDENCIA OVOCITOS', 'PROCEDENCIA SEMEN','ESTADO SEMEN', 'DIA EMBRION', 'GRADO EXPANSIÓN', 'MCI', 'TROFODERMO','DESTINO']
  # Reorganizar columnas
  df_ploidia = df_ploidia[nuevo_orden]

  # PRIMERO PROCESO LAS VARIABLES NUMÉRICAS
  X_test_numeric = df_ploidia[cols_escalares]

  scaler=StandardScaler()
  scaler = load(scalers_ploidy)

  X_test_numeric=scaler.transform(X_test_numeric)

    # PROCESO LAS VARIABLES CATEGÓRICAS (ONEHOT ENCODING)
  categorias = df_ploidia[['PROCEDENCIA OVOCITOS','PROCEDENCIA SEMEN', 'ESTADO SEMEN', 'DIA EMBRION', 'GRADO EXPANSIÓN','MCI', 'TROFODERMO', 'DESTINO']]
  encoder = OneHotEncoder(sparse_output=False)

    # Guardar el encoder
  encoder = load(encoder_ploidy)

  X_test_cat = encoder.transform(df_ploidia[['PROCEDENCIA OVOCITOS','PROCEDENCIA SEMEN', 'ESTADO SEMEN', 'DIA EMBRION', 'GRADO EXPANSIÓN','MCI', 'TROFODERMO', 'DESTINO']])
    # JUNTO AMBAS CATEGORIAS
  column_names = encoder.get_feature_names_out(['PROCEDENCIA OVOCITOS', 'PROCEDENCIA SEMEN', 'ESTADO SEMEN', 'DIA EMBRION', 'GRADO EXPANSIÓN', 'MCI', 'TROFODERMO', 'DESTINO'])

  # Crear un DataFrame con las columnas transformadas
  X_test_cat_df = pd.DataFrame(X_test_cat, columns=column_names)

  X_test = np.column_stack((X_test_numeric, X_test_cat))
  X_test = X_test.astype(np.float32)

  X_test_1 = X_test[:, :13]
  X_test_2 = X_test[:, 12:]###revisar estoo

  X_test_compuesto = [X_test_1, X_test_2]
  dense_model = load_model(model_dnn_ploidy)

  y_pred = dense_model.predict(X_test_compuesto)
  y_pred = np.argmax(y_pred, axis=1)
  #crear dataframe con todos los espectros y otro con una votacion mayoritaria
  df_pred_ploidia_dnn = df_completo_dnn[['REP_ID']].copy()
  df_pred_ploidia_dnn['Prediccion'] = y_pred

  pred_ploidia_dnn = Counter(df_pred_ploidia_dnn['Prediccion']).most_common(1)[0][0]

  if pred_ploidia_dnn == 1:
      pred_ploidia_dnn = 'EUPLOIDE'
  else:
      pred_ploidia_dnn = 'ANEUPLOIDE'

###prediccion embarazo dnn
  # PRIMERO PROCESO LAS VARIABLES NUMÉRICAS
  df_embarazo = df_completo_dnn
  X_test_numeric = df_embarazo[cols_escalares]

  scaler=StandardScaler()
  scaler = load(scalers_pregnancy)

  X_test_numeric=scaler.transform(X_test_numeric)

    # PROCESO LAS VARIABLES CATEGÓRICAS (ONEHOT ENCODING)
  categorias = df_embarazo[['PROCEDENCIA OVOCITOS','PROCEDENCIA SEMEN', 'ESTADO SEMEN', 'DIA EMBRION', 'GRADO EXPANSIÓN','MCI', 'TROFODERMO', 'DESTINO']]
    # Guardar el encoder
  encoder = load(encoder_pregnancy)

  X_test_cat = encoder.transform(df_embarazo[['PROCEDENCIA OVOCITOS','PROCEDENCIA SEMEN', 'ESTADO SEMEN', 'DIA EMBRION', 'GRADO EXPANSIÓN','MCI', 'TROFODERMO', 'DESTINO']])
    # JUNTO AMBAS CATEGORIAS

  X_test = np.column_stack((X_test_numeric, X_test_cat))
  X_test = X_test.astype(np.float32)
  X_test_1 = X_test[:, :13]
  X_test_2 = X_test[:, 12:]#esto tambienn

  X_test_compuesto = [X_test_1, X_test_2]
  dense_model = load_model(model_dnn_pregnancy)

  y_pred = dense_model.predict(X_test_compuesto)
  y_pred = np.argmax(y_pred, axis=1)
  #crear dataframe con todos los espectros y otro con una votacion mayoritaria
  df_pred_embarazo_dnn = df_completo_dnn[['REP_ID']].copy()
  df_pred_embarazo_dnn['Prediccion'] = y_pred

  pred_embarazo_dnn = Counter(df_pred_embarazo_dnn['Prediccion']).most_common(1)[0][0]

  if pred_embarazo_dnn == 1:
      pred_embarazo_dnn = 'EMBARAZO'
  else:
      pred_embarazo_dnn = 'NO EMBARAZO'

  return pred_embarazo_dnn, pred_embarazo_lgb, pred_ploidia_dnn, pred_ploidia_lgb, df_pred_embarazo_dnn, df_pred_embarazo_lgb, df_pred_ploidia_dnn, df_pred_ploidia_lgb

def plot_signals(dataframe, predictions, task):
    traces = []
    # Crear un diccionario con REP_ID como clave y Prediccion como valor
    pred_dict = dict(zip(predictions['REP_ID'], predictions['Prediccion']))

    nombres_archivos = dataframe['REP_ID']
    # Eliminar las columnas no deseadas
    df = dataframe.drop(columns=['ID', 'REP_ID', 'EDAD PTE OVOCITOS', 'PROCEDENCIA OVOCITOS', 
                                 'PROCEDENCIA SEMEN', 'ESTADO SEMEN', 'DIA EMBRION', 
                                 'GRADO EXPANSIÓN', 'MCI', 'TROFODERMO', 'DESTINO'])

    for row in df.index:
        rep_id = nombres_archivos[row]
        # Obtener el valor de predicción para el espectro actual
        pred_value = pred_dict.get(rep_id, 0)  # Si no encuentra el valor, asigna 0 por defecto

        # Asignar color según la predicción
        color = 'red' if pred_value == 1 else 'blue'

        trace = go.Scatter(
            x=df.columns,
            y=df.loc[row],
            mode='lines',
            name=rep_id,
            line=dict(color=color),  # Asignar el color al espectro
            hovertemplate=f"Señal: {rep_id}<br>Predicción: {pred_value}<extra></extra>"
        )
        traces.append(trace)

    if task == 'embarazo':
        positive = 'EMBARAZO'
        negative = 'NO EMBARAZO'
    elif task == 'ploidia':
        positive = 'EUPLOIDE'
        negative = 'ANEUPLOIDE'

    # Agregar "color code" como una traza invisible para la leyenda
    color_code_traces = [
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='red'),
            name=positive
        ),
        go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=10, color='blue'),
            name=negative
        )
    ]

    # Crear la figura con todas las trazas
    fig = go.Figure(traces + color_code_traces)

    # Configuración del layout
    fig.update_layout(
        xaxis_title='Longitud de onda',
        yaxis_title='Absorbancia',
        legend_title='Espectros',
        hovermode='closest',
        height=800
    )

    return fig

# Botón para predecir
if st.button("Predecir"):
    if espectros_recortados:
        pred_embarazo_dnn, pred_embarazo_lgb, pred_ploidia_dnn, pred_ploidia_lgb, df_pred_embarazo_dnn, df_pred_embarazo_lgb, df_pred_ploidia_dnn, df_pred_ploidia_lgb = generar_predicciones(espectros_sindrift)
        data = {
        "Modelo_ML": [pred_embarazo_lgb, pred_ploidia_lgb],
        "Modelo_DL": [pred_embarazo_dnn, pred_ploidia_dnn]
        }

        df = pd.DataFrame(data, index=["Embarazo", "Ploidía"])
        st.subheader("Predicciones")
        st.table(df)

        st.subheader("Predicciones por Espectro")
        tab1, tab2, tab3, tab4 = st.tabs(["EMBARAZO_ML", "EMBARAZO_DL", "PLOIDÍA_ML", "PLOIDÍA_DL"])
        fig_1 = plot_signals(espectros_sindrift, df_pred_embarazo_lgb, 'embarazo')
        fig_2 = plot_signals(espectros_sindrift, df_pred_embarazo_dnn, 'embarazo')
        fig_3 = plot_signals(espectros_sindrift, df_pred_ploidia_lgb, 'ploidia')
        fig_4 = plot_signals(espectros_sindrift, df_pred_ploidia_dnn, 'ploidia')
        tab1.plotly_chart(fig_1, key="embarazo_ml")
        tab2.plotly_chart(fig_2, key="embarazo_dl")
        tab3.plotly_chart(fig_3, key="ploidia_ml")
        tab4.plotly_chart(fig_4, key="ploidia_dl")
    else:
        st.error("Por favor, cargue archivos para continuar.")

