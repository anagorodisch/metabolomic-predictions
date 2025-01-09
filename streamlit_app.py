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
from functions import tratamiento_señal, generar_predicciones, plot_signals, format_prediction

st.set_page_config(layout="wide")

# Título de la aplicación
st.image("embryoxite.png", width=250)
st.title("PREGNANCY AND PLOIDY PREDICTION")


st.markdown("""
    <div style="padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
        <strong>Disclaimer and Professional Use Only</strong><br>
        This tool is intended for use by trained professionals in clinical settings. It does not constitute a definitive medical diagnosis or decision-making tool and should always be used in conjunction with professional judgment and established clinical protocols.
    </div>
""", unsafe_allow_html=True)

st.subheader("Select an Embryo:")
# Importo los datos
datos = pd.read_csv('df_para_app_final.csv')
print(datos.columns)
#datos = datos.drop(columns=['Unnamed: 0'])

columnas_deseadas = ["ID", 'EDAD PTE OVOCITOS', "PROCEDENCIA OVOCITOS", "PROCEDENCIA SEMEN", "ESTADO SEMEN", "DIA EMBRION", "GRADO EXPANSIÓN", "MCI", "TROFODERMO", "DESTINO"]
embriones = datos[columnas_deseadas]
embriones = embriones.drop_duplicates(subset='ID')
embriones = embriones.reset_index(drop=True)
embriones = embriones.rename(columns={
    'EDAD PTE OVOCITOS': 'Oocyte Age',
    'PROCEDENCIA OVOCITOS': 'Oocyte Source',
    'PROCEDENCIA SEMEN': 'Sperm Source',
    'ESTADO SEMEN': 'Sperm State',
    'DIA EMBRION': 'Embryo Transfer Day',
    'GRADO EXPANSIÓN': 'Expansion Grade',
    'MCI': 'ICM',
    'TROFODERMO': 'Trophectoderm',
    'DESTINO': 'Preservation Process'
})
reemplazos = {
    "CONGELADO": "CRYOPRESERVED",
    "ÓVULO DONADO": "DONOR",
    "ÓVULO PROPIO": "SELF",
    "SEMEN DONADO": "DONOR",
    "SEMEN PROPIO": "SELF",
    "FRESCO": "FRESH"
}
embriones = embriones.replace(reemplazos)

event = st.dataframe(
    embriones,
    use_container_width=True,
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
)

# Verificar si se ha seleccionado una fila
if event and event.selection.rows:
    embrion = event.selection.rows
    filtered_df = embriones.iloc[embrion]
    ID_embrion = filtered_df['ID'].values
    ID_embrion = ID_embrion[0]
    ID_embrion = str(ID_embrion)
    st.write(f"Selected ID: {ID_embrion}")

    # Filtrar datos del embrión seleccionado
    df_embrion = datos[datos['ID'] == ID_embrion]
    df_embrion = df_embrion.drop(columns=['embarazo', 'ploidía'])

    # Botón para predecir
    if st.button("Predict"):
        pred_embarazo_dnn, pred_embarazo_lgb, pred_ploidia_dnn, pred_ploidia_lgb, df_pred_embarazo_dnn, df_pred_embarazo_lgb, df_pred_ploidia_dnn, df_pred_ploidia_lgb = generar_predicciones(df_embrion)
        
        pred_embarazo_lgb = format_prediction(pred_embarazo_lgb)
        pred_ploidia_lgb = format_prediction(pred_ploidia_lgb)
        pred_embarazo_dnn = format_prediction(pred_embarazo_dnn)
        pred_ploidia_dnn = format_prediction(pred_ploidia_dnn)

        data = {
        "Model_ML": [pred_embarazo_lgb, pred_ploidia_lgb],
        "Model_DL": [pred_embarazo_dnn, pred_ploidia_dnn]
        }

        df = pd.DataFrame(data, index=["PREGNANCY", "PLOIDY"])
        st.subheader("Predictions")
        st.table(df)

        st.subheader("Predictions by Spectrum")
        tab1, tab2, tab3, tab4 = st.tabs(["PREGNANCY_ML", "PREGNANCY_DL", "PLOIDY_ML", "PLOIDY_DL"])
        fig_1 = plot_signals(df_embrion, df_pred_embarazo_lgb, 'embarazo')
        fig_2 = plot_signals(df_embrion, df_pred_embarazo_dnn, 'embarazo')
        fig_3 = plot_signals(df_embrion, df_pred_ploidia_lgb, 'ploidia')
        fig_4 = plot_signals(df_embrion, df_pred_ploidia_dnn, 'ploidia')
        tab1.plotly_chart(fig_1, key="embarazo_ml")
        tab2.plotly_chart(fig_2, key="embarazo_dl")
        tab3.plotly_chart(fig_3, key="ploidia_ml")
        tab4.plotly_chart(fig_4, key="ploidia_dl")

else:
    st.write("Please select and embryo from the table.")

st.markdown("""
    <p style="font-size: 12px; text-align: center; position: fixed; bottom: 0; width: 100%; background-color: white; padding: 5px;">
        The models are still in development and may evolve over time
    </p>
""", unsafe_allow_html=True)