import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Cargar modelos y transformaciones
modelo = joblib.load("stacking_model(1).pkl")
pca = joblib.load("pca_model(1).pkl")
scaler = joblib.load("scaler(1).pkl")

st.title("Bodyweight Loss Prediction")

st.markdown("Please complete all the following basal information for a correct prediction")

# Función para obtener entrada validada
def get_float_input(label):
    value = st.text_input(label)
    if value:
        try:
            return float(value)
        except ValueError:
            st.error(f"{label} debe ser un número válido.")
    return None


# Entradas de usuario

dbp = st.number_input("Diastolic Blood Pressure", min_value=10, max_value=200) 
age = st.number_input("Age", min_value=1, max_value=100)
ldlch = st.number_input("LDL Cholesterol", min_value=1, max_value=500)
musclemass = st.number_input("Muscle Mass", min_value=0.0, max_value=500.0, step=0.1)
homa_ir = st.number_input("Homeostatic Model Assessment of Insulin Resistance", min_value=1.000, max_value=500.000, step=0.001)
insuline = st.number_input("Insuline Level", min_value=0.00, max_value=200.00, step=0.01)
sbp = st.number_input("Sistolic Blood Pressure", min_value=50, max_value=220) 
intra_water = st.number_input("Intracellular Water", min_value=1.0, max_value=500.0, step=0.1) 
metrate = st.number_input("Metabolic Rate", min_value=500, max_value=5000) 
body_cell_mass = st.number_input("Body Cell Mass", min_value=1.0, max_value=100.0, step=0.1) 
bodyweight = st.number_input("Basal Bodyweight", min_value=1.0, max_value=300.0, step=0.1) 
pcr = st.number_input("PCR", min_value=0.00, max_value=200.00, step=0.01) 
waist_circumference = st.number_input("Waist Circumference", min_value=1.0, max_value=250.0, step=0.1) 
num_cig = st.number_input("Number of cigarettes per day", min_value=0, max_value=100) 
bilirubin = st.number_input("Bilirunin", min_value=0.0, max_value=250.0, step=0.1) 


# Botón para predecir
if st.button("Predecir"):
    try:
        input_data = np.array([[dbp, age, ldlch, musclemass, homa_ir, insuline, sbp,
                                intra_water, metrate, body_cell_mass, bodyweight, pcr,
                                waist_circumference, num_cig, bilirubin]])
        
        # Escalar los datos
        input_scaled = scaler.transform(input_data)

        # PCA
        input_pca = pca.transform(input_scaled)

        # Predicción
        resultado = modelo.predict(input_pca)[0]

        if resultado == 1:
            st.success(f"✅ Predicción: Éxito en la pérdida de peso")
        else:
            st.warning(f"❌ Predicción: No se espera éxito")
    except Exception as e:
        st.error(f"Ocurrió un error al predecir: {str(e)}")
