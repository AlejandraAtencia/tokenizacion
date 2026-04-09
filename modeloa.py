# -*- coding: utf-8 -*-
"""
Aplicación Streamlit – Modelo A
Evaluación de viabilidad para tokenización inmobiliaria
Antioquia, Colombia
 
Modelo entrenado con datos CEED–DANE (2020–2025)
Maestría en Ciencia de Datos
"""
 
import streamlit as st
import pickle
import numpy as np
import pandas as pd
 
# =====================================================
# Carga del modelo y artefactos
# =====================================================
# El pickle contiene:
#  - modelNN      : modelo de clasificación entrenado
#  - variables    : lista ordenada de variables del modelo
#  - scaler       : objeto de normalización (fit en entrenamiento)
#
# Nota: El LabelEncoder fue usado solo en entrenamiento,
# no es necesario para la fase de inferencia.
with open("modelo-class.pkl", "rb") as f:
    modelNN, _, variables, scaler = pickle.load(f)
 
# =====================================================
# Configuración de la página
# =====================================================
st.set_page_config(
    page_title="Viabilidad de Tokenización Inmobiliaria",
    page_icon="🏗️",
    layout="centered"
)
 
st.title("🏗️ Viabilidad de Tokenización Inmobiliaria")
st.markdown(
    "**Antioquia, Colombia** — Modelo de clasificación basado en datos CEED–DANE 2020–2025"
)
st.markdown("---")
st.markdown(
    "Complete las características del proyecto para evaluar su viabilidad "
    "potencial de inversión bajo esquemas de tokenización inmobiliaria."
)
 
# =====================================================
# Formulario de entrada
# =====================================================
col1, col2 = st.columns(2)
 
with col1:
    st.subheader("💰 Precio y valor")
 
    PRECIOVTAX = st.number_input(
        "Precio por m² (en miles de COP)",
        min_value=100,
        max_value=9400,
        value=2500,
        step=100,
        help="Ejemplo: 2500 equivale a COP 2.500.000 por m²"
    )
 
    TIPOVRDEST = st.selectbox(
        "Tipo de valor del precio",
        options=[1, 2],
        format_func=lambda x: "Real" if x == 1 else "Estimado",
        help="Indica si el precio reportado es real o estimado"
    )
 
    st.subheader("🏢 Perfil del proyecto")
 
    ESTRATO = st.selectbox(
        "Estrato socioeconómico",
        options=[1, 2, 3, 4, 5, 6],
        index=2,
        help="Estrato predominante del proyecto (1=bajo, 6=alto)"
    )
 
    RANVIVI = st.selectbox(
        "Rango de precio de vivienda",
        options=[1, 2, 3, 4, 5, 6],
        format_func=lambda x: {
            1: "VIP (hasta 70 SMMLV)",
            2: "VIS (70–135 SMMLV)",
            3: "No VIS bajo (135–235 SMMLV)",
            4: "No VIS medio (235–435 SMMLV)",
            5: "No VIS alto (435–1000 SMMLV)",
            6: "Premium (más de 1000 SMMLV)"
        }[x],
        index=2
    )
 
with col2:
    st.subheader("🔨 Avance de obra")
 
    CAPITULO = st.selectbox(
        "Capítulo actual de obra",
        options=[1, 2, 3, 4, 5, 6],
        format_func=lambda x: {
            1: "Cimentación",
            2: "Estructura",
            3: "Mampostería",
            4: "Acabados",
            5: "Instalaciones",
            6: "Remates"
        }[x],
        index=2,
        help="Variable tratada como ordinal según el entrenamiento"
    )
 
    GRADOAVANC = st.slider(
        "Grado de avance (%)",
        min_value=1,
        max_value=100,
        value=50,
        help="Porcentaje aproximado de avance físico del proyecto"
    )
 
    st.subheader("📋 Formalidad")
 
    OB_FORMAL = st.selectbox(
        "Formalidad de la obra",
        options=[1, 2],
        format_func=lambda x: "Formal" if x == 1 else "Informal",
        help="Cumplimiento normativo y licenciamiento del proyecto"
    )
 
    AMPLIACION = st.selectbox(
        "¿El proyecto corresponde a una ampliación?",
        options=[1, 2],
        format_func=lambda x: "Sí" if x == 1 else "No",
        index=1
    )
 
st.markdown("---")
 
# =====================================================
# Predicción
# =====================================================
if st.button("🔍 Evaluar viabilidad del proyecto", use_container_width=True):
 
    # Inicializar vector de entrada con todas las variables del modelo
    fila = {col: 0 for col in variables}
 
    # Asignación directa de variables numéricas / ordinales
    fila["PRECIOVTAX"] = PRECIOVTAX
    fila["GRADOAVANC"] = GRADOAVANC
    fila["ESTRATO"] = ESTRATO
    fila["RANVIVI"] = RANVIVI
    fila["CAPITULO"] = CAPITULO
 
    # Codificación one-hot de variables categóricas
    if f"TIPOVRDEST_{TIPOVRDEST}" in fila:
        fila[f"TIPOVRDEST_{TIPOVRDEST}"] = 1
 
    if f"OB_FORMAL_{OB_FORMAL}" in fila:
        fila[f"OB_FORMAL_{OB_FORMAL}"] = 1
 
    if f"AMPLIACION_{AMPLIACION}" in fila:
        fila[f"AMPLIACION_{AMPLIACION}"] = 1
 
    # Construir DataFrame y aplicar normalización
    entrada = pd.DataFrame([fila])
    entrada_scaled = scaler.transform(entrada)
 
    # Predicción
    pred = modelNN.predict(entrada_scaled)[0]
    prob = modelNN.predict_proba(entrada_scaled)[0]
 
    # =================================================
    # Resultados
    # =================================================
    st.markdown("## Resultado del modelo")
 
    if pred == 1:
        st.success("✅ **PROYECTO VIABLE PARA TOKENIZACIÓN**")
        st.metric(
            label="Probabilidad estimada de viabilidad",
            value=f"{prob[1] * 100:.1f}%"
        )
        st.markdown(
            "El modelo clasifica este proyecto como viable para esquemas de "
            "tokenización inmobiliaria, con base en patrones históricos de "
            "proyectos formalizados y completados exitosamente en Antioquia."
        )
    else:
        st.error("❌ **PROYECTO NO VIABLE PARA TOKENIZACIÓN**")
        st.metric(
            label="Probabilidad estimada de no viabilidad",
            value=f"{prob[0] * 100:.1f}%"
        )
        st.markdown(
            "El modelo clasifica este proyecto como no viable para esquemas de "
            "tokenización inmobiliaria, al presentar características "
            "frecuentes en proyectos con alto riesgo de paralización."
        )
 
    st.markdown("### Detalle de probabilidades")
    col_a, col_b = st.columns(2)
    col_a.metric("No viable (0)", f"{prob[0] * 100:.1f}%")
    col_b.metric("Viable (1)", f"{prob[1] * 100:.1f}%")
 
    # Explicabilidad básica
    with st.expander("📊 Variables ingresadas al modelo"):
        st.dataframe(entrada, use_container_width=True)
 
    st.info(
        "⚠️ Este resultado es una estimación probabilística basada en datos "
        "históricos. No constituye recomendación financiera, legal ni de inversión."
    )
 
st.markdown("---")
st.caption(
    "Modelo desarrollado con datos CEED–DANE 2020–2025 | "
    "Maestría en Ciencia de Datos | Tokenización Inmobiliaria – Antioquia"
)
