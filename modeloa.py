# -*- coding: utf-8 -*-
"""
Aplicación Streamlit – Modelo A
Predicción de viabilidad de obra (Red Neuronal)
Antioquia, Colombia
"""
 
import streamlit as st
import pickle
import pandas as pd
 
# =====================================================
# Carga del modelo y artefactos
# =====================================================
# pickle = (modelo_nn, labelencoder, variables, scaler)
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
st.markdown("**Antioquia, Colombia** — Modelo CEED–DANE 2020–2025")
st.markdown("---")
 
# =====================================================
# Formulario de entrada
# =====================================================
col1, col2 = st.columns(2)
 
with col1:
    st.subheader("💰 Precio y perfil")
 
    PRECIOVTAX = st.number_input(
        "Precio por m² (miles COP)",
        min_value=100,
        max_value=9400,
        value=2500,
        step=100
    )
 
    ESTRATO = st.selectbox(
        "Estrato socioeconómico",
        options=[1, 2, 3, 4, 5, 6]
    )
 
    RANVIVI = st.selectbox(
        "Rango de precio vivienda",
        options=[1, 2, 3, 4, 5, 6],
     format_func=lambda x: {
            1: "VIP (hasta 70 SMMLV)",
            2: "VIS (70-135 SMMLV)",
            3: "No VIS bajo (135-235 SMMLV)",
            4: "No VIS medio (235-435 SMMLV)",
            5: "No VIS alto (435-1000 SMMLV)",
            6: "Premium (más de 1000 SMMLV)"
        }[x],
        index=2
    )
 
    TIPOVRDEST = st.selectbox(
        "Tipo de valor del precio",
        options=[1, 2],
        format_func=lambda x: "Real" if x == 1 else "Estimado"
    )
 
with col2:
    st.subheader("🔨 Estado de la obra")
 
    CAPITULO = st.selectbox(
        "Capítulo de obra",
        options=[1, 2, 3, 4, 5, 6]
    )
 
    GRADOAVANC = st.slider(
        "Grado de avance (%)",
        min_value=1,
        max_value=100,
        value=50
    )
 
    OB_FORMAL = st.selectbox(
        "Formalidad de la obra",
        options=[1, 2],
        format_func=lambda x: "Formal" if x == 1 else "Informal"
    )
 
    AMPLIACION = st.selectbox(
        "¿Es una ampliación?",
        options=[1, 2],
        format_func=lambda x: "Sí" if x == 1 else "No"
    )
 
    # ✅ USO_DOS (según entrenamiento)
    USO_DOS = st.selectbox(
        "Uso del proyecto",
        options=[1, 3],
        format_func=lambda x: {
            1: "Residencial",
            3: "Mixto / Otro"
        }[x]
    )
 
st.markdown("---")
 
# =====================================================
# UMBRAL DE DECISIÓN
# =====================================================
UMBRAL = st.slider(
    "Umbral de viabilidad",
    min_value=0.50,
    max_value=0.90,
    value=0.65,
    step=0.05
)
 
st.markdown("---")
 
# =====================================================
# Predicción
# =====================================================
if st.button("🔍 Evaluar viabilidad", use_container_width=True):
 
    # -----------------------------
    # Inicializar vector completo
    # -----------------------------
    fila = {col: 0 for col in variables}
 
    # Variables numéricas
    fila["PRECIOVTAX"] = PRECIOVTAX
    fila["GRADOAVANC"] = GRADOAVANC
 
    # -----------------------------
    # DUMMIES (alineadas al training)
    # -----------------------------
    fila[f"ESTRATO_{ESTRATO}"] = 1
    fila[f"CAPITULO_{CAPITULO}"] = 1
    fila[f"RANVIVI_{RANVIVI}"] = 1
 
    if TIPOVRDEST == 2 and "TIPOVRDEST_2" in fila:
        fila["TIPOVRDEST_2"] = 1
 
    if OB_FORMAL == 1 and "OB_FORMAL_1" in fila:
        fila["OB_FORMAL_1"] = 1
 
    if AMPLIACION == 1 and "AMPLIACION_1" in fila:
        fila["AMPLIACION_1"] = 1
 
    if USO_DOS == 1 and "USO_DOS_1" in fila:
        fila["USO_DOS_1"] = 1
 
    if USO_DOS == 3 and "USO_DOS_3" in fila:
        fila["USO_DOS_3"] = 1
 
    # -----------------------------
    # DataFrame final alineado
    # -----------------------------
    entrada = pd.DataFrame([fila])
 
    # ✅ Alineación FINAL con el modelo
    entrada = entrada.reindex(columns=variables, fill_value=0)
 
    # ✅ Escalado correcto (Red Neuronal)
    X_modelo = entrada
 
    # -----------------------------
    # Probabilidades
    # -----------------------------
    prob = modelNN.predict_proba(X_modelo)[0]
 
    # Decisión con umbral
    pred = 1 if prob[1] >= UMBRAL else 0
 
    # =================================================
    # Resultados
    # =================================================
    st.markdown("## Resultado")
 
    if pred == 1:
        st.success("✅ PROYECTO VIABLE")
        st.metric("Probabilidad de viabilidad", f"{prob[1]*100:.1f}%")
    else:
        st.error("❌ PROYECTO NO VIABLE")
        st.metric("Probabilidad de no viabilidad", f"{prob[0]*100:.1f}%")
 
    with st.expander("📊 Variables enviadas al modelo"):
        st.dataframe(entrada)
 
    st.info(
        "La decisión final se basa en el umbral seleccionado y en la probabilidad "
        "estimada por la red neuronal."
    )
 
st.markdown("---")
st.caption(
    "Modelo CEED–DANE 2020–2025 | Maestría en Ciencia de Datos"
)
