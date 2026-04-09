# -*- coding: utf-8 -*-
"""
Aplicación Streamlit – Modelo A
Predicción de viabilidad de obra para tokenización inmobiliaria
Antioquia, Colombia
"""
 
import streamlit as st
import pickle
import pandas as pd
 
# =====================================================
# Carga del modelo y artefactos
# =====================================================
# pickle = (modelo, labelencoder, variables, scaler)
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
st.markdown(
    "Ingrese las características del proyecto para estimar su "
    "viabilidad bajo esquemas de tokenización inmobiliaria."
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
        step=100
    )
 
    TIPOVRDEST = st.selectbox(
        "Tipo de valor del precio",
        options=[1, 2],
        format_func=lambda x: "Real" if x == 1 else "Estimado"
    )
 
    st.subheader("🏢 Perfil del proyecto")
 
    ESTRATO = st.selectbox(
        "Estrato socioeconómico",
        options=[1, 2, 3, 4, 5, 6],
        index=2
    )
 
    RANVIVI = st.selectbox(
        "Rango de precio de vivienda",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: {
            0: "Sin clasificar",
            1: "VIP",
            2: "VIS",
            3: "No VIS bajo",
            4: "No VIS medio",
            5: "No VIS alto",
            6: "Premium"
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
        }[x]
    )
 
    GRADOAVANC = st.slider(
        "Grado de avance (%)",
        min_value=1,
        max_value=100,
        value=50
    )
 
    st.subheader("📋 Legalidad y tipo")
 
    OB_FORMAL = st.selectbox(
        "Formalidad de la obra",
        options=[1, 2],
        format_func=lambda x: "Formal" if x == 1 else "Informal"
    )
 
    AMPLIACION = st.selectbox(
        "¿Es una ampliación?",
        options=[1, 2],
        format_func=lambda x: "Sí" if x == 1 else "No",
        index=1
    )
 
    # ✅ USO_DOS (alineado con el entrenamiento)
    # En el entrenamiento se eliminó USO_DOS_2,
    # por eso SOLO se permiten 1 y 3
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
# Predicción
# =====================================================
if st.button("🔍 Evaluar viabilidad del proyecto", use_container_width=True):
 
    # Inicializar TODAS las variables del modelo
    fila = {col: 0 for col in variables}
 
    # Variables numéricas
    fila["PRECIOVTAX"] = PRECIOVTAX
    fila["GRADOAVANC"] = GRADOAVANC
 
    # =================================================
    # DUMMIES (alineadas exactamente al notebook)
    # =================================================
 
    # --- ESTRATO ---
    for i in range(1, 7):
        col = f"ESTRATO_{i}"
        if col in fila:
            fila[col] = 1 if ESTRATO == i else 0
 
    # --- CAPITULO ---
    for i in range(1, 7):
        col = f"CAPITULO_{i}"
        if col in fila:
            fila[col] = 1 if CAPITULO == i else 0
 
    # --- RANVIVI ---
    for i in range(0, 7):
        col = f"RANVIVI_{i}"
        if col in fila:
            fila[col] = 1 if RANVIVI == i else 0
 
    # --- TIPOVRDEST ---
    if "TIPOVRDEST_2" in fila:
        fila["TIPOVRDEST_2"] = 1 if TIPOVRDEST == 2 else 0
 
    # --- OB_FORMAL ---
    if "OB_FORMAL_1" in fila:
        fila["OB_FORMAL_1"] = 1 if OB_FORMAL == 1 else 0
 
    # --- AMPLIACION ---
    if "AMPLIACION_1" in fila:
        fila["AMPLIACION_1"] = 1 if AMPLIACION == 1 else 0
 
    # ✅ --- USO_DOS (clave del error) ---
    if "USO_DOS_1" in fila:
        fila["USO_DOS_1"] = 1 if USO_DOS == 1 else 0
 
    if "USO_DOS_3" in fila:
        fila["USO_DOS_3"] = 1 if USO_DOS == 3 else 0
 
    # =================================================
    # DataFrame y escalado
    # =====================================================
    entrada = pd.DataFrame([fila])
 
    # Aplica scaler solo si existe
    X_modelo = scaler.transform(entrada) if scaler else entrada
 
    # Predicción
    pred = modelNN.predict(X_modelo)[0]
    prob = modelNN.predict_proba(X_modelo)[0]
 
    # =================================================
    # Resultados
    # =================================================
    st.markdown("## Resultado del modelo")
 
    if pred == 1:
        st.success("✅ **PROYECTO VIABLE PARA TOKENIZACIÓN**")
        st.metric("Probabilidad de viabilidad", f"{prob[1] * 100:.1f}%")
    else:
        st.error("❌ **PROYECTO NO VIABLE PARA TOKENIZACIÓN**")
        st.metric("Probabilidad de no viabilidad", f"{prob[0] * 100:.1f}%")
 
    st.markdown("### Detalle de probabilidades")
    col_a, col_b = st.columns(2)
    col_a.metric("No viable (0)", f"{prob[0] * 100:.1f}%")
    col_b.metric("Viable (1)", f"{prob[1] * 100:.1f}%")
 
    with st.expander("📊 Variables enviadas al modelo"):
        st.dataframe(entrada, use_container_width=True)
 
    st.info(
        "Resultado estimado con base en datos históricos. "
        "No constituye recomendación financiera ni legal."
    )
 
st.markdown("---")
st.caption(
    "Modelo CEED–DANE 2020–2025 | Maestría en Ciencia de Datos | "
    "Tokenización Inmobiliaria – Antioquia"
)
``
