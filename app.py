
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Calidad del Agua - Lagos Pucón y Villarrica",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🌊 Sistema de Análisis de Calidad del Agua")
st.subheader("Lagos de Pucón y Villarrica - Región de la Araucanía")

# Función para cargar y preprocesar datos CORREGIDA
@st.cache_data
def load_and_preprocess_data():
    """Carga y preprocesa los datos del CSV"""
    try:
        # Cargar datos
        df = pd.read_csv('/Users/ignaciozambrano/Library/CloudStorage/OneDrive-Personal/python/analisis-calidad-agua/Consolidado Entrenamiento - Tabla Completa (1).csv')
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip()
        
        # Identificar columnas que pueden tener valores numéricos con comas
        # Convertir TODAS las columnas excepto las claramente categóricas
        categorical_cols = ['Folio', 'Lugar Muestreo', 'Comuna']
        
        for col in df.columns:
            if col not in categorical_cols:
                # Intentar convertir a numérico, reemplazando comas por puntos
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        # Limpiar nombres de lugares y comunas
        df['Lugar Muestreo'] = df['Lugar Muestreo'].str.strip().str.title()
        df['Comuna'] = df['Comuna'].str.strip().str.title()
        
        # Crear variables categóricas con labels descriptivos
        viento_labels = {0: 'Sin viento', 1: 'Viento leve', 2: 'Viento moderado', 3: 'Viento fuerte'}
        oleaje_labels = {0: 'Sin oleaje', 1: 'Oleaje leve', 2: 'Oleaje moderado', 3: 'Oleaje fuerte'}
        musgo_labels = {0: 'Sin musgo', 1: 'Musgo verde', 2: 'Musgo pardo', 3: 'Ambos tipos'}
        algas_labels = {0: 'Sin algas', 1: 'Pocas algas', 2: 'Moderadas algas', 3: 'Muchas algas'}
        cielo_labels = {0: 'Soleado', 1: 'Parcial', 2: 'Nublado'}
        
        df['Viento_Label'] = df['Viento'].map(viento_labels)
        df['Oleaje_Label'] = df['Oleaje'].map(oleaje_labels)
        df['Musgo_Label'] = df['Musgo'].map(musgo_labels)
        df['Algas_Label'] = df['Algas'].map(algas_labels)
        df['Cielo_Label'] = df['Cielo'].map(cielo_labels)
        
        return df
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None

# Cargar datos
df = load_and_preprocess_data()

if df is not None:
    # Sidebar para navegación
    st.sidebar.title("🔧 Panel de Control")
    seccion = st.sidebar.selectbox(
        "Selecciona una sección:",
        ["📊 Exploración de Datos", "📈 Visualizaciones Dinámicas", "🤖 Modelos Predictivos"]
    )
    
    # SECCIÓN 1: EXPLORACIÓN DE DATOS
    if seccion == "📊 Exploración de Datos":
        st.header("📊 Exploración de Datos")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Muestras", len(df))
        with col2:
            st.metric("Playas Monitoreadas", df['Lugar Muestreo'].nunique())
        with col3:
            st.metric("Comunas", df['Comuna'].nunique())
        
        # Mostrar datos
        st.subheader("📋 Datos Cargados")
        st.dataframe(df.head(10))
        
        # Estadísticas descriptivas
        st.subheader("📈 Estadísticas Descriptivas")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.dataframe(df[numeric_cols].describe())
    
    # SECCIÓN 2: VISUALIZACIONES DINÁMICAS
    elif seccion == "📈 Visualizaciones Dinámicas":
        st.header("📈 Visualizaciones Dinámicas")
        
        # Clasificar variables
        variables_fisicas = ['Temp. Amb (°C)', 'Temp Agua (°C)', 'Presión (PSI)']
        variables_quimicas = ['pH', 'ORP (mV)', 'O2 Sat (%)', 'O2 (ppm)', 'Cond (µS/cm)', 
                             'Cond Abs (µS/cm)', 'TDS (ppm)', 'Turb (FNU)', 'Fósforo reactivo total (mg/L)']
        variables_observacionales = ['Viento', 'Oleaje', 'Musgo', 'Algas', 'Cielo']
        variables_categoricas = ['Comuna', 'Lugar Muestreo', 'Viento_Label', 'Oleaje_Label', 
                               'Musgo_Label', 'Algas_Label', 'Cielo_Label']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Selección de Variables")
            tipo_variable = st.selectbox(
                "Tipo de Variable:",
                ["Variables Físicas", "Variables Químicas", "Variables Observacionales", "Variables Categóricas"]
            )
            
            if tipo_variable == "Variables Físicas":
                variables_disponibles = variables_fisicas
            elif tipo_variable == "Variables Químicas":
                variables_disponibles = variables_quimicas
            elif tipo_variable == "Variables Observacionales":
                variables_disponibles = variables_observacionales
            else:
                variables_disponibles = variables_categoricas
            
            variable_x = st.selectbox("Variable X:", variables_disponibles)
            variable_y = st.selectbox("Variable Y:", variables_disponibles, index=1 if len(variables_disponibles) > 1 else 0)
        
        with col2:
            st.subheader("📊 Tipo de Gráfico")
            tipo_grafico = st.selectbox(
                "Selecciona el tipo de gráfico:",
                [
                    "Scatter Plot (Dispersión)",
                    "Box Plot (Cajas)",
                    "Histogram (Histograma)",
                    "Bar Plot (Barras)",
                    "Line Plot (Líneas)",
                    "Violin Plot",
                    "Heatmap (Correlación)",
                    "Pair Plot (Matriz de dispersión)"
                ]
            )
            
            color_var = st.selectbox("Colorear por:", ["Ninguno"] + variables_categoricas)
        
        # Generar gráficos
        st.subheader(f"📊 {tipo_grafico}")
        
        try:
            if tipo_grafico == "Scatter Plot (Dispersión)":
                fig = px.scatter(df, x=variable_x, y=variable_y, 
                               color=color_var if color_var != "Ninguno" else None,
                               title=f"Scatter Plot: {variable_x} vs {variable_y}",
                               template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            
            elif tipo_grafico == "Box Plot (Cajas)":
                fig = px.box(df, x=variable_x, y=variable_y,
                           color=color_var if color_var != "Ninguno" else None,
                           title=f"Box Plot: {variable_y} por {variable_x}",
                           template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            
            elif tipo_grafico == "Histogram (Histograma)":
                fig = px.histogram(df, x=variable_x,
                                 color=color_var if color_var != "Ninguno" else None,
                                 title=f"Histograma: {variable_x}",
                                 template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            
            elif tipo_grafico == "Bar Plot (Barras)":
                if variable_x in variables_categoricas:
                    df_count = df[variable_x].value_counts().reset_index()
                    df_count.columns = [variable_x, 'Frecuencia']
                    fig = px.bar(df_count, x=variable_x, y='Frecuencia',
                               title=f"Frecuencia por {variable_x}",
                               template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Para gráfico de barras, selecciona una variable categórica en X")
            
            elif tipo_grafico == "Line Plot (Líneas)":
                if 'Folio' in df.columns:
                    df_sorted = df.sort_values('Folio')
                    fig = px.line(df_sorted, x=df_sorted.index, y=variable_y,
                                title=f"Evolución temporal: {variable_y}",
                                template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No hay variable temporal disponible")
            
            elif tipo_grafico == "Violin Plot":
                fig = px.violin(df, x=variable_x, y=variable_y,
                              color=color_var if color_var != "Ninguno" else None,
                              title=f"Violin Plot: {variable_y} por {variable_x}",
                              template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            
            elif tipo_grafico == "Heatmap (Correlación)":
                numeric_df = df[variables_fisicas + variables_quimicas + variables_observacionales].corr()
                fig = px.imshow(numeric_df, text_auto=True, aspect="auto",
                              title="Matriz de Correlación",
                              template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
            
            elif tipo_grafico == "Pair Plot (Matriz de dispersión)":
                selected_vars = st.multiselect(
                    "Selecciona variables para el pair plot:",
                    variables_fisicas + variables_quimicas,
                    default=variables_fisicas[:3]
                )
                if len(selected_vars) >= 2:
                    fig = px.scatter_matrix(df[selected_vars],
                                          title="Matriz de Dispersión",
                                          template="plotly_white")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Selecciona al menos 2 variables")
        
        except Exception as e:
            st.error(f"Error generando gráfico: {e}")
    
    # SECCIÓN 3: MODELOS PREDICTIVOS
    elif seccion == "🤖 Modelos Predictivos":
        st.header("🤖 Modelos Predictivos")
        
        # Pestañas para los dos modelos
        tab1, tab2 = st.tabs(["🧪 Predictor de Fósforo", "🌱 Predictor de Algas"])
        
        # Preparar datos para modelos CORREGIDO
        @st.cache_data
        def prepare_model_data():
            # Excluir Folio y Lugar Muestreo como solicita el usuario
            features_cols = ['Comuna', 'Temp. Amb (°C)', 'pH', 'ORP (mV)', 'O2 Sat (%)', 'O2 (ppm)',
                           'Cond (µS/cm)', 'Cond Abs (µS/cm)', 'TDS (ppm)', 'Turb (FNU)', 
                           'Temp Agua (°C)', 'Presión (PSI)', 'Viento', 'Oleaje', 'Musgo', 'Cielo']
            
            # Verificar que todas las columnas existen
            missing_cols = [col for col in features_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Columnas faltantes: {missing_cols}")
                return None, None
            
            model_df = df[features_cols + ['Fósforo reactivo total (mg/L)', 'Algas']].copy()
            
            # Mostrar información de debug
            st.write("📋 **Debug: Información de los datos**")
            st.write(f"Filas antes de limpiar: {len(model_df)}")
            st.write(f"Valores nulos por columna:")
            null_counts = model_df.isnull().sum()
            for col, count in null_counts.items():
                if count > 0:
                    st.write(f"  - {col}: {count} valores nulos")
            
            # Remover filas con valores nulos
            model_df = model_df.dropna()
            st.write(f"Filas después de limpiar: {len(model_df)}")
            
            if len(model_df) < 10:
                st.error("⚠️ Muy pocas muestras válidas para entrenar modelos. Verifica los datos.")
                return None, None
            
            # Codificar Comuna
            le_comuna = LabelEncoder()
            model_df['Comuna_encoded'] = le_comuna.fit_transform(model_df['Comuna'])
            
            # Crear variable binaria para algas
            model_df['Algas_presente'] = (model_df['Algas'] > 0).astype(int)
            
            # Verificar tipos de datos
            st.write("🔍 **Tipos de datos verificados:**")
            for col in ['Comuna_encoded', 'Temp. Amb (°C)', 'pH', 'Fósforo reactivo total (mg/L)']:
                if col in model_df.columns:
                    st.write(f"  - {col}: {model_df[col].dtype}")
            
            return model_df, le_comuna
        
        model_df, le_comuna = prepare_model_data()
        
        if model_df is None:
            st.error("❌ No se pudieron preparar los datos para los modelos.")
        else:
            # MODELO DE FÓSFORO
            with tab1:
                st.subheader("🧪 Predicción de Concentración de Fósforo")
                
                # Entrenar modelo de fósforo CORREGIDO
                @st.cache_resource
                def train_phosphorus_model():
                    try:
                        feature_cols = ['Comuna_encoded', 'Temp. Amb (°C)', 'pH', 'ORP (mV)', 'O2 Sat (%)', 
                                      'O2 (ppm)', 'Cond (µS/cm)', 'Cond Abs (µS/cm)', 'TDS (ppm)', 
                                      'Turb (FNU)', 'Temp Agua (°C)', 'Presión (PSI)', 'Viento', 'Oleaje', 
                                      'Musgo', 'Cielo']
                        
                        X = model_df[feature_cols]
                        y = model_df['Fósforo reactivo total (mg/L)']
                        
                        # Verificar que no hay valores nulos ni infinitos
                        if X.isnull().any().any() or y.isnull().any():
                            raise ValueError("Hay valores nulos en los datos de entrenamiento")
                        
                        if not np.isfinite(X.values).all() or not np.isfinite(y.values).all():
                            raise ValueError("Hay valores infinitos en los datos de entrenamiento")
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Random Forest
                        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                        rf_model.fit(X_train, y_train)
                        
                        y_pred = rf_model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        return rf_model, mse, r2, feature_cols
                    
                    except Exception as e:
                        st.error(f"Error entrenando modelo de fósforo: {e}")
                        return None, None, None, None
                
                phos_model, phos_mse, phos_r2, phos_features = train_phosphorus_model()
                
                if phos_model is None:
                    st.error("❌ No se pudo entrenar el modelo de fósforo.")
                else:
                    # Mostrar métricas del modelo
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Error Cuadrático Medio", f"{phos_mse:.6f}")
                    with col2:
                        st.metric("R² Score", f"{phos_r2:.3f}")
                    
                    # Correlaciones estadísticas
                    st.subheader("📊 Correlaciones con Fósforo")
                    correlations = model_df[['Fósforo reactivo total (mg/L)', 'Temp. Amb (°C)', 'pH', 
                                           'O2 Sat (%)', 'Turb (FNU)']].corr()['Fósforo reactivo total (mg/L)'].sort_values(key=abs, ascending=False)
                    
                    for var, corr in correlations.items():
                        if var != 'Fósforo reactivo total (mg/L)':
                            if abs(corr) > 0.1:
                                direction = "aumenta" if corr > 0 else "disminuye"
                                st.info(f"📈 Cuando {var} aumenta, el fósforo {direction} (correlación: {corr:.3f})")
                    
                    # Predictor interactivo
                    st.subheader("🔮 Hacer Predicción")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Variables Físicas**")
                        temp_amb = st.slider("Temperatura Ambiente (°C)", 10.0, 30.0, 20.0)
                        temp_agua = st.slider("Temperatura Agua (°C)", 10.0, 30.0, 20.0)
                        presion = st.slider("Presión (PSI)", 13.0, 15.0, 14.3)
                    
                    with col2:
                        st.write("**Variables Químicas**")
                        ph_val = st.slider("pH", 6.0, 9.0, 7.5)
                        orp_val = st.slider("ORP (mV)", 150, 300, 220)
                        o2_sat = st.slider("O2 Saturación (%)", 70, 120, 95)
                        o2_ppm = st.slider("O2 (ppm)", 6.0, 12.0, 8.5)
                        cond = st.slider("Conductividad (µS/cm)", 40, 80, 60)
                        cond_abs = st.slider("Conductividad Abs (µS/cm)", 40, 80, 55)
                        tds = st.slider("TDS (ppm)", 20, 50, 30)
                        turb = st.slider("Turbidez (FNU)", 0.0, 5.0, 0.5)
                    
                    with col3:
                        st.write("**Variables Observacionales**")
                        comuna_pred = st.selectbox("Comuna", ["Pucón", "Villarrica"])
                        viento_pred = st.selectbox("Viento", [0, 1, 2, 3], format_func=lambda x: {0: 'Sin viento', 1: 'Leve', 2: 'Moderado', 3: 'Fuerte'}[x])
                        oleaje_pred = st.selectbox("Oleaje", [0, 1, 2, 3], format_func=lambda x: {0: 'Sin oleaje', 1: 'Leve', 2: 'Moderado', 3: 'Fuerte'}[x])
                        musgo_pred = st.selectbox("Musgo", [0, 1, 2, 3], format_func=lambda x: {0: 'Sin musgo', 1: 'Verde', 2: 'Pardo', 3: 'Ambos'}[x])
                        cielo_pred = st.selectbox("Cielo", [0, 1, 2], format_func=lambda x: {0: 'Soleado', 1: 'Parcial', 2: 'Nublado'}[x])
                    
                    if st.button("🔮 Predecir Concentración de Fósforo"):
                        # Preparar datos para predicción
                        comuna_encoded = le_comuna.transform([comuna_pred])[0]
                        input_data = np.array([[comuna_encoded, temp_amb, ph_val, orp_val, o2_sat, o2_ppm,
                                              cond, cond_abs, tds, turb, temp_agua, presion, 
                                              viento_pred, oleaje_pred, musgo_pred, cielo_pred]])
                        
                        prediction = phos_model.predict(input_data)[0]
                        
                        st.success(f"🧪 **Concentración de Fósforo Predicha: {prediction:.6f} mg/L**")
                        
                        if prediction > 0.02:
                            st.warning("⚠️ Concentración alta de fósforo detectada")
                        elif prediction < 0:
                            st.info("ℹ️ Concentración muy baja o no detectable")
                        else:
                            st.info("✅ Concentración normal de fósforo")
            
            # MODELO DE ALGAS
            with tab2:
                st.subheader("🌱 Predicción de Presencia de Algas")
                
                # Entrenar modelo de algas CORREGIDO
                @st.cache_resource
                def train_algae_model():
                    try:
                        feature_cols = ['Comuna_encoded', 'Temp. Amb (°C)', 'pH', 'ORP (mV)', 'O2 Sat (%)', 
                                      'O2 (ppm)', 'Cond (µS/cm)', 'Cond Abs (µS/cm)', 'TDS (ppm)', 
                                      'Turb (FNU)', 'Temp Agua (°C)', 'Presión (PSI)', 'Viento', 'Oleaje', 
                                      'Musgo', 'Cielo']
                        
                        X = model_df[feature_cols]
                        y = model_df['Algas_presente']
                        
                        # Verificar que no hay valores nulos ni infinitos
                        if X.isnull().any().any() or y.isnull().any():
                            raise ValueError("Hay valores nulos en los datos de entrenamiento")
                        
                        if not np.isfinite(X.values).all():
                            raise ValueError("Hay valores infinitos en los datos de entrenamiento")
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Random Forest Classifier
                        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                        rf_model.fit(X_train, y_train)
                        
                        y_pred = rf_model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        return rf_model, accuracy, feature_cols
                    
                    except Exception as e:
                        st.error(f"Error entrenando modelo de algas: {e}")
                        return None, None, None
                
                algae_model, algae_accuracy, algae_features = train_algae_model()
                
                if algae_model is None:
                    st.error("❌ No se pudo entrenar el modelo de algas.")
                else:
                    # Mostrar métricas del modelo
                    st.metric("Precisión del Modelo", f"{algae_accuracy:.3f}")
                    
                    # Correlaciones estadísticas
                    st.subheader("📊 Correlaciones con Presencia de Algas")
                    correlations_algae = model_df[['Algas_presente', 'Temp. Amb (°C)', 'pH', 'O2 Sat (%)', 
                                                 'Turb (FNU)', 'Fósforo reactivo total (mg/L)']].corr()['Algas_presente'].sort_values(key=abs, ascending=False)
                    
                    for var, corr in correlations_algae.items():
                        if var != 'Algas_presente':
                            if abs(corr) > 0.1:
                                direction = "aumenta" if corr > 0 else "disminuye"
                                percentage = abs(corr) * 100
                                st.info(f"📈 Cuando {var} aumenta, la probabilidad de algas {direction} (~{percentage:.1f}% de correlación)")
                    
                    # Predictor interactivo
                    st.subheader("🔮 Hacer Predicción")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Variables Físicas**")
                        temp_amb_a = st.slider("Temperatura Ambiente (°C)", 10.0, 30.0, 20.0, key="temp_amb_algae")
                        temp_agua_a = st.slider("Temperatura Agua (°C)", 10.0, 30.0, 20.0, key="temp_agua_algae")
                        presion_a = st.slider("Presión (PSI)", 13.0, 15.0, 14.3, key="presion_algae")
                    
                    with col2:
                        st.write("**Variables Químicas**")
                        ph_val_a = st.slider("pH", 6.0, 9.0, 7.5, key="ph_algae")
                        orp_val_a = st.slider("ORP (mV)", 150, 300, 220, key="orp_algae")
                        o2_sat_a = st.slider("O2 Saturación (%)", 70, 120, 95, key="o2_sat_algae")
                        o2_ppm_a = st.slider("O2 (ppm)", 6.0, 12.0, 8.5, key="o2_ppm_algae")
                        cond_a = st.slider("Conductividad (µS/cm)", 40, 80, 60, key="cond_algae")
                        cond_abs_a = st.slider("Conductividad Abs (µS/cm)", 40, 80, 55, key="cond_abs_algae")
                        tds_a = st.slider("TDS (ppm)", 20, 50, 30, key="tds_algae")
                        turb_a = st.slider("Turbidez (FNU)", 0.0, 5.0, 0.5, key="turb_algae")
                    
                    with col3:
                        st.write("**Variables Observacionales**")
                        comuna_pred_a = st.selectbox("Comuna", ["Pucón", "Villarrica"], key="comuna_algae")
                        viento_pred_a = st.selectbox("Viento", [0, 1, 2, 3], format_func=lambda x: {0: 'Sin viento', 1: 'Leve', 2: 'Moderado', 3: 'Fuerte'}[x], key="viento_algae")
                        oleaje_pred_a = st.selectbox("Oleaje", [0, 1, 2, 3], format_func=lambda x: {0: 'Sin oleaje', 1: 'Leve', 2: 'Moderado', 3: 'Fuerte'}[x], key="oleaje_algae")
                        musgo_pred_a = st.selectbox("Musgo", [0, 1, 2, 3], format_func=lambda x: {0: 'Sin musgo', 1: 'Verde', 2: 'Pardo', 3: 'Ambos'}[x], key="musgo_algae")
                        cielo_pred_a = st.selectbox("Cielo", [0, 1, 2], format_func=lambda x: {0: 'Soleado', 1: 'Parcial', 2: 'Nublado'}[x], key="cielo_algae")
                    
                    if st.button("🔮 Predecir Presencia de Algas"):
                        # Preparar datos para predicción
                        comuna_encoded_a = le_comuna.transform([comuna_pred_a])[0]
                        input_data_a = np.array([[comuna_encoded_a, temp_amb_a, ph_val_a, orp_val_a, o2_sat_a, o2_ppm_a,
                                                cond_a, cond_abs_a, tds_a, turb_a, temp_agua_a, presion_a, 
                                                viento_pred_a, oleaje_pred_a, musgo_pred_a, cielo_pred_a]])
                        
                        prediction_algae = algae_model.predict(input_data_a)[0]
                        probability = algae_model.predict_proba(input_data_a)[0]
                        
                        if prediction_algae == 1:
                            st.error(f"🌱 **PRESENCIA DE ALGAS DETECTADA** (Probabilidad: {probability[1]:.2%})")
                            st.warning("⚠️ Se recomienda monitoreo adicional del cuerpo de agua")
                        else:
                            st.success(f"✅ **NO SE DETECTA PRESENCIA DE ALGAS** (Probabilidad: {probability[0]:.2%})")
                            st.info("ℹ️ Condiciones del agua aparentemente normales")

else:
    st.error("No se pudo cargar el archivo CSV. Asegúrate de que el archivo 'Consolidado Entrenamiento  Tabla Completa 1.csv' esté disponible.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>🌊 Sistema de Análisis de Calidad del Agua</strong></p>
    <p>Región de la Araucanía - Lagos Pucón y Villarrica</p>
    <p><em>Desarrollado para monitoreo ambiental y divulgación científica</em></p>
</div>
""", unsafe_allow_html=True)
