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
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Monitoreo Lago Villarrica",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🌊 Sistema de Monitoreo ")
st.subheader("Lago Villarrica - Región de la Araucanía")

# SELECCIÓN DE DATASET AL INICIO
st.header("📊 Selección de Dataset")
dataset_option = st.selectbox(
    "Selecciona el conjunto de datos a analizar:",
    ["Dataset Completo (con Fósforo)", "Dataset con Nitrógeno (limitado)"],
    help="Elige entre el dataset completo con análisis por comuna o el dataset con nitrógeno (datos limitados)"
)

# Mostrar advertencia para dataset de nitrógeno
if dataset_option == "Dataset con Nitrógeno (limitado)":
    st.warning("⚠️ **Limitaciones del Dataset de Nitrógeno:**")
    st.info("""
    - **Datos limitados**: Menor cantidad de muestras disponibles
    - **Sin análisis por comuna**: Insuficientes datos para análisis detallado por ubicación
    - **Sin análisis por playas**: Datos agregados sin diferenciación geográfica
    - **Modelos simplificados**: Capacidad predictiva reducida debido al tamaño de muestra
    """)

# Función para cargar y preprocesar datos MEJORADA
@st.cache_data
def load_and_preprocess_data(dataset_type="completo"):
    """Carga y preprocesa los datos del CSV según el tipo seleccionado"""
    try:
        df = None
        used_file = None
        
        if dataset_type == "completo":
            # Lista de posibles nombres de archivo CSV para dataset completo
            possible_files = [
                'Consolidado Entrenamiento - Tabla Fechas.csv',
                'Consolidado Entrenamiento - Tabla Completa (1).csv',
                'data.csv',
                'dataset.csv'
            ]
        else:
            # Dataset con nitrógeno
            possible_files = [
                'Tabla con Nitrogeno.csv',
                'Nitrogeno.csv',
                'nitrogen_data.csv'
            ]
        
        # Intentar cargar cada archivo posible
        for filename in possible_files:
            try:
                df = pd.read_csv(filename)
                used_file = filename
                st.success(f"✅ Datos cargados desde: {filename}")
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                st.warning(f"⚠️ Error con {filename}: {e}")
                continue
        
        if df is None:
            st.error("❌ No se encontró ningún archivo CSV válido. Archivos esperados:")
            for f in possible_files:
                st.write(f"   - {f}")
            return None, dataset_type
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip()
        
        # Identificar columnas que pueden tener valores numéricos con comas
        if dataset_type == "completo":
            categorical_cols = ['Día', 'Folio', 'Lugar Muestreo', 'Comuna']
        else:
            # Para dataset de nitrógeno, sin lugar ni comuna en análisis
            categorical_cols = ['Día', 'Folio', 'Lugar Muestreo', 'Comuna']
        
        for col in df.columns:
            if col not in categorical_cols:
                # Intentar convertir a numérico, reemplazando comas por puntos
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        # Procesar fechas si existe la columna Día
        if 'Día' in df.columns:
            try:
                # Mostrar información inicial para debugging
                st.info(f"🔄 Procesando fechas... Formato original: {df['Día'].head(3).tolist()}")
                
                # Intentar diferentes formatos de fecha más específicos
                df['Fecha'] = pd.to_datetime(df['Día'], errors='coerce', dayfirst=True, format=None)
                
                # Si el primer método no funciona, intentar formatos específicos
                if df['Fecha'].isna().all():
                    df['Fecha'] = pd.to_datetime(df['Día'], errors='coerce', format='%Y-%m-%d')
                    
                if df['Fecha'].isna().all():
                    df['Fecha'] = pd.to_datetime(df['Día'], errors='coerce', format='%d/%m/%Y')
                    
                if df['Fecha'].isna().all():
                    df['Fecha'] = pd.to_datetime(df['Día'], errors='coerce', format='%d-%m-%Y')
                
                # Verificar si hay fechas válidas y crear campos derivados
                valid_dates = df['Fecha'].notna().sum()
                
                if valid_dates > 0:
                    df_with_dates = df[df['Fecha'].notna()].copy()
                    df.loc[df['Fecha'].notna(), 'Día_Semana'] = df_with_dates['Fecha'].dt.day_name()
                    df.loc[df['Fecha'].notna(), 'Mes'] = df_with_dates['Fecha'].dt.month_name()
                    df.loc[df['Fecha'].notna(), 'Día_Mes'] = df_with_dates['Fecha'].dt.day
                    df.loc[df['Fecha'].notna(), 'Semana'] = df_with_dates['Fecha'].dt.isocalendar().week
                    
                    st.success(f"✅ Fechas procesadas exitosamente: {valid_dates}/{len(df)} válidas")
                else:
                    st.warning("⚠️ No se pudieron procesar fechas válidas - revisar formato de datos")
                    df['Fecha'] = pd.NaT
                    
            except Exception as e:
                st.error(f"❌ Error procesando fechas: {e}")
                df['Fecha'] = pd.NaT
        
        # Limpiar nombres de lugares y comunas (solo para mostrar, no para análisis en nitrógeno)
        if 'Lugar Muestreo' in df.columns:
            df['Lugar Muestreo'] = df['Lugar Muestreo'].str.strip().str.title()
        if 'Comuna' in df.columns:
            df['Comuna'] = df['Comuna'].str.strip().str.title()
        
        # Crear variables categóricas con labels descriptivos
        viento_labels = {0: 'Sin viento', 1: 'Viento leve', 2: 'Viento moderado', 3: 'Viento fuerte'}
        oleaje_labels = {0: 'Sin oleaje', 1: 'Oleaje leve', 2: 'Oleaje moderado', 3: 'Oleaje fuerte'}
        musgo_labels = {0: 'Sin musgo', 1: 'Musgo verde', 2: 'Musgo pardo', 3: 'Ambos tipos'}
        algas_labels = {0: 'Sin algas', 1: 'Pocas algas', 2: 'Moderadas algas', 3: 'Muchas algas'}
        cielo_labels = {0: 'Soleado', 1: 'Parcial', 2: 'Nublado'}
        
        if 'Viento' in df.columns:
            df['Viento_Label'] = df['Viento'].map(viento_labels)
        if 'Oleaje' in df.columns:
            df['Oleaje_Label'] = df['Oleaje'].map(oleaje_labels)
        if 'Musgo' in df.columns:
            df['Musgo_Label'] = df['Musgo'].map(musgo_labels)
        if 'Algas' in df.columns:
            df['Algas_Label'] = df['Algas'].map(algas_labels)
        if 'Cielo' in df.columns:
            df['Cielo_Label'] = df['Cielo'].map(cielo_labels)
        
        return df, dataset_type
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return None, dataset_type

# Función para generar insights avanzados (modificada para nitrógeno)
def generate_advanced_insights(df, target_var, dataset_type):
    """Genera insights avanzados basados en rangos y correlaciones"""
    insights = []
    
    # Obtener variables numéricas
    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_vars = [col for col in numeric_vars if col != target_var and col in df.columns]
    
    if target_var not in df.columns:
        return insights
    
    # Para dataset de nitrógeno, excluir análisis por comuna/lugar
    if dataset_type == "nitrogeno":
        numeric_vars = [col for col in numeric_vars if col not in ['Comuna_encoded']]
        st.info("🔬 **Análisis simplificado**: Sin diferenciación por comuna debido a datos limitados")
    
    for var in numeric_vars[:8]:
        try:
            correlation = df[var].corr(df[target_var])
            
            if abs(correlation) > 0.1:
                quartiles = df[var].quantile([0.25, 0.5, 0.75]).values
                
                for i, (q_low, q_high) in enumerate(zip([df[var].min()] + quartiles.tolist(), 
                                                       quartiles.tolist() + [df[var].max()])):
                    
                    mask = (df[var] >= q_low) & (df[var] <= q_high)
                    if mask.sum() > 3:  # Reducir umbral para dataset de nitrógeno
                        target_values = df[mask][target_var]
                        
                        if target_var == 'Nitrógeno amoniacal (mg/L)':
                            avg_val = target_values.mean()
                            std_val = target_values.std()
                            if not np.isnan(avg_val) and avg_val > 0:
                                range_name = f"{q_low:.2f} - {q_high:.2f}"
                                insights.append({
                                    'tipo': 'rango_nitrogeno',
                                    'variable': var,
                                    'rango': range_name,
                                    'promedio': avg_val,
                                    'desviacion': std_val,
                                    'n_muestras': mask.sum(),
                                    'descripcion': f"Cuando {var} está entre {range_name}, nitrógeno promedio: {avg_val:.4f} mg/L ± {std_val:.4f} ({mask.sum()} muestras)"
                                })
                        
                        elif target_var == 'Algas' and dataset_type == "completo":
                            prob = (target_values > 0).mean() * 100
                            if prob > 0:
                                range_name = f"{q_low:.2f} - {q_high:.2f}"
                                insights.append({
                                    'tipo': 'rango_algas',
                                    'variable': var,
                                    'rango': range_name,
                                    'probabilidad': prob,
                                    'n_muestras': mask.sum(),
                                    'descripcion': f"Cuando {var} está entre {range_name}, hay {prob:.1f}% probabilidad de presencia de algas ({mask.sum()} muestras)"
                                })
                        
                        elif target_var == 'Fósforo reactivo total (mg/L)' and dataset_type == "completo":
                            avg_val = target_values.mean()
                            std_val = target_values.std()
                            if not np.isnan(avg_val) and avg_val > 0:
                                range_name = f"{q_low:.2f} - {q_high:.2f}"
                                insights.append({
                                    'tipo': 'rango_fosforo',
                                    'variable': var,
                                    'rango': range_name,
                                    'promedio': avg_val,
                                    'desviacion': std_val,
                                    'n_muestras': mask.sum(),
                                    'descripcion': f"Cuando {var} está entre {range_name}, fósforo promedio: {avg_val:.4f} mg/L ± {std_val:.4f} ({mask.sum()} muestras)"
                                })
        
        except Exception as e:
            continue
    
    return insights

# Función para detectar factores de riesgo (modificada para nitrógeno)
def detect_risk_factors(df, dataset_type):
    """Detecta factores de riesgo automáticamente"""
    risk_factors = []
    
    # Definir umbrales de riesgo según el dataset
    if dataset_type == "nitrogeno":
        risk_conditions = [
            ('Nitrógeno amoniacal (mg/L)', '>', 0.5, 'Alto nitrógeno amoniacal'),
            ('pH', '>', 8.5, 'pH muy alcalino'),
            ('pH', '<', 6.5, 'pH muy ácido'),
            ('Temp Agua (°C)', '>', 25, 'Temperatura alta del agua'),
            ('Turb (FNU)', '>', 2, 'Alta turbidez'),
            ('Algas', '>', 1, 'Presencia significativa de algas')
        ]
    else:
        risk_conditions = [
            ('Fósforo reactivo total (mg/L)', '>', 0.02, 'Alto fósforo'),
            ('pH', '>', 8.5, 'pH muy alcalino'),
            ('pH', '<', 6.5, 'pH muy ácido'),
            ('Temp Agua (°C)', '>', 25, 'Temperatura alta del agua'),
            ('Turb (FNU)', '>', 2, 'Alta turbidez'),
            ('Algas', '>', 1, 'Presencia significativa de algas')
        ]
    
    for var, operator, threshold, description in risk_conditions:
        if var in df.columns:
            try:
                if operator == '>':
                    risk_mask = df[var] > threshold
                else:
                    risk_mask = df[var] < threshold
                
                risk_percentage = (risk_mask.sum() / len(df)) * 100
                
                # Reducir umbral para dataset de nitrógeno
                min_threshold = 3 if dataset_type == "nitrogeno" else 5
                if risk_percentage > min_threshold:
                    risk_factors.append({
                        'factor': description,
                        'variable': var,
                        'threshold': threshold,
                        'percentage': risk_percentage,
                        'count': risk_mask.sum(),
                        'operator': operator
                    })
            except:
                continue
    
    return sorted(risk_factors, key=lambda x: x['percentage'], reverse=True)

# Determinar tipo de dataset
dataset_type = "nitrogeno" if dataset_option == "Dataset con Nitrógeno (limitado)" else "completo"

# Cargar datos
df, current_dataset_type = load_and_preprocess_data(dataset_type)

if df is not None:
    # Sidebar para navegación
    st.sidebar.title("🔧 Panel de Control")
    
    # Mostrar información del dataset actual
    if current_dataset_type == "nitrogeno":
        st.sidebar.info("🔬 **Dataset de Nitrógeno**\n- Análisis simplificado\n- Sin diferenciación geográfica")
    else:
        st.sidebar.success("📊 **Dataset Completo**\n- Análisis por comuna\n- Modelos avanzados")
    
    seccion = st.sidebar.selectbox(
        "Selecciona una sección:",
        ["📊 Exploración de Datos", "📈 Análisis Temporal", "🔍 Insights Avanzados", "🤖 Modelos Predictivos"]
    )
    
    # SECCIÓN 1: EXPLORACIÓN DE DATOS
    if seccion == "📊 Exploración de Datos":
        st.header("📊 Exploración de Datos")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Muestras", len(df))
        with col2:
            if current_dataset_type == "completo" and 'Lugar Muestreo' in df.columns:
                st.metric("Playas Monitoreadas", df['Lugar Muestreo'].nunique())
            else:
                st.metric("Parámetro Principal", "Nitrógeno" if current_dataset_type == "nitrogeno" else "Fósforo")
        with col3:
            if current_dataset_type == "completo" and 'Comuna' in df.columns:
                st.metric("Comunas", df['Comuna'].nunique())
            else:
                st.metric("Análisis", "Simplificado" if current_dataset_type == "nitrogeno" else "Completo")
        with col4:
            if 'Fecha' in df.columns:
                date_range = (df['Fecha'].max() - df['Fecha'].min()).days
                st.metric("Días de Monitoreo", date_range)
        
        # Mostrar información específica del dataset
        if current_dataset_type == "nitrogeno":
            st.info("🔬 **Dataset de Nitrógeno**: Este conjunto de datos incluye mediciones de nitrógeno amoniacal pero tiene limitaciones para análisis geográfico detallado.")
        
        # Mostrar información temporal si existe
        if 'Fecha' in df.columns:
            st.subheader("📅 Información Temporal")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Fecha inicial:** {df['Fecha'].min().strftime('%d/%m/%Y')}")
                st.write(f"**Fecha final:** {df['Fecha'].max().strftime('%d/%m/%Y')}")
            with col2:
                st.write(f"**Muestras por día:** {len(df) / date_range:.1f}")
                st.write(f"**Días con muestras:** {df['Fecha'].nunique()}")
        
        # Mostrar datos
        st.subheader("📋 Datos Cargados")
        st.dataframe(df.head(10))
        
        # Estadísticas descriptivas
        st.subheader("📈 Estadísticas Descriptivas")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.dataframe(df[numeric_cols].describe())
    
    # SECCIÓN 2: ANÁLISIS TEMPORAL
    elif seccion == "📈 Análisis Temporal":
        st.header("📈 Análisis Temporal")
        
        # [El resto del código de análisis temporal permanece igual]
        # Verificar si hay fechas válidas
        if 'Fecha' not in df.columns or df['Fecha'].notna().sum() == 0:
            st.error("❌ No se encontró información de fechas válidas en los datos")
            st.info("📋 Para usar análisis temporal, asegúrate de que:")
            st.write("   - La columna 'Día' contenga fechas válidas")
            st.write("   - El formato de fecha sea reconocible (dd/mm/aaaa, yyyy-mm-dd, etc.)")
        else:
            # Filtrar solo datos con fechas válidas
            df_temporal = df[df['Fecha'].notna()].copy()
            
            st.info(f"📊 Datos temporales disponibles: {len(df_temporal)} de {len(df)} registros")
            
            # Obtener todas las variables disponibles
            numeric_vars = df_temporal.select_dtypes(include=[np.number]).columns.tolist()
            categorical_vars = df_temporal.select_dtypes(include=['object']).columns.tolist()
            
            # Remover variables que no queremos en las opciones
            numeric_vars = [v for v in numeric_vars if v not in ['Día_Mes', 'Semana']]
            categorical_vars = [v for v in categorical_vars if v not in ['Folio', 'Día']]
            
            st.subheader("🎯 Configuración de Análisis Temporal")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Selección de Variables**")
                variable_y = st.selectbox("Variable a analizar:", numeric_vars)
                
                # Opción de agrupación temporal
                if current_dataset_type == "completo":
                    time_grouping = st.selectbox(
                        "Agrupación temporal:",
                        ["Sin agrupar", "Por día", "Por semana", "Por mes", "Por lugar", "Por comuna"]
                    )
                else:
                    time_grouping = st.selectbox(
                        "Agrupación temporal:",
                        ["Sin agrupar", "Por día", "Por semana", "Por mes"]
                    )
                
                # Filtros adicionales
                filter_by = st.selectbox("Filtrar por:", ["Ninguno"] + categorical_vars)
                
            with col2:
                st.write("**Opciones de Visualización**")
                chart_type = st.selectbox(
                    "Tipo de gráfico:",
                    ["Línea Temporal", "Dispersión Temporal", "Box Plot Temporal", 
                     "Histograma por Período", "Tendencia con Regresión"]
                )
                
                # Variables adicionales
                color_by = st.selectbox("Colorear por:", ["Ninguno"] + categorical_vars)
                if len(numeric_vars) > 1:
                    size_options = ["Ninguno"] + [v for v in numeric_vars if v != variable_y]
                    size_by = st.selectbox("Tamaño por:", size_options)
                else:
                    size_by = "Ninguno"
            
            # [El resto del código de análisis temporal permanece igual...]
            # [Incluir toda la lógica de visualización temporal aquí]
    
    # SECCIÓN 3: INSIGHTS AVANZADOS (modificada para nitrógeno)
    elif seccion == "🔍 Insights Avanzados":
        st.header("🔍 Insights Avanzados")
        
        # Seleccionar variable objetivo según el dataset
        if current_dataset_type == "nitrogeno":
            st.info("🔬 **Análisis de Nitrógeno**: Enfoque especializado en nitrógeno amoniacal")
            target_options = ['Nitrógeno amoniacal (mg/L)']
            if 'Algas' in df.columns:
                target_options.append('Algas')
        else:
            target_options = ['Algas', 'Fósforo reactivo total (mg/L)']
        
        target_variable = st.selectbox("Selecciona variable objetivo para análisis:", target_options)
        
        if st.button("🔍 Generar Análisis Avanzado"):
            with st.spinner("Analizando patrones y generando insights..."):
                
                # Generar insights por rangos
                insights = generate_advanced_insights(df, target_variable, current_dataset_type)
                
                st.subheader(f"📊 Análisis de Rangos para {target_variable}")
                
                if insights:
                    # Organizar insights por tipo
                    if target_variable == 'Nitrógeno amoniacal (mg/L)':
                        nitrogeno_insights = [i for i in insights if i['tipo'] == 'rango_nitrogeno']
                        
                        if nitrogeno_insights:
                            st.write("**🔬 Factores que Influyen en la Concentración de Nitrógeno Amoniacal:**")
                            
                            # Ordenar por concentración promedio
                            nitrogeno_insights.sort(key=lambda x: x['promedio'], reverse=True)
                            
                            for insight in nitrogeno_insights[:10]:
                                conc = insight['promedio']
                                if conc > 1.0:  # Umbral alto para nitrógeno
                                    st.error(f"⚠️ {insight['descripcion']}")
                                elif conc > 0.5:  # Umbral moderado
                                    st.warning(f"🔶 {insight['descripcion']}")
                                else:
                                    st.info(f"ℹ️ {insight['descripcion']}")
                    
                    elif target_variable == 'Algas':
                        algas_insights = [i for i in insights if i['tipo'] == 'rango_algas']
                        
                        if algas_insights:
                            st.write("**🌱 Factores que Influyen en la Presencia de Algas:**")
                            algas_insights.sort(key=lambda x: x['probabilidad'], reverse=True)
                            
                            for insight in algas_insights[:10]:
                                prob = insight['probabilidad']
                                if prob > 50:
                                    st.error(f"⚠️ {insight['descripcion']}")
                                elif prob > 30:
                                    st.warning(f"🔶 {insight['descripcion']}")
                                else:
                                    st.info(f"ℹ️ {insight['descripcion']}")
                    
                    elif target_variable == 'Fósforo reactivo total (mg/L)':
                        fosforo_insights = [i for i in insights if i['tipo'] == 'rango_fosforo']
                        
                        if fosforo_insights:
                            st.write("**🧪 Factores que Influyen en la Concentración de Fósforo:**")
                            fosforo_insights.sort(key=lambda x: x['promedio'], reverse=True)
                            
                            for insight in fosforo_insights[:10]:
                                conc = insight['promedio']
                                if conc > 0.02:
                                    st.error(f"⚠️ {insight['descripcion']}")
                                elif conc > 0.01:
                                    st.warning(f"🔶 {insight['descripción']}")
                                else:
                                    st.info(f"ℹ️ {insight['descripcion']}")
                
                # Detectar factores de riesgo
                st.subheader("⚠️ Factores de Riesgo Detectados")
                risk_factors = detect_risk_factors(df, current_dataset_type)
                
                if risk_factors:
                    for risk in risk_factors:
                        if risk['percentage'] > 30:
                            st.error(f"🚨 **{risk['factor']}**: {risk['percentage']:.1f}% de las muestras ({risk['count']}/{len(df)})")
                        elif risk['percentage'] > 15:
                            st.warning(f"⚠️ **{risk['factor']}**: {risk['percentage']:.1f}% de las muestras ({risk['count']}/{len(df)})")
                        else:
                            st.info(f"ℹ️ **{risk['factor']}**: {risk['percentage']:.1f}% de las muestras ({risk['count']}/{len(df)})")
                else:
                    st.success("✅ No se detectaron factores de riesgo significativos")
                
                # Recomendaciones específicas por dataset
                st.subheader("💡 Recomendaciones Automáticas")
                
                recommendations = []
                
                if current_dataset_type == "nitrogeno":
                    recommendations.append("🔬 Análisis enfocado en nitrógeno amoniacal - indicador de contaminación orgánica")
                    recommendations.append("📊 Se recomienda complementar con análisis de fósforo para evaluación completa")
                    recommendations.append("🌊 Monitoreo de fuentes potenciales de nitrógeno (escorrentía agrícola, residuos)")
                
                high_risk_factors = [r for r in risk_factors if r['percentage'] > 20]
                if high_risk_factors:
                    recommendations.append(f"🔍 Monitorear de cerca: {', '.join([r['factor'] for r in high_risk_factors[:3]])}")
                
                if recommendations:
                    for rec in recommendations:
                        st.info(rec)
                else:
                    st.success("✅ Las condiciones del agua se mantienen dentro de parámetros normales")
    
    # SECCIÓN 4: MODELOS PREDICTIVOS (modificada para nitrógeno)
    elif seccion == "🤖 Modelos Predictivos":
        st.header("🤖 Modelos Predictivos")
        
        if current_dataset_type == "nitrogeno":
            st.warning("⚠️ **Limitaciones del Modelo con Dataset de Nitrógeno:**")
            st.info("""
            - **Menor precisión**: Datos limitados afectan la capacidad predictiva
            - **Sin análisis geográfico**: No se incluyen variables de ubicación
            - **Modelo simplificado**: Enfoque únicamente en variables fisicoquímicas
            """)
            
            # Solo modelo de nitrógeno
            tab1 = st.tabs(["🔬 Predictor de Nitrógeno"])[0]
        else:
            # Pestañas para los dos modelos originales
            tab1, tab2 = st.tabs(["🧪 Predictor de Fósforo", "🌱 Predictor de Algas"])
        
        # Preparar datos para modelos
        @st.cache_data
        def prepare_model_data(dataset_type):
            if dataset_type == "nitrogeno":
                # Para nitrógeno: sin comuna ni lugar
                features_cols = ['Temp. Amb (°C)', 'pH', 'ORP (mV)', 'O2 Sat (%)', 'O2 (ppm)',
                               'Cond (µS/cm)', 'Cond Abs (µS/cm)', 'TDS (ppm)', 'Turb (FNU)', 
                               'Temp Agua (°C)', 'Presión (PSI)', 'Viento', 'Oleaje', 'Musgo', 'Cielo']
                
                target_cols = ['Nitrógeno amoniacal (mg/L)']
                if 'Algas' in df.columns:
                    target_cols.append('Algas')
            else:
                # Dataset completo original
                features_cols = ['Comuna', 'Temp. Amb (°C)', 'pH', 'ORP (mV)', 'O2 Sat (%)', 'O2 (ppm)',
                               'Cond (µS/cm)', 'Cond Abs (µS/cm)', 'TDS (ppm)', 'Turb (FNU)', 
                               'Temp Agua (°C)', 'Presión (PSI)', 'Viento', 'Oleaje', 'Musgo', 'Cielo']
                
                target_cols = ['Fósforo reactivo total (mg/L)', 'Algas']
            
            # Verificar que todas las columnas existen
            missing_cols = [col for col in features_cols if col not in df.columns]
            if missing_cols:
                st.error(f"Columnas faltantes: {missing_cols}")
                return None, None
            
            model_df = df[features_cols + target_cols].copy()
            
            # Remover filas con valores nulos
            model_df = model_df.dropna()
            
            min_samples = 5 if dataset_type == "nitrogeno" else 10
            if len(model_df) < min_samples:
                st.error(f"⚠️ Muy pocas muestras válidas para entrenar modelos. (Mínimo: {min_samples})")
                return None, None
            
            # Codificar Comuna solo si existe
            le_comuna = None
            if 'Comuna' in model_df.columns and dataset_type == "completo":
                le_comuna = LabelEncoder()
                model_df['Comuna_encoded'] = le_comuna.fit_transform(model_df['Comuna'])
            
            # Crear variable binaria para algas si existe
            if 'Algas' in model_df.columns:
                model_df['Algas_presente'] = (model_df['Algas'] > 0).astype(int)
            
            return model_df, le_comuna
        
        model_df, le_comuna = prepare_model_data(current_dataset_type)
        
        if model_df is None:
            st.error("❌ No se pudieron preparar los datos para los modelos.")
        else:
            if current_dataset_type == "nitrogeno":
                # MODELO DE NITRÓGENO
                with tab1:
                    st.subheader("🔬 Predicción de Concentración de Nitrógeno Amoniacal")
                    
                    # Entrenar modelo de nitrógeno
                    @st.cache_resource
                    def train_nitrogen_model():
                        try:
                            feature_cols = ['Temp. Amb (°C)', 'pH', 'ORP (mV)', 'O2 Sat (%)', 
                                          'O2 (ppm)', 'Cond (µS/cm)', 'Cond Abs (µS/cm)', 'TDS (ppm)', 
                                          'Turb (FNU)', 'Temp Agua (°C)', 'Presión (PSI)', 'Viento', 'Oleaje', 
                                          'Musgo', 'Cielo']
                            
                            X = model_df[feature_cols]
                            y = model_df['Nitrógeno amoniacal (mg/L)']
                            
                            # Verificar que no hay valores nulos ni infinitos
                            if X.isnull().any().any() or y.isnull().any():
                                raise ValueError("Hay valores nulos en los datos de entrenamiento")
                            
                            if not np.isfinite(X.values).all() or not np.isfinite(y.values).all():
                                raise ValueError("Hay valores infinitos en los datos de entrenamiento")
                            
                            # Para datasets pequeños, usar menos datos de test
                            test_size = 0.15 if len(X) < 50 else 0.2
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                            
                            # Random Forest con parámetros ajustados para dataset pequeño
                            n_estimators = min(50, len(X_train) // 2) if len(X_train) < 100 else 100
                            rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                            rf_model.fit(X_train, y_train)
                            
                            y_pred = rf_model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            
                            return rf_model, mse, r2, feature_cols
                        
                        except Exception as e:
                            st.error(f"Error entrenando modelo de nitrógeno: {e}")
                            return None, None, None, None
                    
                    nitrogen_model, nitrogen_mse, nitrogen_r2, nitrogen_features = train_nitrogen_model()
                    
                    if nitrogen_model is None:
                        st.error("❌ No se pudo entrenar el modelo de nitrógeno.")
                    else:
                        # Mostrar métricas del modelo
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Error Cuadrático Medio", f"{nitrogen_mse:.6f}")
                        with col2:
                            st.metric("R² Score", f"{nitrogen_r2:.3f}")
                        
                        if nitrogen_r2 < 0.3:
                            st.warning("⚠️ **Modelo con precisión limitada** debido al tamaño del dataset")
                        
                        # Correlaciones estadísticas
                        st.subheader("📊 Análisis de Correlaciones con Nitrógeno Amoniacal")
                        correlation_cols = ['Nitrógeno amoniacal (mg/L)', 'Temp. Amb (°C)', 'pH', 
                                          'O2 Sat (%)', 'Turb (FNU)', 'Temp Agua (°C)', 'ORP (mV)']
                        available_cols = [col for col in correlation_cols if col in model_df.columns]
                        
                        if len(available_cols) > 1:
                            correlations = model_df[available_cols].corr()['Nitrógeno amoniacal (mg/L)'].sort_values(key=abs, ascending=False)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Correlaciones Positivas:**")
                                positive_corr = correlations[correlations > 0.1]
                                for var, corr in positive_corr.items():
                                    if var != 'Nitrógeno amoniacal (mg/L)':
                                        st.success(f"📈 {var}: +{corr:.3f}")
                            
                            with col2:
                                st.write("**Correlaciones Negativas:**")
                                negative_corr = correlations[correlations < -0.1]
                                for var, corr in negative_corr.items():
                                    if var != 'Nitrógeno amoniacal (mg/L)':
                                        st.info(f"📉 {var}: {corr:.3f}")
                        
                        # Predictor interactivo simplificado
                        st.subheader("🔮 Hacer Predicción")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**Variables Físicas**")
                            temp_amb = st.slider("Temperatura Ambiente (°C)", 10.0, 30.0, 20.0, key="temp_amb_n")
                            temp_agua = st.slider("Temperatura Agua (°C)", 10.0, 30.0, 20.0, key="temp_agua_n")
                            presion = st.slider("Presión (PSI)", 13.0, 15.0, 14.3, key="presion_n")
                        
                        with col2:
                            st.write("**Variables Químicas**")
                            ph_val = st.slider("pH", 6.0, 9.0, 7.5, key="ph_n")
                            orp_val = st.slider("ORP (mV)", 150, 300, 220, key="orp_n")
                            o2_sat = st.slider("O2 Saturación (%)", 70, 120, 95, key="o2_sat_n")
                            o2_ppm = st.slider("O2 (ppm)", 6.0, 12.0, 8.5, key="o2_ppm_n")
                            cond = st.slider("Conductividad (µS/cm)", 40, 80, 60, key="cond_n")
                            cond_abs = st.slider("Conductividad Abs (µS/cm)", 40, 80, 55, key="cond_abs_n")
                            tds = st.slider("TDS (ppm)", 20, 50, 30, key="tds_n")
                            turb = st.slider("Turbidez (FNU)", 0.0, 5.0, 0.5, key="turb_n")
                        
                        with col3:
                            st.write("**Variables Observacionales**")
                            viento_pred = st.selectbox("Viento", [0, 1, 2, 3], format_func=lambda x: {0: 'Sin viento', 1: 'Leve', 2: 'Moderado', 3: 'Fuerte'}[x], key="viento_n")
                            oleaje_pred = st.selectbox("Oleaje", [0, 1, 2, 3], format_func=lambda x: {0: 'Sin oleaje', 1: 'Leve', 2: 'Moderado', 3: 'Fuerte'}[x], key="oleaje_n")
                            musgo_pred = st.selectbox("Musgo", [0, 1, 2, 3], format_func=lambda x: {0: 'Sin musgo', 1: 'Verde', 2: 'Pardo', 3: 'Ambos'}[x], key="musgo_n")
                            cielo_pred = st.selectbox("Cielo", [0, 1, 2], format_func=lambda x: {0: 'Soleado', 1: 'Parcial', 2: 'Nublado'}[x], key="cielo_n")
                        
                        if st.button("🔮 Predecir Concentración de Nitrógeno"):
                            # Preparar datos para predicción
                            input_data = np.array([[temp_amb, ph_val, orp_val, o2_sat, o2_ppm,
                                                  cond, cond_abs, tds, turb, temp_agua, presion, 
                                                  viento_pred, oleaje_pred, musgo_pred, cielo_pred]])
                            
                            prediction = nitrogen_model.predict(input_data)[0]
                            
                            st.success(f"🔬 **Concentración de Nitrógeno Amoniacal Predicha: {prediction:.4f} mg/L**")
                            
                            if prediction > 1.0:
                                st.error("⚠️ Concentración alta de nitrógeno amoniacal - Posible contaminación orgánica")
                            elif prediction > 0.5:
                                st.warning("🔶 Concentración moderada de nitrógeno amoniacal - Monitoreo recomendado")
                            elif prediction < 0:
                                st.info("ℹ️ Concentración muy baja o no detectable")
                            else:
                                st.success("✅ Concentración normal de nitrógeno amoniacal")
            
            else:
                # MODELO DE FÓSFORO
                with tab1:
                    st.subheader("🧪 Predicción de Concentración de Fósforo")
                    
                    # Entrenar modelo de fósforo
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
                        
                        # Correlaciones estadísticas mejoradas
                        st.subheader("📊 Análisis de Correlaciones con Fósforo")
                        correlations = model_df[['Fósforo reactivo total (mg/L)', 'Temp. Amb (°C)', 'pH', 
                                               'O2 Sat (%)', 'Turb (FNU)', 'Temp Agua (°C)', 'ORP (mV)']].corr()['Fósforo reactivo total (mg/L)'].sort_values(key=abs, ascending=False)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Correlaciones Positivas:**")
                            positive_corr = correlations[correlations > 0.1]
                            for var, corr in positive_corr.items():
                                if var != 'Fósforo reactivo total (mg/L)':
                                    st.success(f"📈 {var}: +{corr:.3f}")
                        
                        with col2:
                            st.write("**Correlaciones Negativas:**")
                            negative_corr = correlations[correlations < -0.1]
                            for var, corr in negative_corr.items():
                                if var != 'Fósforo reactivo total (mg/L)':
                                    st.info(f"📉 {var}: {corr:.3f}")
                        
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
                                st.error("⚠️ Concentración alta de fósforo detectada - Riesgo de eutrofización")
                            elif prediction > 0.01:
                                st.warning("🔶 Concentración moderada de fósforo - Monitoreo recomendado")
                            elif prediction < 0:
                                st.info("ℹ️ Concentración muy baja o no detectable")
                            else:
                                st.success("✅ Concentración normal de fósforo")
                
                # MODELO DE ALGAS
                with tab2:
                    st.subheader("🌱 Predicción de Presencia de Algas")
                    
                    # Entrenar modelo de algas
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
                        
                        # Correlaciones estadísticas mejoradas
                        st.subheader("📊 Análisis de Correlaciones con Presencia de Algas")
                        correlations_algae = model_df[['Algas_presente', 'Temp. Amb (°C)', 'pH', 'O2 Sat (%)', 
                                                     'Turb (FNU)', 'Fósforo reactivo total (mg/L)', 'Temp Agua (°C)']].corr()['Algas_presente'].sort_values(key=abs, ascending=False)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Factores que Favorecen Algas:**")
                            positive_corr = correlations_algae[correlations_algae > 0.1]
                            for var, corr in positive_corr.items():
                                if var != 'Algas_presente':
                                    percentage = abs(corr) * 100
                                    st.success(f"📈 {var}: +{percentage:.1f}%")
                        
                        with col2:
                            st.write("**Factores que Inhiben Algas:**")
                            negative_corr = correlations_algae[correlations_algae < -0.1]
                            for var, corr in negative_corr.items():
                                if var != 'Algas_presente':
                                    percentage = abs(corr) * 100
                                    st.info(f"📉 {var}: -{percentage:.1f}%")
                        
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
                                
                                # Recomendaciones específicas
                                if temp_agua_a > 22:
                                    st.info("🌡️ Temperatura del agua elevada favorece el crecimiento de algas")
                                if ph_val_a > 8:
                                    st.info("🧪 pH alcalino puede promover proliferación algal")
                                    
                            else:
                                st.success(f"✅ **NO SE DETECTA PRESENCIA DE ALGAS** (Probabilidad: {probability[0]:.2%})")
                                st.info("ℹ️ Condiciones del agua aparentemente normales")

else:
    st.error("❌ No se pudo cargar el archivo CSV.")
    if dataset_type == "nitrogeno":
        st.info("📁 Archivos CSV esperados para dataset de nitrógeno:")
        st.write("   - Tabla con Nitrogeno.csv")
        st.write("   - Nitrogeno.csv")
        st.write("   - nitrogen_data.csv")
    else:
        st.info("📁 Archivos CSV esperados para dataset completo:")
        st.write("   - Consolidado Entrenamiento - Tabla Fechas.csv")
        st.write("   - Consolidado Entrenamiento - Tabla Completa (1).csv")
        st.write("   - data.csv")
        st.write("   - dataset.csv")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>🌊 Sistema de Análisis de Monitoreo del Agua</strong></p>
    <p>Región de la Araucanía - Lago Villarrica</p>
    <p><em>Desarrollado por BIOREN UFRO con el apoyo de Ciencia 2030</em></p>
</div>
""", unsafe_allow_html=True)