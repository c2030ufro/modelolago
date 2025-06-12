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
    page_title="Análisis de Calidad del Agua - Lago Villarrica",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🌊 Sistema de Análisis de Calidad del Agua")
st.subheader("Lago Villarrica - Región de la Araucanía")

# Función para cargar y preprocesar datos MEJORADA
@st.cache_data
def load_and_preprocess_data():
    """Carga y preprocesa los datos del CSV"""
    try:
        # Lista de posibles nombres de archivo CSV
        possible_files = [
            'Consolidado Entrenamiento  Tabla Fechas.csv',
            'Consolidado Entrenamiento - Tabla Completa (1).csv',
            'Consolidado Entrenamiento  Tabla Completa 1.csv',
            'data.csv',
            'dataset.csv'
        ]
        
        df = None
        used_file = None
        
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
            return None
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip()
        
        # Mostrar información del archivo cargado
        st.info(f"📊 Archivo cargado: {used_file} ({len(df)} filas, {len(df.columns)} columnas)")
        
        # Identificar columnas que pueden tener valores numéricos con comas
        # Convertir TODAS las columnas excepto las claramente categóricas
        categorical_cols = ['Día', 'Folio', 'Lugar Muestreo', 'Comuna']
        
        for col in df.columns:
            if col not in categorical_cols:
                # Intentar convertir a numérico, reemplazando comas por puntos
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        
        # Procesar fechas si existe la columna Día
        if 'Día' in df.columns:
            try:
                # Intentar diferentes formatos de fecha
                df['Fecha'] = pd.to_datetime(df['Día'], errors='coerce', dayfirst=True)
                
                # Verificar si hay fechas válidas
                valid_dates = df['Fecha'].notna().sum()
                total_dates = len(df)
                
                if valid_dates > 0:
                    # Solo crear campos derivados si hay fechas válidas
                    df['Día_Semana'] = df['Fecha'].dt.day_name()
                    df['Mes'] = df['Fecha'].dt.month_name()
                    df['Día_Mes'] = df['Fecha'].dt.day
                    df['Semana'] = df['Fecha'].dt.isocalendar().week
                    st.success(f"✅ Fechas procesadas: {valid_dates}/{total_dates} válidas")
                    
                    # Mostrar ejemplos de fechas procesadas
                    sample_dates = df[df['Fecha'].notna()]['Fecha'].head(3)
                    st.info(f"📅 Ejemplos de fechas: {', '.join([d.strftime('%d/%m/%Y') for d in sample_dates])}")
                else:
                    st.warning("⚠️ No se pudieron procesar fechas válidas")
                    df['Fecha'] = None
                    
            except Exception as e:
                st.warning(f"⚠️ Error procesando fechas: {e}")
                df['Fecha'] = None
        
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

# Función para generar insights avanzados
def generate_advanced_insights(df, target_var):
    """Genera insights avanzados basados en rangos y correlaciones"""
    insights = []
    
    # Obtener variables numéricas
    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_vars = [col for col in numeric_vars if col != target_var and col in df.columns]
    
    if target_var not in df.columns:
        return insights
    
    for var in numeric_vars[:8]:  # Limitar a 8 variables para evitar sobrecarga
        try:
            # Calcular correlación
            correlation = df[var].corr(df[target_var])
            
            if abs(correlation) > 0.1:  # Solo mostrar correlaciones significativas
                # Dividir en rangos (cuartiles)
                quartiles = df[var].quantile([0.25, 0.5, 0.75]).values
                
                # Analizar cada rango
                for i, (q_low, q_high) in enumerate(zip([df[var].min()] + quartiles.tolist(), 
                                                       quartiles.tolist() + [df[var].max()])):
                    
                    mask = (df[var] >= q_low) & (df[var] <= q_high)
                    if mask.sum() > 5:  # Al menos 5 observaciones
                        target_values = df[mask][target_var]
                        
                        if target_var == 'Algas':
                            # Para algas (categórica), calcular probabilidad
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
                        
                        elif target_var == 'Fósforo reactivo total (mg/L)':
                            # Para fósforo (numérica), calcular promedio y rango
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

# Función para detectar factores de riesgo
def detect_risk_factors(df):
    """Detecta factores de riesgo automáticamente"""
    risk_factors = []
    
    # Definir umbrales de riesgo
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
                
                if risk_percentage > 5:  # Solo mostrar si afecta más del 5%
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

# Cargar datos
df = load_and_preprocess_data()

if df is not None:
    # Sidebar para navegación
    st.sidebar.title("🔧 Panel de Control")
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
            st.metric("Playas Monitoreadas", df['Lugar Muestreo'].nunique())
        with col3:
            st.metric("Comunas", df['Comuna'].nunique())
        with col4:
            if 'Fecha' in df.columns:
                date_range = (df['Fecha'].max() - df['Fecha'].min()).days
                st.metric("Días de Monitoreo", date_range)
        
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
                time_grouping = st.selectbox(
                    "Agrupación temporal:",
                    ["Sin agrupar", "Por día", "Por semana", "Por mes", "Por lugar", "Por comuna"]
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
            
            # Aplicar filtros si se seleccionaron
            filtered_df = df_temporal.copy()
            if filter_by != "Ninguno" and filter_by in filtered_df.columns:
                available_values = filtered_df[filter_by].dropna().unique()
                if len(available_values) > 0:
                    filter_values = st.multiselect(
                        f"Valores de {filter_by}:",
                        available_values,
                        default=available_values
                    )
                    filtered_df = filtered_df[filtered_df[filter_by].isin(filter_values)]
            
            if len(filtered_df) == 0:
                st.error("❌ No hay datos después del filtrado")
            else:
                # Generar gráfico temporal
                st.subheader(f"📊 {chart_type}: {variable_y}")
                
                try:
                    fig = None
                    
                    if chart_type == "Línea Temporal":
                        if time_grouping == "Por día":
                            daily_data = filtered_df.groupby(filtered_df['Fecha'].dt.date)[variable_y].mean().reset_index()
                            daily_data['Fecha'] = pd.to_datetime(daily_data['Fecha'])
                            fig = px.line(daily_data, x='Fecha', y=variable_y,
                                        title=f"Evolución Diaria Promedio de {variable_y}",
                                        template="plotly_white")
                        elif time_grouping == "Por semana":
                            weekly_data = filtered_df.groupby('Semana')[variable_y].mean().reset_index()
                            fig = px.line(weekly_data, x='Semana', y=variable_y,
                                        title=f"Evolución Semanal Promedio de {variable_y}",
                                        template="plotly_white")
                        elif time_grouping == "Por mes" and 'Mes' in filtered_df.columns:
                            monthly_data = filtered_df.groupby('Mes')[variable_y].mean().reset_index()
                            fig = px.line(monthly_data, x='Mes', y=variable_y,
                                        title=f"Evolución Mensual Promedio de {variable_y}",
                                        template="plotly_white")
                        else:
                            # Sin agrupar
                            fig = px.line(filtered_df.sort_values('Fecha'), x='Fecha', y=variable_y,
                                        color=color_by if color_by != "Ninguno" else None,
                                        title=f"Evolución Temporal de {variable_y}",
                                        template="plotly_white")
                        
                    elif chart_type == "Dispersión Temporal":
                        fig = px.scatter(filtered_df, x='Fecha', y=variable_y,
                                       color=color_by if color_by != "Ninguno" else None,
                                       size=size_by if size_by != "Ninguno" else None,
                                       title=f"Dispersión Temporal de {variable_y}",
                                       template="plotly_white")
                    
                    elif chart_type == "Box Plot Temporal":
                        if 'Mes' in filtered_df.columns and filtered_df['Mes'].notna().sum() > 0:
                            fig = px.box(filtered_df, x='Mes', y=variable_y,
                                       color=color_by if color_by != "Ninguno" else None,
                                       title=f"Distribución Mensual de {variable_y}",
                                       template="plotly_white")
                        elif 'Día_Semana' in filtered_df.columns:
                            fig = px.box(filtered_df, x='Día_Semana', y=variable_y,
                                       color=color_by if color_by != "Ninguno" else None,
                                       title=f"Distribución por Día de la Semana de {variable_y}",
                                       template="plotly_white")
                        else:
                            # Fallback: agrupar por semana
                            fig = px.box(filtered_df, x='Semana', y=variable_y,
                                       title=f"Distribución Semanal de {variable_y}",
                                       template="plotly_white")
                    
                    elif chart_type == "Histograma por Período":
                        if 'Mes' in filtered_df.columns:
                            fig = px.histogram(filtered_df, x=variable_y, 
                                             color='Mes',
                                             title=f"Distribución de {variable_y} por Mes",
                                             template="plotly_white")
                        else:
                            fig = px.histogram(filtered_df, x=variable_y,
                                             title=f"Distribución de {variable_y}",
                                             template="plotly_white")
                    
                    elif chart_type == "Tendencia con Regresión":
                        fig = px.scatter(filtered_df, x='Fecha', y=variable_y,
                                       color=color_by if color_by != "Ninguno" else None,
                                       trendline="ols",
                                       title=f"Tendencia de {variable_y} con Regresión",
                                       template="plotly_white")
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Mostrar estadísticas temporales
                        st.subheader("📊 Estadísticas Temporales")
                        col1, col2, col3 = st.columns(3)
                        
                        try:
                            with col1:
                                # Calcular tendencia
                                sorted_data = filtered_df.sort_values('Fecha')
                                if len(sorted_data) > 1:
                                    x_vals = np.arange(len(sorted_data))
                                    y_vals = sorted_data[variable_y].dropna()
                                    if len(y_vals) > 1:
                                        trend_slope = np.polyfit(x_vals[:len(y_vals)], y_vals, 1)[0]
                                        trend_direction = "📈 Creciente" if trend_slope > 0 else "📉 Decreciente" if trend_slope < 0 else "➡️ Estable"
                                        st.metric("Tendencia General", trend_direction)
                                    else:
                                        st.metric("Tendencia General", "No disponible")
                                else:
                                    st.metric("Tendencia General", "Datos insuficientes")
                            
                            with col2:
                                volatility = filtered_df[variable_y].std()
                                if not np.isnan(volatility):
                                    st.metric("Volatilidad (Desv. Est.)", f"{volatility:.4f}")
                                else:
                                    st.metric("Volatilidad", "No disponible")
                            
                            with col3:
                                if len(filtered_df) > 1:
                                    sorted_data = filtered_df.sort_values('Fecha')
                                    last_values = sorted_data[variable_y].dropna()
                                    if len(last_values) >= 2:
                                        last_value = last_values.iloc[-1]
                                        first_value = last_values.iloc[0]
                                        if first_value != 0:
                                            change = ((last_value - first_value) / first_value) * 100
                                            st.metric("Cambio Total", f"{change:.1f}%")
                                        else:
                                            st.metric("Cambio Total", "No calculable")
                                    else:
                                        st.metric("Cambio Total", "Datos insuficientes")
                                else:
                                    st.metric("Cambio Total", "Una sola muestra")
                        except Exception as e:
                            st.warning(f"Error calculando estadísticas: {e}")
                    else:
                        st.error("No se pudo generar el gráfico con los parámetros seleccionados")
                        
                except Exception as e:
                    st.error(f"Error generando gráfico: {e}")
                    st.write("Detalles del error para debugging:")
                    st.write(f"- Datos filtrados: {len(filtered_df)} filas")
                    st.write(f"- Variable Y: {variable_y}")
                    st.write(f"- Tipo de datos: {filtered_df[variable_y].dtype}")
                    st.write(f"- Valores nulos: {filtered_df[variable_y].isnull().sum()}")
                    
                    # Mostrar muestra de datos para debugging
                    if len(filtered_df) > 0:
                        st.write("Muestra de datos:")
                        st.dataframe(filtered_df[['Fecha', variable_y]].head())
    
    # SECCIÓN 3: INSIGHTS AVANZADOS
    elif seccion == "🔍 Insights Avanzados":
        st.header("🔍 Insights Avanzados")
        
        # Seleccionar variable objetivo para análisis
        target_options = ['Algas', 'Fósforo reactivo total (mg/L)']
        target_variable = st.selectbox("Selecciona variable objetivo para análisis:", target_options)
        
        if st.button("🔍 Generar Análisis Avanzado"):
            with st.spinner("Analizando patrones y generando insights..."):
                
                # Generar insights por rangos
                insights = generate_advanced_insights(df, target_variable)
                
                st.subheader(f"📊 Análisis de Rangos para {target_variable}")
                
                if insights:
                    # Organizar insights por tipo
                    if target_variable == 'Algas':
                        algas_insights = [i for i in insights if i['tipo'] == 'rango_algas']
                        
                        if algas_insights:
                            st.write("**🌱 Factores que Influyen en la Presencia de Algas:**")
                            
                            # Ordenar por probabilidad
                            algas_insights.sort(key=lambda x: x['probabilidad'], reverse=True)
                            
                            for insight in algas_insights[:10]:  # Top 10
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
                            
                            # Ordenar por concentración promedio
                            fosforo_insights.sort(key=lambda x: x['promedio'], reverse=True)
                            
                            for insight in fosforo_insights[:10]:  # Top 10
                                conc = insight['promedio']
                                if conc > 0.02:
                                    st.error(f"⚠️ {insight['descripcion']}")
                                elif conc > 0.01:
                                    st.warning(f"🔶 {insight['descripcion']}")
                                else:
                                    st.info(f"ℹ️ {insight['descripcion']}")
                
                # Detectar factores de riesgo
                st.subheader("⚠️ Factores de Riesgo Detectados")
                risk_factors = detect_risk_factors(df)
                
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
                
                # Análisis de correlaciones temporales
                if 'Fecha' in df.columns:
                    st.subheader("📈 Análisis de Tendencias Temporales")
                    
                    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    trends = {}
                    for var in numeric_vars[:8]:  # Limitar a 8 variables
                        try:
                            # Calcular tendencia temporal
                            df_temp = df.dropna(subset=[var, 'Fecha']).sort_values('Fecha')
                            if len(df_temp) > 5:
                                x_numeric = np.arange(len(df_temp))
                                slope, intercept = np.polyfit(x_numeric, df_temp[var], 1)
                                
                                # Clasificar tendencia
                                if abs(slope) < 0.01:
                                    trend_type = "Estable"
                                    emoji = "➡️"
                                elif slope > 0:
                                    trend_type = "Creciente"
                                    emoji = "📈"
                                else:
                                    trend_type = "Decreciente"
                                    emoji = "📉"
                                
                                trends[var] = {
                                    'slope': slope,
                                    'type': trend_type,
                                    'emoji': emoji
                                }
                        except:
                            continue
                    
                    if trends:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Tendencias Crecientes:**")
                            growing = {k: v for k, v in trends.items() if v['type'] == 'Creciente'}
                            for var, trend in sorted(growing.items(), key=lambda x: x[1]['slope'], reverse=True):
                                st.write(f"{trend['emoji']} {var}: +{trend['slope']:.4f}/día")
                        
                        with col2:
                            st.write("**Tendencias Decrecientes:**")
                            declining = {k: v for k, v in trends.items() if v['type'] == 'Decreciente'}
                            for var, trend in sorted(declining.items(), key=lambda x: x[1]['slope']):
                                st.write(f"{trend['emoji']} {var}: {trend['slope']:.4f}/día")
                
                # Recomendaciones automáticas
                st.subheader("💡 Recomendaciones Automáticas")
                
                recommendations = []
                
                # Basadas en factores de riesgo
                high_risk_factors = [r for r in risk_factors if r['percentage'] > 20]
                if high_risk_factors:
                    recommendations.append(f"🔍 Monitorear de cerca: {', '.join([r['factor'] for r in high_risk_factors[:3]])}")
                
                # Basadas en correlaciones
                if target_variable == 'Algas' and insights:
                    high_prob_factors = [i for i in insights if i['tipo'] == 'rango_algas' and i['probabilidad'] > 40]
                    if high_prob_factors:
                        recommendations.append(f"🌱 Controlar variables críticas para algas: {', '.join([i['variable'] for i in high_prob_factors[:3]])}")
                
                # Basadas en tendencias temporales
                if trends:
                    critical_trends = [var for var, trend in trends.items() if 
                                     var in ['Fósforo reactivo total (mg/L)', 'Algas', 'pH'] and 
                                     trend['type'] in ['Creciente', 'Decreciente']]
                    if critical_trends:
                        recommendations.append(f"📊 Investigar tendencias en: {', '.join(critical_trends[:3])}")
                
                if recommendations:
                    for rec in recommendations:
                        st.info(rec)
                else:
                    st.success("✅ Las condiciones del agua se mantienen dentro de parámetros normales")
    
    # SECCIÓN 4: MODELOS PREDICTIVOS (Mantenido igual)
    elif seccion == "🤖 Modelos Predictivos":
        st.header("🤖 Modelos Predictivos")
        
        # Pestañas para los dos modelos
        tab1, tab2 = st.tabs(["🧪 Predictor de Fósforo", "🌱 Predictor de Algas"])
        
        # Preparar datos para modelos
        @st.cache_data
        def prepare_model_data():
            # Excluir Folio, Lugar Muestreo y variables temporales
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
            
            return model_df, le_comuna
        
        model_df, le_comuna = prepare_model_data()
        
        if model_df is None:
            st.error("❌ No se pudieron preparar los datos para los modelos.")
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
    st.info("📁 Archivos CSV esperados en el directorio raíz:")
    st.write("   - Consolidado Entrenamiento  Tabla Fechas.csv")
    st.write("   - Consolidado Entrenamiento - Tabla Completa (1).csv")
    st.write("   - data.csv")
    st.write("   - dataset.csv")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>🌊 Sistema de Monitero  Calidad del Agua</strong></p>
    <p>Región de la Araucanía - Lagos Pucón y Villarrica</p>
    <p><em>Desarrollado por BIOREN UFRO con el apoyo de Ciencia 2030/em></p>
</div>
""", unsafe_allow_html=True)