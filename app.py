from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from supabase import create_client, Client
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'clave_secreta_12345'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite de 16MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================
# CONFIGURACIÓN DE SUPABASE
# ============================================
try:
    # Intentar obtener de variables de entorno (Configuración segura en Render/Koyeb)
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
    
    # Si no están en el entorno, usar las de respaldo (solo para desarrollo local)
    if not SUPABASE_URL: SUPABASE_URL = "https://twhkixjwrmpiikgejrut.supabase.co"
    if not SUPABASE_KEY: SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    # Probar la conexión solicitando un dato simple
    supabase.table("practicantes").select("id", count="exact").limit(1).execute()
    print("✅ Conexión verificada y exitosa a Supabase")
except Exception as e:
    print(f"❌ ERROR CRÍTICO: No se pudo conectar a Supabase. Revisa tus credenciales. Detalle: {e}")
    # En producción, podrías querer lanzar una excepción o manejar el error de otra forma
    supabase = None

# ============================================
# CONFIGURACIÓN
# ============================================
DATOS_EXCEL = 'datos_practicantes_2500.xlsx'
ALLOWED_EXTENSIONS = {'doc', 'docx', 'pdf', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Horario laboral
HORARIO_INICIO = 10  # 10:00 AM
HORARIO_FIN = 17     # 5:00 PM

# Columnas a ELIMINAR
COLUMNAS_ELIMINAR = [
    'id_tarea', 'fecha', 'dia_semana', 'hora_inicio', 'hora_fin',
    'clima_laboral', 'horas_sueno_previas', 'carga_mental', 
    'motivacion', 'calidad_entregable', 'categoria_tiempo'
]

# Columnas para predicción
COLUMNAS_USAR = [
    'practicante', 'nivel_practicante', 'experiencia_meses', 
    'certificaciones', 'puntuacion_practicante', 'tipo_tarea', 
    'tecnologia', 'dificultad', 'urgencia', 'tareas_paralelas', 
    'conocimiento_previo'
]

# ============================================
# FUNCIÓN PARA CALCULAR HORAS TRABAJADAS
# ============================================
def calcular_horas_trabajadas(fecha_inicio, fecha_fin):
    """
    Calcula las horas trabajadas entre dos fechas,
    respetando el horario laboral (10am - 5pm)
    Solo días laborales (Lunes a Viernes)
    """
    tz = pytz.timezone('America/Lima')
    
    # Supabase guarda en UTC. Si la fecha es naive, primero la marcamos como UTC
    # y luego la convertimos a Lima. Si ya tiene zona horaria, astimezone hace la conversión.
    if fecha_inicio.tzinfo is None:
        fecha_inicio = pytz.UTC.localize(fecha_inicio)
    if fecha_fin.tzinfo is None:
        fecha_fin = pytz.UTC.localize(fecha_fin)

    fecha_inicio = fecha_inicio.astimezone(tz)
    fecha_fin = fecha_fin.astimezone(tz)

    if fecha_inicio >= fecha_fin:
        return 0
    
    horas_totales = 0
    dia_actual = fecha_inicio.date()
    fecha_fin_dt = fecha_fin.date()
    
    while dia_actual <= fecha_fin_dt:
        # Verificar si es día laboral (Lunes=0, Domingo=6)
        es_laboral = dia_actual.weekday() < 5  # 0=Lunes, 4=Viernes
        
        if es_laboral:
            # Definir límites del horario laboral para el día actual localizados
            limite_inicio = tz.localize(datetime(dia_actual.year, dia_actual.month, dia_actual.day, HORARIO_INICIO, 0))
            limite_fin = tz.localize(datetime(dia_actual.year, dia_actual.month, dia_actual.day, HORARIO_FIN, 0))

            # Hora de inicio para este día
            if dia_actual == fecha_inicio.date():
                inicio_dia = max(fecha_inicio, limite_inicio)
            else:
                inicio_dia = limite_inicio
            
            # Hora de fin para este día
            if dia_actual == fecha_fin_dt:
                fin_dia = min(fecha_fin, limite_fin)
            else:
                fin_dia = limite_fin
            
            if fin_dia > inicio_dia:
                horas_dia = (fin_dia - inicio_dia).total_seconds() / 3600
                horas_totales += horas_dia
        
        dia_actual += timedelta(days=1)
    # Retornamos el valor exacto para que format_hms haga el resto
    return horas_totales

def format_hms(horas_float):
    """Convierte horas en formato decimal a una cadena legible 'Xh Ym Zs'"""
    if horas_float is None or horas_float == "":
        return "—"
    try:
        # Usamos round para evitar errores de precisión de punto flotante (ej: 0.0499999)
        total_segundos = int(round(float(horas_float) * 3600))
        
        # Fallback para tareas completadas en menos de un segundo durante pruebas
        if float(horas_float) > 0 and total_segundos == 0:
            total_segundos = 1
            
        horas = total_segundos // 3600
        minutos = (total_segundos % 3600) // 60
        segundos = total_segundos % 60
        return f"{horas}h {minutos}m {segundos}s"
    except:
        return "—"

# ============================================
# MODELO Y PREDICCIONES
# ============================================
modelo_global = None
label_encoders = {}

def cargar_y_entrenar_modelo():
    global modelo_global, label_encoders
    
    if not os.path.exists(DATOS_EXCEL):
        print(f"⚠️ No se encuentra {DATOS_EXCEL}")
        return False
    
    df = pd.read_excel(DATOS_EXCEL)
    columnas_existentes = [col for col in COLUMNAS_ELIMINAR if col in df.columns]
    df = df.drop(columns=columnas_existentes)
    
    if 'tiempo_real_hs' in df.columns:
        df['tiempo_real_hs'] = df['tiempo_real_hs'].round(0).astype(int)
    
    df = df.dropna()
    
    categorical_cols = ['practicante', 'nivel_practicante', 'tipo_tarea', 'tecnologia']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    X = df[[col for col in COLUMNAS_USAR if col in df.columns]]
    y = df['tiempo_real_hs']
    
    modelo_global = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
    modelo_global.fit(X, y)
    
    print(f"✅ Modelo entrenado con {len(df)} registros")
    return True

def predecir_tiempo(tarea_dict):
    """Realiza la predicción de tiempo para un practicante y tarea específicos"""
    if modelo_global is None:
        return round(tarea_dict.get('dificultad', 3) * 2, 0)
    
    df_pred = pd.DataFrame([tarea_dict])
    
    # Columnas que necesitan encoding
    for col in ['practicante', 'nivel_practicante', 'tipo_tarea', 'tecnologia']:
        if col in df_pred.columns and col in label_encoders:
            # Si el valor no fue visto en el entrenamiento, usamos el primer valor del encoder 
            # o manejamos el error para no romper la ejecución
            try:
                df_pred[col] = label_encoders[col].transform(df_pred[col].astype(str))
            except:
                df_pred[col] = 0
    
    for col in modelo_global.feature_names_in_:
        if col not in df_pred.columns:
            df_pred[col] = 0
    
    df_pred = df_pred[modelo_global.feature_names_in_]
    return int(round(modelo_global.predict(df_pred)[0], 0))

def predecir_todos_practicantes(tarea_base, solo_disponibles=False):
    """Predice tiempos para todos los practicantes basándose en sus datos históricos"""
    if not os.path.exists(DATOS_EXCEL): return []
    
    df_excel = pd.read_excel(DATOS_EXCEL)
    # Obtenemos el perfil más reciente/único de cada practicante
    practicantes_info = df_excel.drop_duplicates(subset=['practicante'], keep='last')
    
    resultados = []
    for _, datos_p in practicantes_info.iterrows():
        p = datos_p['practicante']
        
        tarea_con_practicante = {
            'practicante': p,
            'nivel_practicante': datos_p['nivel_practicante'],
            'experiencia_meses': datos_p['experiencia_meses'],
            'certificaciones': datos_p['certificaciones'],
            'puntuacion_practicante': datos_p['puntuacion_practicante'],
            'tipo_tarea': tarea_base.get('tipo_tarea', 'web_page'),
            'tecnologia': tarea_base.get('tecnologia', 'python'),
            'dificultad': tarea_base.get('dificultad', 3),
            'urgencia': tarea_base.get('urgencia', 2),
            'tareas_paralelas': tarea_base.get('tareas_paralelas', 1),
            'conocimiento_previo': tarea_base.get('conocimiento_previo', 3)
        }
        
        tiempo = predecir_tiempo(tarea_con_practicante)
        resultados.append({
            'practicante': p,
            'tiempo_predicho': tiempo
        })
    
    resultados.sort(key=lambda x: x['tiempo_predicho'])
    return resultados

def obtener_historial_practicante(nombre_practicante):
    if not os.path.exists(DATOS_EXCEL):
        return []
    
    df = pd.read_excel(DATOS_EXCEL)
    df_practicante = df[df['practicante'] == nombre_practicante]
    
    if len(df_practicante) == 0:
        return []
    
    columnas_mostrar = ['tipo_tarea', 'tecnologia', 'dificultad', 'tiempo_real_hs']
    columnas_existentes = [col for col in columnas_mostrar if col in df_practicante.columns]
    
    historial = df_practicante[columnas_existentes].copy()
    
    if 'tiempo_real_hs' in historial.columns:
        historial['tiempo_real_hs'] = historial['tiempo_real_hs'].round(0).astype(int)
    
    historial = historial.rename(columns={
        'tipo_tarea': 'Tipo Tarea',
        'tecnologia': 'Tecnología',
        'dificultad': 'Dificultad',
        'tiempo_real_hs': 'Tiempo (horas)'
    })
    
    return historial.to_dict('records')

# ============================================
# RUTAS
# ============================================
@app.route('/')
def index():
    # Si ya hay una sesión activa, redirigir al panel correspondiente
    if session.get('rol') == 'admin':
        return redirect(url_for('admin'))
    elif session.get('rol') == 'practicante' and session.get('practicante_id'):
        return redirect(url_for('practicante'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Limpia la sesión y regresa al login"""
    session.clear()
    return redirect(url_for('index'))

@app.route('/login', methods=['POST'])
def login():
    rol = request.form.get('rol')
    
    if rol == 'admin':
        password = request.form.get('password')
        # Añadimos una validación simple de contraseña para el Admin
        if password == 'admin123': # Contraseña maestra de ejemplo
            session['rol'] = 'admin'
            session['practicante_id'] = None
            return redirect(url_for('admin'))
        else:
            return render_template('login_admin.html', error="Contraseña administrativa incorrecta")
    return redirect(url_for('index'))

@app.route('/login_admin')
def login_admin_page():
    return render_template('login_admin.html')

@app.route('/login_practicante', methods=['GET', 'POST'])
def login_practicante_page():
    # Obtener practicantes de Supabase para el directorio
    resp = supabase.table("practicantes").select("*").execute()
    practicantes = resp.data

    if request.method == 'POST':
        email = request.form.get('email')
        match = [p for p in practicantes if p['email'] == email]
        if match:
            session['rol'] = 'practicante'
            session['practicante_id'] = int(match[0]['id'])
            return redirect(url_for('practicante'))
        return render_template('login_practicante.html', error="Email no encontrado", practicantes=practicantes)
    
    return render_template('login_practicante.html', practicantes=practicantes)

# ============================================
# ADMINISTRADOR
# ============================================
@app.route('/admin')
def admin():
    if session.get('rol') != 'admin':
        return redirect(url_for('index'))
    
    # Obtener datos desde Supabase
    resp_tareas = supabase.table("tareas").select("*").order("id", desc=True).execute()
    tareas_info = resp_tareas.data

    resp_practicantes = supabase.table("practicantes").select("*").execute()
    practicantes_info = resp_practicantes.data

    for t in tareas_info:
        resp = t.get('responsable')
        t['asignado_a'] = resp if resp else "—"
        # Formatear el tiempo real para la vista
        t['tiempo_fmt'] = format_hms(t.get('tiempo_real'))

    return render_template('admin.html', tareas=tareas_info, practicantes=practicantes_info)

@app.route('/subir_tarea', methods=['POST'])
def subir_tarea():
    nuevo_id = int(datetime.now().timestamp())
    
    # Manejo de archivo
    archivo_nombre = None
    if 'archivo' in request.files:
        file = request.files['archivo']
        if file and file.filename != '' and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            archivo_nombre = f"{nuevo_id}_{filename}"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], archivo_nombre))

    tarea_base = {
        'id': nuevo_id,
        'nombre': request.form.get('nombre'),
        'estado': 'pendiente',
        'tipo_tarea': request.form.get('tipo_tarea'),
        'tecnologia': request.form.get('tecnologia'),
        'dificultad': int(request.form.get('dificultad')),
        'urgencia': int(request.form.get('urgencia')),
        'tareas_paralelas': int(request.form.get('tareas_paralelas')),
        'conocimiento_previo': int(request.form.get('conocimiento_previo')),
        'tiempo_real': None,
        'fecha_inicio': None,
        'fecha_aceptacion': None,
        'archivo': archivo_nombre,
        'responsable': ''
    }
    
    # GUARDAR EN SUPABASE (Ya no usamos CSV)
    supabase.table("tareas").insert(tarea_base).execute()
    
    return redirect(url_for('admin'))

def intentar_asignacion_automatica(tarea_id):
    """Lógica centralizada para asignar una tarea al mejor practicante libre"""
    # 1. Obtener la tarea de Supabase
    tarea_resp = supabase.table("tareas").select("*").eq("id", tarea_id).single().execute()
    tarea = tarea_resp.data
    if not tarea: return False

    # 2. Obtener practicantes libres en Supabase (tarea_asignada is null)
    pract_resp = supabase.table("practicantes").select("*").is_("tarea_asignada", "null").execute()
    disponibles = pract_resp.data
    
    if disponibles:
        tiempos = predecir_todos_practicantes(tarea)
        nombres_disponibles = [p['nombre'] for p in disponibles]
        tiempos_disponibles = [t for t in tiempos if t['practicante'] in nombres_disponibles]
        
        if tiempos_disponibles:
            mejor = tiempos_disponibles[0]
            
            # 3. Actualizar Practicante en Supabase
            supabase.table("practicantes").update({
                "tarea_asignada": tarea_id,
                "tiempo_predicho": mejor['tiempo_predicho']
            }).eq("nombre", mejor['practicante']).execute()
            
            # 4. Actualizar Tarea en Supabase
            supabase.table("tareas").update({
                "estado": "asignada",
                "responsable": mejor['practicante']
            }).eq("id", tarea_id).execute()
            
            return True
    return False

@app.route('/predecir_y_mostrar_top10', methods=['POST'])
def predecir_y_mostrar_top10():
    tarea_id = request.form.get('tarea_id')
    recalcular_top10_sesion(tarea_id)
    return redirect(url_for('admin'))

def recalcular_top10_sesion(target_id=None):
    """Muestra el Top 10 para una tarea específica usando Supabase"""
    resp = supabase.table("tareas").select("*").eq("id", target_id).single().execute()
    tarea = resp.data
    if tarea:
        top10 = predecir_todos_practicantes(tarea)
        session['resultados_top10'] = [{
            'tarea_id': tarea['id'],
            'tarea_nombre': tarea['nombre'],
            'top10': top10[:10]
        }]
        session.modified = True

@app.route('/asignar_tarea_automatica', methods=['POST'])
def asignar_tarea_automatica():
    """Asigna manualmente desde el top 10 y limpia la sesión"""
    tarea_id = int(request.form.get('tarea_id'))

    if intentar_asignacion_automatica(tarea_id):
        # Eliminar esta tarea de los resultados del top10 en sesión
        if 'resultados_top10' in session:
            session['resultados_top10'] = [r for r in session['resultados_top10'] if r['tarea_id'] != tarea_id]
            session.modified = True

    return redirect(url_for('admin'))


@app.route('/eliminar_tarea/<int:tarea_id>')
def eliminar_tarea(tarea_id):
    supabase.table("tareas").delete().eq("id", tarea_id).execute()
    return redirect(url_for('admin'))

@app.route('/reset_sistema')
def reset_sistema():
    # Limpiar asignaciones en Supabase
    supabase.table("practicantes").update({
        "tarea_asignada": None, "tiempo_predicho": None
    }).neq("id", 0).execute()
    # Borrar tareas que no estén completadas
    supabase.table("tareas").delete().neq("estado", "completada").execute()
    return redirect(url_for('admin'))

# ============================================
# PRACTICANTE - NUEVAS RUTAS
# ============================================
@app.route('/practicante')
def practicante():
    if session.get('rol') != 'practicante':
        return redirect(url_for('index'))
    
    practicante_id = session.get('practicante_id')
    if not practicante_id:
        return redirect(url_for('index'))
    
    resp_p = supabase.table("practicantes").select("*").eq("id", practicante_id).single().execute()
    practicante = resp_p.data
    if not practicante: return redirect(url_for('index'))
    
    tarea_asignada = None
    tarea_aceptada = False
    fecha_inicio = None
    
    if practicante.get('tarea_asignada'):
        resp_t = supabase.table("tareas").select("*").eq("id", practicante['tarea_asignada']).single().execute()
        if resp_t.data:
            tarea_asignada = resp_t.data
            tarea_asignada['tiempo_predicho'] = practicante.get('tiempo_predicho')
            tarea_aceptada = tarea_asignada.get('fecha_inicio') is not None
            fecha_inicio = tarea_asignada.get('fecha_inicio')
    
    estadisticas = {
        'tareas_completadas': int(practicante.get('tareas_completadas', 0)),
        'tiempo_promedio': round(practicante.get('tiempo_total', 0) / max(practicante.get('tareas_completadas', 1), 1), 1)
    }
    
    # 1. Obtener historial reciente de Supabase (tareas completadas en esta sesión)
    resp_h = supabase.table("tareas").select("*").eq("responsable", practicante['nombre']).eq("estado", "completada").order("id", desc=True).execute()
    tareas_recientes = resp_h.data
    
    for t_rec in tareas_recientes:
        t_rec['tiempo_fmt'] = format_hms(t_rec.get('tiempo_real'))

    # 2. Obtener historial antiguo de Excel (archivo histórico)
    historial_antiguo = obtener_historial_practicante(practicante['nombre'])
    
    return render_template('practicante.html', 
                          practicante=practicante, 
                          tarea=tarea_asignada,
                          estadisticas=estadisticas,
                          tareas_recientes=tareas_recientes,
                          historial_antiguo=historial_antiguo,
                          tarea_aceptada=tarea_aceptada,
                          fecha_inicio=fecha_inicio)

@app.route('/aceptar_tarea', methods=['POST'])
def aceptar_tarea():
    """El practicante acepta la tarea y registra la hora de inicio"""
    practicante_id = session.get('practicante_id')
    resp_p = supabase.table("practicantes").select("tarea_asignada").eq("id", practicante_id).single().execute()
    tarea_id = resp_p.data.get('tarea_asignada')
    
    if tarea_id:
        ahora = datetime.now(pytz.timezone('America/Lima'))
        # Se guarda la fecha y hora exacta de aceptación para iniciar el conteo
        supabase.table("tareas").update({
            "fecha_aceptacion": ahora.isoformat(),
            "fecha_inicio": ahora.isoformat()
        }).eq("id", tarea_id).execute()
    
    return redirect(url_for('practicante'))

@app.route('/completar_tarea_practicante', methods=['POST'])
def completar_tarea_practicante():
    practicante_id = session.get('practicante_id')
    resp_p = supabase.table("practicantes").select("*").eq("id", practicante_id).single().execute()
    practicante = resp_p.data
    tarea_id = practicante['tarea_asignada']
    
    if tarea_id:
        # Obtenemos los datos de la tarea para saber cuándo se inició
        resp_t = supabase.table("tareas").select("*").eq("id", tarea_id).single().execute()
        tarea = resp_t.data
        fecha_inicio_str = tarea.get('fecha_inicio')
        ahora = datetime.now(pytz.timezone('America/Lima'))
        
        if fecha_inicio_str:
            fecha_inicio = datetime.fromisoformat(fecha_inicio_str)
            # Calculamos cuánto tiempo demoró basándonos en el horario laboral (10am - 5pm)
            tiempo_real = calcular_horas_trabajadas(fecha_inicio, ahora)
        else:
            tiempo_real = 0
        
        # 1. Actualizar Tarea: Cambiar estado a completada y guardar tiempo demorado
        supabase.table("tareas").update({
            "estado": "completada",
            "tiempo_real": tiempo_real,
            "fecha_fin": ahora.isoformat()
        }).eq("id", tarea_id).execute()
        
        # 2. Actualizar Practicante: Liberar tarea y acumular estadísticas de tiempo y cantidad
        supabase.table("practicantes").update({
            "tarea_asignada": None,
            "tareas_completadas": int(practicante.get('tareas_completadas', 0)) + 1,
            "tiempo_total": float(practicante.get('tiempo_total', 0)) + tiempo_real
        }).eq("id", practicante_id).execute()
    
    return redirect(url_for('practicante'))

@app.route('/actualizar_tarea_practicante', methods=['POST'])
def actualizar_tarea_practicante():
    """Busca la mejor tarea disponible en Supabase para el practicante"""
    practicante_id = session.get('practicante_id')

    # 1. Obtener datos del practicante desde Supabase
    resp_p = supabase.table("practicantes").select("*").eq("id", practicante_id).single().execute()
    practicante = resp_p.data
    
    if not practicante or practicante.get('tarea_asignada'):
        return redirect(url_for('practicante'))

    # 2. Buscar tareas pendientes en Supabase
    resp_t = supabase.table("tareas").select("*").eq("estado", "pendiente").execute()
    tareas_pendientes = resp_t.data
    
    if tareas_pendientes:
        practicante_nombre = practicante['nombre']
        mejor_tarea = None
        mejor_tiempo = float('inf')
        
        # Cargar perfil del practicante desde el Excel de entrenamiento para el modelo
        df_excel = pd.read_excel(DATOS_EXCEL) if os.path.exists(DATOS_EXCEL) else None
        datos_p = df_excel[df_excel['practicante'] == practicante_nombre].iloc[-1] if df_excel is not None and not df_excel[df_excel['practicante'] == practicante_nombre].empty else None
        
        for tarea in tareas_pendientes:
            if datos_p is not None:
                tarea_con_practicante = {
                    'practicante': practicante_nombre,
                    'nivel_practicante': datos_p['nivel_practicante'],
                    'experiencia_meses': datos_p['experiencia_meses'],
                    'certificaciones': datos_p['certificaciones'],
                    'puntuacion_practicante': datos_p.get('puntuacion_practicante', 80),
                    'tipo_tarea': tarea['tipo_tarea'], 'tecnologia': tarea['tecnologia'],
                    'dificultad': tarea['dificultad'], 'urgencia': tarea['urgencia'],
                    'tareas_paralelas': tarea['tareas_paralelas'], 'conocimiento_previo': tarea['conocimiento_previo']
                }
                
                tiempo = predecir_tiempo(tarea_con_practicante)
                if tiempo < mejor_tiempo:
                    mejor_tiempo = tiempo
                    mejor_tarea = tarea
        
        if mejor_tarea is not None:
            # 3. Actualizar en Supabase
            supabase.table("practicantes").update({
                "tarea_asignada": mejor_tarea['id'],
                "tiempo_predicho": mejor_tiempo
            }).eq("id", practicante_id).execute()
            
            supabase.table("tareas").update({
                "estado": "asignada",
                "responsable": practicante_nombre
            }).eq("id", mejor_tarea['id']).execute()
    
    return redirect(url_for('practicante'))

@app.route('/descargar/<filename>')
def descargar_archivo(filename):
    """Permite descargar archivos de la carpeta de uploads"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ============================================
# INICIAR
# ============================================
if __name__ == '__main__':
    cargar_y_entrenar_modelo()
    # El puerto 5000 es el predeterminado local, pero en la nube se usa la variable de entorno PORT
    port = int(os.environ.get("PORT", 5000)) 
    print("\n" + "="*50)
    print("🚀 SISTEMA DE TAREAS INICIADO")
    print("="*50)
    print(f"📌 Servidor corriendo en el puerto: {port}")
    print("📅 Horario laboral: 10:00 AM - 5:00 PM (Lunes a Viernes)")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=port)