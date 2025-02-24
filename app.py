from flask import Flask, request, jsonify, render_template
import ollama
from waitress import serve
import logging
import os
from flask_cors import CORS
from docx import Document  # Importar la librería para leer el archivo .docx

# Configuración inicial
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder='templates')
CORS(app)
MODEL_NAME = "mi-historia"  # Nombre del modelo fine-tuned
CONTEXT_LENGTH = 1500

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Función para leer el contenido de documento.docx
def leer_documento():
    """Lee el contenido del archivo documento.docx"""
    doc_path = os.path.join(BASE_DIR, 'modelo', 'documento.docx')
    doc = Document(doc_path)
    texto = []
    for para in doc.paragraphs:
        texto.append(para.text)
    return '\n'.join(texto)  # Unir el texto de todas las partes del documento

# Clase de manejo del modelo
class StoryModel:
    def __init__(self):
        self._verify_model()
        self.documento = leer_documento()  # Leer el contenido del documento
    
    def _verify_model(self):
        """Valida que el modelo esté disponible"""
        try:
            models = ollama.list()["models"]
            if not any(m["name"] == f"{MODEL_NAME}:latest" for m in models):
                raise RuntimeError(f"Modelo {MODEL_NAME} no encontrado")
        except Exception as e:
            logger.error(f"Error verificando modelo: {str(e)}")
            raise

    def generate_response(self, question):
        """Genera respuesta usando el modelo fine-tuned"""
        try:
            # Utilizar el contenido completo del cuento como contexto
            context = self.documento
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": f"Contexto: {context[:CONTEXT_LENGTH]}"},
                    {"role": "user", "content": question}
                ],
                options={"temperature": 0.5, "num_ctx": 2048}
            )
            return response.get('message', {}).get('content', 'No se pudo generar respuesta')
        except Exception as e:
            logger.error(f"Error generando respuesta: {str(e)}")
            return "Error al generar la respuesta."

# Inicializar el modelo
try:
    story_model = StoryModel()
    logger.info("✅ Modelo cargado correctamente")
except Exception as e:
    logger.error(f"🔥 Error crítico inicializando modelo: {str(e)}")
    raise

# Ruta para servir el frontend
@app.route('/')
def index():
    return render_template('index.html')

# Endpoints de la API
@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.json
    if not data or 'question' not in data:
        return jsonify({"error": "Faltan campos requeridos"}), 400
    
    try:
        # Obtener respuesta del modelo fine-tuned
        response = story_model.generate_response(
            question=data['question']
        )
        return jsonify({
            "question": data['question'],
            "response": response,
            "model": MODEL_NAME
        }), 200
    except Exception as e:
        logger.error(f"Error en /api/ask: {str(e)}")
        return jsonify({"error": "Error procesando la solicitud"}), 500

# Health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model": MODEL_NAME, "ready": True})

# Iniciar servidor
def main():
    os.chdir(BASE_DIR)
    serve(app, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"🚀 Servidor corriendo en:")
    print(f"🔹 Local:      http://127.0.0.1:5000/")
    print(f"🔹 Red local:  http://{local_ip}:5000/")

    main()

