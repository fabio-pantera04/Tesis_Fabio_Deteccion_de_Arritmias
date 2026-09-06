import os
from google import genai

# Nueva forma de inicializar: se pasa la API key al construir el cliente
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Nueva forma de generar contenido
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Traduce al ingles en una sola linea: 'ritmo sinusal, ECG normal'"
)

print("Respuesta de Gemini:", response.text)