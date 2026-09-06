# Actualización v2 — módulo 3 interfaz

Esta actualización contiene SOLO los archivos modificados respecto a la versión inicial.

## Cambios incluidos

1. **Form de registro** (templates/registro.html)
   - Todos los campos obligatorios excepto Sub-especialidad
   - Al seleccionar "Otra" en Especialidad aparece un input de texto
   - Validación server-side robusta de campos vacíos

2. **Página de selección** (templates/seleccion.html)
   - Sin información del paciente (edad, sexo, etiquetas, escenarios)
   - Solo muestra "Señal 1", "Señal 2", ..., "Señal 5" con info técnica

3. **Página de evaluación** (templates/evaluar.html + static/js/evaluar.js + static/css/medico.css)
   - Sin información del paciente
   - ECG en 6 tiras de 10 segundos cada una (formato clínico estándar 25mm/s, 10mm/mV)
   - Cuadrícula milimetrada en cada tira para que sea fácil de leer
   - Pistas LATIDO y RITMO con semáforos siguen al final

4. **App backend** (app.py)
   - Validación robusta sin crashes
   - Manejo correcto de "Otra" especialidad
   - Calcula signal_number (1..5) para mostrar en la página

5. **Clasificador** (classifier/prepare_signals.py)
   - Lee label_encoder_*_MSM.npy desde data/models/ (no shapelets/) — donde tú los pusiste

## Cómo aplicar

Copia cada archivo sobre el equivalente en tu D:\Modelo_Tesis\Modulo_I\modulo_3_interfaz\

  templates/registro.html   → reemplaza el actual
  templates/seleccion.html  → reemplaza el actual
  templates/evaluar.html    → reemplaza el actual
  static/js/evaluar.js      → reemplaza el actual
  static/css/medico.css     → reemplaza el actual
  app.py                    → reemplaza el actual
  classifier/prepare_signals.py → reemplaza el actual

NO toques los otros archivos (config.py, requirements.txt, llm_backends/, etc.)
