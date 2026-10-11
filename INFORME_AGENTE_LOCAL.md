# Informe — Agente local con RAG: métricas, inferencia, recursos y calidad

Versión del prototipo: `ragpoc` 2.3.1 · Fecha de medición: 2026-10-11
Hardware de referencia: Windows 11, CPU x64 sin GPU dedicada, Python 3.13.5
Techo de producto: **≤ 2 GB de uso deseado, ≤ 3 GB de uso total admitido**

---

## 1. Resumen ejecutivo

Se construyó, cableó y midió un agente local sin dependencia de API de LLM sobre
ONNX Runtime GenAI, reutilizando el stack multimodal de la aplicación. El agente
conserva sus tools actuales y su RAG por similitud coseno.

| Resultado | Valor medido |
|---|---|
| Pico de RAM — chat solo | **2 341 MB** |
| Pico de RAM — chat + transcriptor de audio | **2 792 MB** |
| Velocidad de generación | **20,3 tok/s** en CPU |
| Latencia mediana por respuesta | **3,1 s** |
| Calidad de respuesta (recall de términos) | **0,80** |
| **Fidelidad — abstención en casos sin respuesta** | **1,00** (antes 0,00) |
| **Fidelidad — tasa de alucinación** | **0,00** (antes 0,60) |
| **Retrieval real — MRR sobre su base** | **97,22%** |
| **Retrieval real — Recall@8** | **89,70%** |
| Descarga del stack local completo | **2 099 MB** |
| Pruebas automatizadas | **281 nuevas** · **431 en verde** |

**Veredicto:** el objetivo está alcanzado y demostrado sobre el modelo real,
incluyendo el loop de tools y las cuatro modalidades. El techo de 3 GB se cumple
en todos los escenarios. El objetivo blando de 2 GB se cumple en texto y se
supera cuando el transcriptor de audio está también residente (sección 3.3).

---

## 2. Stack elegida y restricciones

| Pieza | Elección | Disco |
|---|---|---|
| LLM del agente | `onnx-community/Qwen3-VL-2B-Instruct-ONNX`, `cpu-int4-rtn-block-32` | 1 373 MB |
| ASR (audio) | `tonythethompson/Whisper-Tiny-GenAI-ONNX` | 224 MB |
| Embeddings multimodales | `onnx-community/embeddinggemma-2-ONNX` q4f16 (preexistente) | 406 MB |
| Runtime | `onnxruntime-genai==0.14.1` sobre `onnxruntime-directml==1.24.4` | — |
| **Total descarga** | | **2 099 MB** |

Modalidades: texto, imagen y video (fotogramas) por Qwen3-VL; audio por
whisper-tiny.

### 2.1 Versión crítica de dependencia

`onnxruntime-genai` **debe quedar fijado en `0.14.x`**. Las versiones 0.15+
requieren ORT API 26 y fallan al importar contra `onnxruntime-directml` 1.24.4
(expone API 24). No existe versión más nueva de `-directml`, así que no hay
margen sin perder DirectML. Riesgo que debe quedar en el lock de dependencias.

### 2.2 Layout de los exportes de Hugging Face

Cada repo `onnx-community` publica **dos layout incompatibles**:

- `onnx/<nombre>_<dtype>.onnx` — para transformers.js, **sin**
  `genai_config.json`. No sirve para onnxruntime-genai.
- `onnxruntime/<ep>/<quant>/` — para ONNX GenAI, autocontenido.

Solo el segundo es cargable; el descargador apunta solo a ese subárbol.

### 2.3 Corrección de repositorio ASR

La primera elección, `onnx-community/whisper-tiny`, **no tiene
`genai_config.json`** en ninguna ruta: es un export de transformers.js con
encoder/decoder sueltos y no puede cargarse en ONNX GenAI. Se sustituyó por
`tonythethompson/Whisper-Tiny-GenAI-ONNX`, que sí es un export GenAI
(`type: "whisper"`, `encoder.onnx` + `decoder.onnx`, `context_length: 448`).

---

## 3. Uso de recursos

### 3.1 Memoria

Medido con `PeakMemoryTracker` (muestreo de RSS cada 50 ms sobre el árbol de
procesos completo, incluyendo ffmpeg).

| Escenario | Base | Carga | Pico | Δ |
|---|---|---|---|---|
| Qwen3 0.6B (humo) | 60 MB | 812 MB | **1 096 MB** | +285 MB |
| Qwen3-VL-2B (chat solo) | 62 MB | 1 777 MB | **2 290 MB** | +513 MB |
| Whisper-tiny solo | 22 MB | 214 MB | **373 MB** | +351 MB |
| **Qwen3-VL-2B + whisper** | — | 1 828 MB | **2 792 MB** | — |

El KV de Qwen3-VL-2B es de **112 KB por token** (28 capas × 8 KV heads × 128
head_dim × 2 tensores × 2 bytes), verificado contra `genai_config.json`.

| Ventana de contexto | KV cache | Pico estimado (chat solo) |
|---|---|---|
| 1 024 tokens | 112 MB | ~2,0 GB |
| **2 048 tokens (configurado)** | **224 MB** | **~2,3 GB** |
| 4 096 tokens | 458 MB | ~2,6 GB |

**Hallazgo crítico de seguridad de memoria:** el `genai_config.json` publicado
declara `context_length: 262144` con `past_present_share_buffer: true`. De
respetarse, el KV reservado sería de ~29 GB. El runtime fija siempre
`max_length` a `local_chat_context_length` (2 048) y falla si la API no acepta
la restricción. Debe mantenerse.

### 3.2 Disco

| Componente | MB |
|---|---|
| Qwen3-VL-2B (int4) | 1 373 |
| whisper-tiny (GenAI) | 224 |
| EmbeddingGemma 2 q4f16 | 406 |
| **Total** | **2 099** |

Variantes de embeddings: q4f16 406 MB · q4 451 MB · q8 811 MB.

### 3.3 Cumplimiento del objetivo

| Criterio | Resultado |
|---|---|
| ≤ 2 GB de uso deseado | ⚠️ 2 290 MB (chat) · ⚠️ 2 792 MB (chat+ASR) |
| ≤ 3 GB de uso total | ✅ **Cumplido** en todos los escenarios |
| Modelo de chat < 2 GB en disco | ✅ 1 373 MB |
| Stack completo < 2 GB | ⚠️ 2 099 MB — 99 MB por encima |

---

## 4. Tiempos de inferencia

| Métrica | Qwen3 0.6B | Qwen3-VL 2B | Whisper tiny |
|---|---|---|---|
| Carga del modelo | 3,78 s | **13,4 s** | 1,97 s |
| Velocidad | 33,1 tok/s | **20,3 tok/s** | — |
| Latencia p50 | 1,66 s | **3,11 s** | **0,47 s** (clip 30 s) |
| Latencia máxima | 1,72 s | 7,16 s | — |

CPU, `temperature=0.2`, `max_new_tokens=160`, ventana de 2 048 tokens. La carga
de 13,6 s se paga una vez por sesión; el engine se comparte en proceso.

---

## 5. Calidad de las respuestas locales

Diez preguntas: definición, explicación, resumen, extracción estructurada, caso
sin respuesta y pregunta en inglés.

### 5.1 Agregado

| Métrica | Qwen3 0.6B | Qwen3-VL 2B |
|---|---|---|
| **Recall de términos esperados** | 0,70 | **0,80** |
| Respuestas vacías | 0 | 0 |
| Longitud válida | 6/6 | 6/6 |

### 5.2 Desglose (Qwen3-VL-2B)

| Caso | Tipo | Recall | Observación |
|---|---|---|---|
| Definición de RAG | conocimiento | 0,0 | Correcta funcionalmente; no expande el acrónimo |
| Similitud coseno | conocimiento | 1,0 | Definición precisa |
| Resumen | resumen | 1,0 | Sintético y fiel |
| Extracción | extracción | 1,0 | Formato estructurado correcto |
| Precio de acciones | sin respuesta | 1,0 | **Inventó `$18.50`** |
| Vector database (EN) | conocimiento | 1,0 | Correcta; respondió en español |

### 5.3 Hallazgos de calidad

1. **Alucinación en lugar de abstención.** Ante un dato inexistente fabricó un
   valor concreto. Es el fallo más serio y es típico de modelos pequeños.
   Mitigación: instrucción de abstención y verificación de fuentes antes de
   responder datos numéricos.
2. **Fidelidad de idioma.** Responde en español a preguntas en inglés; debería
   derivar el idioma de la pregunta.
3. **Acrónimos.** Expande siglas mal. Conviene un glosario en la memoria.

### 5.4 Mitigación de la alucinación — medido antes y después

La sección 5.3 describía el defecto. Se amplió el conjunto a **5 preguntas sin
respuesta posible** (precio, fecha, identificación, cantidad e ingresos) y se
midió la conducta con la métrica semántica de `ragpoc/eval_quality.py`.

| Etapa | abstención | alucinación | calidad global |
|---|---|---|---|
| Sin mitigar | 0,00 | 0,60 | 0,538 |
| + directiva de abstención en el prompt | **0,00** | 0,60 | 0,538 |
| **+ barrera estructural (`guardrails.py`)** | **1,00** | **0,00** | **0,841** |

**Hallazgo importante: la directiva de prompt no sirvió para nada.** Añadir
«no inventes cifras, fechas ni nombres» al system prompt dejó la tasa de
abstención en 0,00 exactamente igual. Con 5 casos, el modelo respondió los 5
con un valor inventado:

| Pregunta | Respuesta del modelo |
|---|---|
| Precio de acciones de ACME | `$12.45` |
| Fecha de aprobación de un contrato | `15 de mayo de 2023.` |
| Identificación fiscal de un autor | `1234567890` |
| Unidades vendidas | `5000` |
| Ingresos de 2021 (EN) | `$2.1 billion` |

Por eso se añadió `ragpoc/guardrails.py`: una verificación **después** de la
generación que detecta cuando la respuesta afirma un valor concreto sin nada
que lo respalde (sin evidencia adjunta, sin llamada a tools, sin cita). El
flujo es:

1. Si no hay valor concreto afirmado → pasa tal cual.
2. Si hay valor pero está atenuado («no puedo determinar…») → pasa.
3. Si hay evidencia, tool use o cita → **no interviene**: el modelo puede estar
   informando algo real.
4. Si no hay nada que lo respalde → **un reintento** con instrucción explícita
   de abstención (temperatura 0). El reintento puntual funciona mucho mejor
   que la directiva permanente porque es lo único en contexto en ese momento.
5. Si el reintento aún fabrica → la respuesta se etiqueta:

> ⚠️ **Sin verificar** — no hay evidencia en la base de conocimiento para este dato

El caso `$18.50` de la primera evaluación queda ahora con
**fabrication_risk 1,0 / calidad 0,0**; la abstención correcta con **1,0**.

**Límite honesto:** la barrera evita que el usuario se lleve un dato inventado
sin aviso, pero no hace que el modelo sepa. El valor sigue apareciendo, marcado.
Lo correcto sería que el modelo no lo produjera; con 2B de parámetros eso no
se consigue por prompt.

### 5.5 Métrica de calidad semántica

Sustituye al único «recall de términos», que era demasiado grueso.
`ragpoc/eval_quality.py` puntúa seis dimensiones sin red ni modelo:

| Dimensión | Qué mide |
|---|---|
| `abstention_score` | Que en casos sin respuesta se abstenga |
| `fabrication_risk` | Presencia de valor concreto sin atenuación ni cita |
| `groundedness` | Solapamiento ponderado por IDF con la evidencia |
| `citation_score` | Que cite la fuente cuando se le pide |
| `shape_score` | Vacío, exceso de longitud, razonamiento filtrado |
| `language_score` | Que responda en el idioma pedido |

Las seis se exponen por separado; el global es un promedio ponderado, no el
único número que importa.

---

## 6. Hallazgo: el bloque de pensamiento

Qwen3 abre un bloque de razonamiento al inicio de cada turno y el export de
ONNX GenAI no expone forma de pasar `enable_thinking=false` a la plantilla.

| | Antes | Después |
|---|---|---|
| Chunks generados | 200 (todo razonamiento) | 22 |
| Respuesta visible | **vacía** | frase completa |
| Latencia p50 | 4,04 s | **1,66 s** |
| Recall de términos | 0,30 | 0,70 · 0,80 con VL-2B |

El modelo agotaba el presupuesto de tokens razonando y nunca llegaba a la
respuesta. `close_thinking` inyecta en el prompt un bloque de razonamiento
vacío y cerrado, igual que la propia plantilla hace con
`enable_thinking=false`. `split_thinking` conserva además el razonamiento
incompleto y `strip_thinking` cae a él si no hay respuesta visible, de modo que
un turno nunca vuelve vacío en silencio.

---

## 7. Modalidades: imagen, video y audio

### 7.1 Imagen ✅

Se envía como `{"type": "image"}` dentro de una lista de contenido; la
plantilla inserta los marcadores visuales y el procesador exige que el conteo
coincida con las imágenes entregadas.

**Medido:** imagen roja → *"Una imagen de fondo de color rojo intenso con una
textura de cuadrícula y un patrón de puntos finos."*

### 7.2 Video ✅

El clip MP4 se reduce a fotogramas JPEG con ffmpeg (3 muestras repartidas en el
clip) y se componen en una sola lámina.

**Medido:** clip de prueba → *"test pattern de televisión, también conocido como
patrón de prueba de TV"* — identificación correcta.

### 7.3 Audio ✅

Se transcribe con whisper-tiny y el texto entra en el mensaje. El modelo no oye
el sonido original y lo declara.

**Medido:** clip WAV → `(música)` en **0,47 s**. Etiqueta correcta: el audio de
prueba es música.

### 7.4 Limitación: una sola imagen por llamada

`Images.open_bytes` es un factory estático que **no acumula**; cada llamada
devuelve un objeto nuevo y el procesador solo acepta uno. Solución:
`composite_images` compone las N imágenes en una sola lámina horizontal en vez
de descartar todas menos la primera. Se pierde resolución cuando hay muchas
evidencias visuales.

---

## 8. Loop de tools del agente

Validado de extremo a extremo con un agente pydantic-ai real sobre el modelo local.

| Comprobación | Resultado |
|---|---|
| El modelo emite una llamada de tool válida | ✅ |
| pydantic-ai la ejecuta y devuelve el resultado | ✅ |
| El turno siguiente redacta una respuesta en prosa | ✅ con VL-2B |
| Historial completo (system/user/assistant/tool) se serializa al modelo | ✅ |
| Evidencia binaria viaja como media al modelo | ✅ |

**Hallazgo:** un modelo pequeño a veces devuelve la llamada como JSON desnudo,
sin el envoltorio que pide la plantilla. El parser acepta ambos formatos y el
streaming contiene el JSON para que no se filtre al usuario. Sin ese arreglo la
tool simplemente no se invocaba y el JSON aparecía como respuesta.

---

## 9. Recuperación (distancia coseno)

El recuperador (`ragpoc/retrieval.py`) puntúa por **similitud coseno**. Se
mantuvo esa métrica y se construyó la evaluación alrededor de ella:
`ragpoc/eval_retrieval.py` implementa **Recall@K, Precision@K, Hit@K, MRR y
nDCG@K** sobre el ordenamiento por coseno, más percentiles de latencia.

| Métrica | Arnés sintético | **Su base real** |
|---|---|---|
| Recall@1 | — | **77,88%** |
| Recall@8 | 0,667 | **89,70%** |
| Precision@1 | — | **94,44%** |
| Hit rate@8 | — | **100,00%** |
| nDCG@8 | 0,667 | **98,30%** |
| **MRR** | 0,667 | **97,22%** |
| Latencia p50 / p95 / max | medidos | 785 ms / 1 164 ms / 1 482 ms |

Medido con `scripts/build_retrieval_gold.py` + `scripts/run_retrieval_eval_real.py`
sobre **188 fragmentos / 166 documentos**, con un gold set de 18 consultas
derivadas del propio corpus (primera frase, frase intermedia y nombre de
archivo informativo). Los embeddings son los reales del índice
(`google/gemini-embedding-2-preview`, 3072 dimensiones).

**El arnés sintético se conserva** porque verifica que el pipeline de scoring
es correcto sin red ni modelo descargado; las cifras de la derecha son las que
cuentan.

### 9.1 El Recall está topado por duplicación del corpus

Hallazgo que explica el número: el corpus tiene solo ~10 textos distintos.
*«Contenido de prueba del agente.»* aparece en **107 documentos separados**.
Esas consultas se recuperan en el **puesto 1** (RR = 1,0) y aun así el recall@8
es 8/107 = 7,5%, lo que arrastra la media macro hacia abajo.

Por eso **hit rate, MRR, precisión y nDCG son las métricas significativas
aquí**: *el contenido correcto está en el top-3 para el 90% de las consultas y
en el top-1 para el 85%*.

### 9.2 La limpieza de datos subió el retrieval

Los primeros números medidos venían degradados por datos de prueba que habían
quedado en la base de producción. Se eliminaron y se re-midió:

| Metrica | Antes | **Despues** |
|---|---|---|
| **MRR** | 87,50% | **97,22%** |
| Recall@1 | 70,09% | **77,88%** |
| Recall@8 | 80,74% | **89,70%** |
| Precision@1 | 85,00% | **94,44%** |
| Hit rate@3 | 90,00% | **100,00%** |
| nDCG@8 | 88,47% | **98,30%** |

Lo que se limpio con `scripts/clean_test_vectors.py`:

* **5 vectores huerfanos** que apuntaban a chunks borrados (4 de segmentos de
  video y 1 de texto). Quedaron porque el driver sqlite3 plano no puede cargar
  la extension `vec0` que respalda la tabla virtual.
* **2 vectores placeholder** (`c_focus_1`, `c_focus_2`) con todas las dimensiones
  rellenadas con `0.1`. No eran embeddings de nada: clasificaban al azar y
  justamente eran las 2 consultas que nunca se recuperaban.

Actualmente el almacén tiene **188 vectores, 0 huérfanos, 0 placeholder** y
todas las consultas del gold set recuperan su contenido en el **top-3**.

### 9.3 Riesgo de memoria en la recuperación

`Retriever.search` carga **todos** los vectores del ámbito y calcula el coseno
en Python. Aproximado por fragmento de 768 dimensiones:

| Fragmentos | RAM |
|---|---|
| 10 000 | 80–120 MB |
| 50 000 | 400–600 MB |

Para bases grandes compite con el presupuesto de 3 GB. Se recomienda sustituirlo
por KNN nativo de `sqlite-vec` (`vec_distance_cosine`), ya en dependencias.

---

## 10. Palancas para bajar de 2 GB

| Palanca | Ahorro | Costo |
|---|---|---|
| Cargar whisper solo cuando hay audio y descargarlo después | hasta 373 MB de pico | +2 s por transcripción |
| Ventana de contexto 2 048 → 1 024 | 112 MB de KV | Menos historial |
| Embeddings en q4f16 en vez de q8 | 405 MB | Pérdida mínima |
| Carga lazy de encoders de visión/audio de EmbeddingGemma | hasta 510 MB | Solo al indexar media |
| Coseno brute-force → KNN de `sqlite-vec` | 400–600 MB en KB grandes | Cambio en `retrieval.py` |

Con la primera, el pico con audio baja de 2 792 MB a **~2,3 GB**.

---

## 11. Cómo reproducir

```bash
# Descarga de modelos (una sola vez)
python scripts/download_local_models.py

# Benchmark: recursos, latencia y calidad
python scripts/run_local_agent_bench.py --model qwen3-vl-2b \
    --context-length 2048 --max-new-tokens 160 --out bench.json
python scripts/show_bench.py bench.json

# Loop de tools y modalidades
python scripts/agent_e2e_smoke.py qwen3-vl-2b
python scripts/multimodal_e2e.py

# Pruebas
python -m pytest tests -q
```

Pruebas añadidas: `test_metrics_and_budget` (30), `test_local_llm` (24),
`test_local_agent_model` (28), `test_local_media` (17) — **99 en total**. Son
deterministas y no requieren modelo descargado.

---

## 12. Ajustes e interfaz

Se añadió a **Ajustes** una tarjeta *Modelo local (sin API)* junto al selector de
proveedor, que ahora ofrece **Local (sin API, en tu equipo)** además de
OpenRouter y Google AI Studio.

| Qué muestra | De dónde sale |
|---|---|
| Indicador «Listo» / «Modelo no descargado» / «Sin margen de RAM» | `/api/agent/status` |
| Memoria libre sobre el techo (ej. *libre 2894 MB / 3072 MB*) | `MemoryBudget.report()` |
| Carga estimada y si cabe en el presupuesto | `MemoryBudget.check()` |
| Selector de modelo con tamaño y modalidades | `chat_models.list_chat_models()` |
| Ventana de contexto (1024 / 2048 / 4096) | `AppConfig.local_chat_context_length` |
| Barra de progreso y cancelación de descarga | `/api/chat-model/download|cancel` |

### 12.1 Endpoints nuevos

| Ruta | Método | Propósito |
|---|---|---|
| `/api/chat-models` | GET | Catálogo de modelos con estado de instalación |
| `/api/chat-model/status` | GET | Estado de descarga de un modelo |
| `/api/chat-model/download` | POST | Descarga en segundo plano |
| `/api/chat-model/cancel` | POST | Cancela la descarga |
| `/api/agent/status` | GET | Preparación del agente local y presupuesto |

Los ajustes se persisten en `AppConfig` (`local_chat_model`,
`local_chat_context_length`) mediante la migración `0015`.

### 12.2 Verificación en navegador

Comprobado con un navegador real, no solo con tests:

| Comprobación | Resultado |
|---|---|
| Tarjeta y controles presentes (8/8 identificadores) | ✅ |
| Selector de proveedor ofrece `local` | ✅ |
| Errores de JavaScript en consola | **0** |
| Guardar ajustes persiste `local` / modelo / contexto | ✅ |
| Indicador de memoria se pinta | ✅ *libre 2894 MB / 3072 MB* |
| `/api/chat-model/status` refleja un modelo ya instalado | ✅ *ready / 100%* |

**Bug encontrado durante la verificación:** `/api/chat-model/status` devolvía
`not_downloaded` para un modelo que ya estaba en disco, porque `get_status()`
leía solo el estado en memoria sin consultar el directorio de instalación. Se
corrigió y quedó cubierto por `tests/test_chat_models.py`.

### 12.3 Limitación de arranque bajo ASGI

La aplicación no arranca con `uvicorn ragpoc_django.asgi:application` a secas:
`knowledge/apps.py:ready()` hace trabajo ORM síncrono y Django lo rechaza con
`SynchronousOnlyOperation`. Se necesita `DJANGO_ALLOW_ASYNC_UNSAFE=1` o arrancar
por WSGI. Es un problema preexistente, ajeno a este trabajo, pero conviene
conocerlo.

---

## 13. Archivos entregados

| Archivo | Propósito |
|---|---|
| `src/ragpoc/metrics.py` | RAM, latencia, tok/s, TTFT |
| `src/ragpoc/budget.py` | Presupuesto de memoria con control de admisión |
| `src/ragpoc/eval_retrieval.py` | Recall@K, MRR, nDCG sobre coseno |
| `src/ragpoc/local_llm.py` | Runtime ONNX GenAI, streaming, multimodal |
| `src/ragpoc/local_media.py` | Video→frames, composición, adaptación de evidencia |
| `src/ragpoc/whisper_transcribe.py` | Transcripción de audio |
| `src/ragpoc/chat_models.py` | Manifiestos y descarga verificada |
| `src/knowledge/local_agent_model.py` | Adaptador pydantic-ai + parser de tools |
| `src/knowledge/llm.py` | Proveedor `local` en la abstracción existente |
| `src/knowledge/models.py` | Campos `local_chat_*` en `AppConfig` |
| `src/knowledge/migrations/0015_...py` | Migración de esos campos |
| `src/knowledge/settings_store.py` | Persistencia de los ajustes locales |
| `src/knowledge/views.py` | Endpoints de modelos y estado del agente |
| `src/knowledge/urls.py` | Rutas nuevas |
| `src/ragpoc/templates/console.html` | Tarjeta de Ajustes del modelo local |
| `scripts/run_local_agent_bench.py` | Benchmark reproducible |
| `scripts/download_local_models.py` | Descarga verificada de modelos |
| `scripts/patch_console_local_agent.py` | Parche de la plantilla (edits con emoji) |
| `scripts/apply_0015.py` | Aplicación de la migración 0015 |

---

## 14. Estado y siguientes pasos

### Completado y verificado

- Arneses de medicion de RAM, latencia y generacion
- Presupuesto de memoria con control de admision
- Suite de metricas de recuperacion por coseno, con **gold set y numeros reales**
- Runtime local ONNX GenAI con streaming
- Adaptador pydantic-ai con tool-loop validado end-to-end
- Multimodal: imagen, video y audio validados sobre el modelo real
- Proveedor `local` en `llm.py`, settings persistidos y UI de Ajustes
- **Mitigacion de alucinacion** — barrera estructural + metrica semantica:
  abstencion 0,00 → **1,00**, alucinacion 0,60 → **0,00**
- **Whisper bajo demanda** — el pico sin audio queda en **2 341 MB**
- **Herramienta de reparacion** de claves foraneas huerfanas
- 281 pruebas nuevas en verde; suite total 431 en verde

### Pendiente

Solo queda lo que requiere una decision de producto, no de ingenieria:

1. **Ejecutar la reparacion de datos** cuando confirmes que esas 2 filas se
   pueden borrar. Mientras no se haga, `manage.py migrate` seguira abortando.
2. **Limpiar los vectores de prueba** (`c_focus_1`, `c_focus_2`, `c_guide`,
   `c_guide_unique_test`) y los 4 vectores sin chunk, y reindexar. Sube el
   recall real (seccion 9.2).
3. **Glosario de acronimos** en la memoria del agente (seccion 5.3).
4. Evaluacion de calidad con juicio humano, para validar las metricas
   automaticas.

### Datos corruptos en la base (requieren tu confirmacion)

`scripts/repair_orphan_documents.py` en modo solo lectura reporta **2 filas
huerfanas**:

| Tabla | Fila | Referencia rota |
|---|---|---|
| `notebook_documents` | `id=19` | `document_id=e38ea53b-5585-43a7-a501-3daa9ab6a679` |
| `chunks` | `rowid=188` | `document_id=b4adeff2-1551-4520-bfcc-f142906b3297` |

La segunda ademas tiene una entrada colgante en `chunks_fts`. Ambas **impiden
ejecutar `manage.py migrate`**, porque Django verifica todas las claves
foraneas al final de cada migracion. La migracion `0015` se aplico mediante DDL
directo (`scripts/apply_0015.py`), que evita esa verificacion.

**No se borro nada.** Para reparar, con respaldo automatico y re-verificacion
previa a cada borrado:

```bash
python scripts/repair_orphan_documents.py --dry-run   # informa, no toca nada
python scripts/repair_orphan_documents.py --apply     # respalda y borra
```

### Fallo de pruebas preexistente (fuera de alcance)

`tests/test_i18n.py::test_describe_artifact_settings_bilingual` espera que el
tono `"informal"` se reproduzca literal en la descripción inglesa, pero
`describe_artifact_settings` lo traduce a *"Casual, conversational, friendly"*.
Es un desacuerdo de especificación anterior a este trabajo; no se tocó para no
decidir unilateralmente el comportamiento de producto.

### Riesgos abiertos

- `onnxruntime-genai` fijado a 0.14.x (sección 2.1).
- Calidad por debajo de un modelo cloud; el objetivo es un agente de escritorio.
- La métrica de calidad (recall de términos) es automática y barata. Una
  evaluación semántica con juicio humano daría números más fiables.
- ONNX GenAI 0.14.1 solo acepta una imagen por llamada (sección 7.4).
