# RAGPoC — Knowledge Studio

Aplicación de escritorio para Windows que convierte tus documentos en una base de conocimiento
consultable con IA, sin subir nada a ningún servicio propio: todo se guarda en un SQLite local,
junto al ejecutable.

## Qué hace

- **Espacios de trabajo, cuadernos y páginas** para organizar la información con un editor propio.
- **Ingesta multimodal**: documentos, imágenes, audio, vídeo y PDF. La búsqueda devuelve el
  fragmento real que respalda cada respuesta, incluido el trozo de medio correspondiente.
- **Chat con agente** sobre tu propio material, con memoria persistente y habilidades definibles.
- **Generación de artefactos** (resúmenes, informes, material derivado) exportables a PDF.
- **Búsqueda transversal** sobre todo el contenido indexado, con embeddings vectoriales
  (`sqlite-vec`) almacenados en la misma base de datos.
- **Actualizaciones integradas**: la app comprueba GitHub Releases y se actualiza desde Ajustes.

## Descargas

En la [página de releases](https://github.com/AngelDann/RAGPoC-App-Code/releases):

- **`RAGPoC-Setup.exe`** — instalador recomendado. Instala para el usuario actual, sin permisos de
  administrador, y crea los accesos directos.
- **`RAGPoC-windows.zip`** — versión portable, y el paquete que usa el actualizador integrado.
  Descomprímelo en una carpeta con permiso de escritura y ejecuta `RAGPoC.exe`.

### Sobre el aviso de SmartScreen

Puede que Windows muestre *"Windows protegió tu PC"* al abrir una versión recién publicada. Es el
comportamiento normal de SmartScreen con binarios que todavía no han acumulado reputación de
descarga; no significa que el fichero esté alterado. Para continuar: **Más información →
Ejecutar de todas formas**. Cada release publica el binario firmado (ver la política de firma más
abajo), y el aviso desaparece a medida que la reputación se acumula.

## Requisitos

- Windows 10 o superior (x64).
- WebView2, que viene preinstalado en Windows 11 y en Windows 10 actualizado. Si falta, la
  aplicación abre igualmente en una ventana de Edge/Chrome.
- Una clave de API de [OpenRouter](https://openrouter.ai/) para las funciones de IA. Se configura
  desde Ajustes dentro de la app, y se guarda solo en tu equipo.

## Dónde se guardan tus datos

Todo vive en la carpeta `data\` junto al ejecutable: la base de datos SQLite, los ficheros
subidos, los medios derivados y los renders. Las actualizaciones no la tocan, y desinstalar no la
borra. Para hacer copia de seguridad, copia esa carpeta.

## Compilar desde el código

```bash
git clone https://github.com/AngelDann/RAGPoC-App-Code.git
cd RAGPoC-App-Code
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-windows.txt
pyinstaller ragpoc.spec --clean --noconfirm
```

El resultado queda en `dist\RAGPoC\`. Para ejecutarlo sin compilar: `python desktop_launcher.py`.
El instalador se genera con Inno Setup: `iscc installer.iss`.

Las versiones publicadas se compilan exclusivamente desde
[`.github/workflows/release.yml`](.github/workflows/release.yml), disparado por una etiqueta de
versión (`v1.2.3`), a partir del código de este repositorio.

## Política de firma de código (Code signing policy)

Los binarios publicados se firman con firma de código gratuita proporcionada por
[SignPath.io](https://signpath.io/), con certificado de
[SignPath Foundation](https://signpath.org/).

- **Autoría y revisión de los cambios**: los responsables del repositorio
  [AngelDann/RAGPoC-App-Code](https://github.com/AngelDann/RAGPoC-App-Code).
- **Aprobación de releases**: Angel Daniel Lopez Alvarez, propietario del repositorio.
- **Origen de los binarios firmados**: únicamente artefactos producidos por el flujo de trabajo
  `release.yml` de GitHub Actions sobre una etiqueta de versión de este repositorio. No se firma
  ningún binario compilado en local ni de ninguna otra procedencia.
- **Alcance**: se firman tanto `RAGPoC.exe` como el instalador `RAGPoC-Setup.exe`.

## Política de privacidad (Privacy policy)

La aplicación no recopila telemetría, analítica ni datos de uso, y no envía tu contenido a ningún
servicio del autor. Solo hace dos tipos de conexión saliente:

1. **`openrouter.ai`** — únicamente si tú configuras tu propia clave de API, y solo para las
   funciones de IA que invoques: se envía el texto de tus consultas y los fragmentos de tus
   documentos necesarios para generar embeddings o respuestas. Sin clave configurada, la app no
   contacta con OpenRouter en absoluto.
2. **`api.github.com`** — para consultar si existe una versión más reciente, y
   `github.com` / `objects.githubusercontent.com` para descargar la actualización si la aceptas.
   Esta comprobación no envía ningún dato tuyo.

El servidor local escucha exclusivamente en `127.0.0.1`, nunca en la red. Tus documentos, la base
de datos y tu clave de API no salen de tu equipo por ninguna otra vía.

## Licencia

[MIT](LICENSE) © 2026 Angel Daniel Lopez Alvarez.
