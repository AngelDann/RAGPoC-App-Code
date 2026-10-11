"""Patch console.html with the Local agent settings card.

The template contains emoji that round-trip badly through the shell, so the
edits are applied from a script with explicit encoding rather than by hand.

Anchors are exact strings taken from the file; every edit reports whether it
matched, so a template change upstream fails loudly instead of silently doing
nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "src" / "ragpoc" / "templates" / "console.html"

OPTION_ANCHOR = '              <option value="google">\u2601\ufe0f Google AI Studio</option>\n'
OPTION_NEW = OPTION_ANCHOR + '              <option value="local">\U0001f4bb Local (sin API, en tu equipo)</option>\n'

CARD_ANCHOR = """        <div class="craft-card p-3">
          <div class="fw-bold small text-dark mb-2 d-flex align-items-center gap-2">
            <i class="bi bi-arrow-repeat text-dark"></i> Actualizaciones"""

LOCAL_CARD = """        <!-- Local Agent Card -->
        <div class="craft-card p-3" id="settings-local-agent-card">
          <div class="fw-bold small text-dark mb-2 d-flex align-items-center gap-2">
            <i class="bi bi-hdd-network text-primary"></i>
            <span data-i18n="settings.local_agent_title">Modelo local (sin API)</span>
            <span class="badge bg-light text-secondary border ms-auto" id="local-agent-ready-badge" style="font-size:10px;">-</span>
          </div>
          <div class="text-muted mb-2" style="font-size:10.5px;" data-i18n="settings.local_agent_hint">
            Ejecuta el agente en tu equipo sin enviar nada a la nube. Necesita el modelo descargado.
          </div>
          <div class="mb-2">
            <label class="form-label text-muted mb-1" style="font-size:11px;">Modelo</label>
            <select id="settings-local-chat-model-select" class="form-select form-select-sm border" style="border-radius: var(--radius-sm); background: #ffffff;">
              <option value="qwen3-vl-2b">Qwen3-VL 2B - texto, imagen y video (1.4 GB)</option>
              <option value="qwen3-0.6b">Qwen3 0.6B - solo texto, muy ligero (0.5 GB)</option>
            </select>
            <div class="text-muted mt-1" style="font-size:10.5px;" id="local-chat-model-desc"></div>
          </div>
          <div class="mb-2">
            <label class="form-label text-muted mb-1" style="font-size:11px;">Ventana de contexto</label>
            <select id="settings-local-chat-context-select" class="form-select form-select-sm border" style="border-radius: var(--radius-sm); background: #ffffff;">
              <option value="1024">1024 tokens - menor uso de memoria</option>
              <option value="2048" selected>2048 tokens - recomendado</option>
              <option value="4096">4096 tokens - mas historial</option>
            </select>
            <div class="text-muted mt-1" style="font-size:10.5px;">
              Controla el KV cache. A mas ventana, mas RAM durante la generacion.
            </div>
          </div>

          <div id="local-agent-status-box" class="mb-2 p-2 rounded border bg-light">
            <div class="d-flex justify-content-between align-items-center mb-1" style="font-size:11px;">
              <span class="fw-medium text-dark"><i class="bi bi-gauge text-primary"></i> Memoria</span>
              <span id="local-agent-memory" class="fw-bold text-primary">-</span>
            </div>
            <div id="local-agent-memory-note" class="text-muted" style="font-size:10px;">-</div>
            <div class="d-flex justify-content-between align-items-center mt-1" style="font-size:10px;">
              <span id="local-chat-download-bytes" class="text-muted"></span>
              <span id="local-chat-download-actions">
                <button class="btn-craft btn-craft-sm" id="btn-download-chat-model" style="display:none;">
                  <i class="bi bi-cloud-arrow-down"></i> Descargar modelo
                </button>
                <a href="javascript:void(0)" id="btn-cancel-chat-model-download" class="text-danger text-decoration-none" style="display:none;">
                  <i class="bi bi-x-circle"></i> Cancelar
                </a>
              </span>
            </div>
            <div class="progress mt-1" style="height: 6px; display:none;" id="local-chat-progress-bar">
              <div id="local-chat-progress-fill" class="progress-bar progress-bar-striped progress-bar-animated bg-primary" role="progressbar" style="width: 0%;"></div>
            </div>
            <div id="local-chat-download-error" class="text-danger mt-1" style="font-size:10px; display:none;"></div>
          </div>

          <div class="d-flex justify-content-end">
            <button class="btn-craft btn-craft-sm" id="btn-save-local-agent">
              <i class="bi bi-save"></i> Guardar ajustes locales
            </button>
          </div>
        </div>

"""

I18N_ES_ANCHOR = "          llm_provider_hint: 'Aplica al agente de chat"
I18N_ES_NEW = """          local_agent_title: 'Modelo local (sin API)',
          local_agent_hint: 'Ejecuta el agente en tu equipo sin enviar nada a la nube. Necesita el modelo descargado.',
"""
I18N_EN_NEW = """          local_agent_title: 'Local model (no API)',
          local_agent_hint: 'Runs the agent on your machine without sending anything to the cloud. Requires the downloaded model.',
"""

SCRIPT = """
    // ---- Local agent (ONNX Runtime GenAI) settings ----
    (function () {
      const $ = (sel) => document.querySelector(sel);
      let chatDownloadTimer = null;

      async function api(url, opts) {
        const res = await fetch(url, opts);
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      }

      function fmtMB(bytes) {
        if (!bytes) return '0 MB';
        return (bytes / 1024 / 1024).toFixed(1) + ' MB';
      }

      async function refreshLocalAgentStatus() {
        const badge = $('#local-agent-ready-badge');
        const mem = $('#local-agent-memory');
        const note = $('#local-agent-memory-note');
        try {
          const s = await api('/api/agent/status');
          if (badge) {
            const ready = s.model_installed && s.would_fit;
            badge.textContent = ready ? 'Listo' : (s.model_installed ? 'Sin margen de RAM' : 'Modelo no descargado');
            badge.className = 'badge ms-auto border ' + (ready ? 'bg-success text-white' : 'bg-light text-secondary');
            badge.style.fontSize = '10px';
          }
          if (mem) {
            mem.textContent = 'libre ' + Math.round(s.memory.headroom_mb) + ' MB / ' + Math.round(s.memory.hard_limit_mb) + ' MB';
          }
          if (note) {
            note.textContent = 'Carga estimada: ' + s.estimate_mb + ' MB - ' + (s.would_fit ? 'cabe en el presupuesto' : s.budget_note);
          }
          const desc = $('#local-chat-model-desc');
          if (desc && s.manifest) desc.textContent = s.manifest.description;
          return s;
        } catch (e) {
          if (badge) { badge.textContent = 'sin conexion'; }
          return null;
        }
      }

      function renderChatDownload(st) {
        const bar = $('#local-chat-progress-bar');
        const fill = $('#local-chat-progress-fill');
        const bytes = $('#local-chat-download-bytes');
        const dl = $('#btn-download-chat-model');
        const cancel = $('#btn-cancel-chat-model-download');
        const err = $('#local-chat-download-error');
        if (!bar) return;
        const downloading = st.state === 'downloading';
        bar.style.display = downloading ? 'block' : 'none';
        if (fill) fill.style.width = (st.percent || 0) + '%';
        if (bytes) bytes.textContent = st.total_bytes ? (fmtMB(st.downloaded_bytes) + ' / ' + fmtMB(st.total_bytes)) : '';
        if (dl) dl.style.display = (st.state === 'not_downloaded' || st.state === 'error') ? 'inline-block' : 'none';
        if (cancel) cancel.style.display = downloading ? 'inline-block' : 'none';
        if (err) {
          err.style.display = st.error ? 'block' : 'none';
          err.textContent = st.error || '';
        }
      }

      async function pollChatDownload() {
        try {
          const st = await api('/api/chat-model/status');
          renderChatDownload(st);
          if (st.state === 'downloading') {
            chatDownloadTimer = setTimeout(pollChatDownload, 1200);
          } else {
            chatDownloadTimer = null;
                refreshLocalAgentStatus();
          }
        } catch (e) {
          chatDownloadTimer = null;
        }
      }

      async function initLocalChatCard() {
        await refreshLocalAgentStatus();
        try {
          const st = await api('/api/chat-model/status');
          renderChatDownload(st);
          if (st.state === 'downloading' && !chatDownloadTimer) pollChatDownload();
        } catch (e) { /* ignore */ }
      }

      const dlBtn = $('#btn-download-chat-model');
      if (dlBtn) {
        dlBtn.onclick = async () => {
          const sel = $('#settings-local-chat-model-select');
          try {
            await api('/api/chat-model/download', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ model: sel ? sel.value : '' })
            });
            if (!chatDownloadTimer) pollChatDownload();
          } catch (e) {
            const err = $('#local-chat-download-error');
            if (err) { err.style.display = 'block'; err.textContent = e.message; }
          }
        };
      }

      const cancelBtn = $('#btn-cancel-chat-model-download');
      if (cancelBtn) {
        cancelBtn.onclick = async () => {
          const sel = $('#settings-local-chat-model-select');
          try {
            await api('/api/chat-model/cancel', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ model: sel ? sel.value : '' })
            });
          } catch (e) { /* ignore */ }
        };
      }

      const saveBtn = $('#btn-save-local-agent');
      if (saveBtn) {
        saveBtn.onclick = async () => {
          const sel = $('#settings-local-chat-model-select');
          const ctx = $('#settings-local-chat-context-select');
          const payload = {
            llm_provider: 'local',
            local_chat_model: sel ? sel.value : 'qwen3-vl-2b',
            local_chat_context_length: ctx ? parseInt(ctx.value, 10) : 2048
          };
          try {
            const res = await fetch('/api/settings', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload)
            });
            if (!res.ok) throw new Error('HTTP ' + res.status);
            await refreshLocalAgentStatus();
            const st = await api('/api/chat-model/status');
            renderChatDownload(st);
            if (st.state !== 'ready' && st.state !== 'downloading') {
              if (confirm('El modelo local aun no esta descargado.\\n\\nDescargarlo ahora? (' + fmtMB(st.total_bytes) + ')')) {
                await api('/api/chat-model/download', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ model: payload.local_chat_model })
                });
                if (!chatDownloadTimer) pollChatDownload();
              }
            }
          } catch (e) {
            const err = $('#local-chat-download-error');
            if (err) { err.style.display = 'block'; err.textContent = e.message; }
          }
        };
      }

      // Populate the card once the settings endpoint has been used by the page.
      const originalLoad = window.loadSettingsStatus;
      if (typeof originalLoad === 'function') {
        window.loadSettingsStatus = async function () {
          const out = await originalLoad.apply(this, arguments);
          await initLocalChatCard();
          return out;
        };
      } else {
        document.addEventListener('DOMContentLoaded', initLocalChatCard);
        setTimeout(initLocalChatCard, 800);
      }
    })();
"""


def apply(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        print(f"  [FALLA] {label}: {count} coincidencias (se esperaba 1)")
        return text
    print(f"  [ok]    {label}")
    return text.replace(old, new, 1)


def main() -> int:
    if not TEMPLATE.exists():
        print(f"no template at {TEMPLATE}", file=sys.stderr)
        return 1
    text = TEMPLATE.read_text(encoding="utf-8")
    print(f"plantilla: {TEMPLATE} ({len(text)} caracteres)")

    text = apply(text, OPTION_ANCHOR, OPTION_NEW, "opcion de proveedor 'local'")
    text = apply(text, CARD_ANCHOR, LOCAL_CARD + CARD_ANCHOR, "tarjeta de modelo local")
    text = apply(text, I18N_ES_ANCHOR, I18N_ES_NEW + I18N_ES_ANCHOR, "claves i18n ES")
    text = apply(text, "          llm_provider_hint: 'Applies to the chat agent",
                 I18N_EN_NEW + "          llm_provider_hint: 'Applies to the chat agent",
                 "claves i18n EN")

    if "Local agent (ONNX Runtime GenAI) settings" not in text:
        closing = text.rfind("</script>")
        if closing < 0:
            print("  [FALLA] no se encontro </script> para anadir el bloque JS")
            return 1
        text = text[:closing] + SCRIPT + text[closing:]
        print("  [ok]    bloque JS de la tarjeta local")
    else:
        print("  [skip]  bloque JS ya presente")

    TEMPLATE.write_text(text, encoding="utf-8")
    print(f"escrito: {len(text)} caracteres")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
