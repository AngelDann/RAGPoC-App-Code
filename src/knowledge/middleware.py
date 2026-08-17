import base64
import secrets

from django.http import HttpResponse, JsonResponse

from ragpoc.config import get_settings


class BasicAuthMiddleware:
    """
    Middleware that provides Basic Authentication parity with FastAPI baseline.
    Protected paths:
      - / (Console UI)
      - /api/*
      - /chat/*
      - /ingest
      - /search
      - /documents
    Admin paths (/admin/*) use Django's standard session/auth system.

    When RAGPOC_DESKTOP_MODE is enabled (set by desktop_launcher.py for the
    compiled local build), the app only ever listens on 127.0.0.1 for a single
    local user, so Basic Auth is skipped entirely.
    """

    def __init__(self, get_response):
        self.get_response = get_response
        self.settings = get_settings()

    def __call__(self, request):
        path = request.path_info
        # Skip auth for static files or admin
        if path.startswith("/static/") or path.startswith("/admin/"):
            return self.get_response(request)

        if self.settings.desktop_mode:
            return self.get_response(request)

        if not self.settings.ui_password:
            return HttpResponse("RAGPOC_UI_PASSWORD must be configured before serving the application.", status=503)

        auth_header = request.META.get("HTTP_AUTHORIZATION", "")
        if not auth_header.startswith("Basic "):
            return self._auth_required_response(path)

        try:
            auth_decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
            username, password = auth_decoded.split(":", 1)
        except Exception:
            return self._auth_required_response(path)

        valid_user = secrets.compare_digest(username, self.settings.ui_username)
        valid_pass = secrets.compare_digest(password, self.settings.ui_password)

        if not (valid_user and valid_pass):
            return self._auth_required_response(path)

        request.auth_username = username
        return self.get_response(request)

    def _auth_required_response(self, path: str):
        if path.startswith("/api/") or path.startswith("/chat/") or path in ("/search", "/ingest", "/documents"):
            response = JsonResponse({"detail": "Authentication required."}, status=401)
        else:
            response = HttpResponse("Authentication required.", status=401)
        response["WWW-Authenticate"] = 'Basic realm="RAGPoC"'
        return response
