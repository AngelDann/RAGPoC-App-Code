import ctypes
import http.server
import json
import socket
import threading
import time

import desktop_launcher as launcher

HOST = "127.0.0.1"
# Deliberately far from PREFERRED_PORT (47823) and its PORT_SCAN_SPAN (47823-47842): using that
# range would make these tests silently talk to a real RAGPoC instance a developer happens to
# have open, and fail in a way that has nothing to do with the code under test.
TEST_PORT_BASE = 58230


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


class _SilentListener:
    """Accepts connections and never answers -- the "silent" case probe_port() has to resolve:
    a port that is bound but not yet serving, exactly what our own launcher looks like in the
    gap between reserving its socket and uvicorn actually being ready."""

    def __init__(self, port: int):
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((HOST, port))
        self._sock.listen(4)
        self._sock.settimeout(0.2)
        self._stop = threading.Event()
        self._held = []
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                self._held.append(self._sock.accept()[0])
            except OSError:
                pass

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
        for conn in self._held:
            conn.close()
        self._sock.close()


class _HealthServer:
    """A tiny HTTP server answering GET /health with an arbitrary JSON body, used to stand in
    for either a real RAGPoC instance ({"app": "ragpoc"}) or an unrelated server (anything
    else) -- probe_port() tells the two apart by that body, not by whether something merely
    answers."""

    def __init__(self, port: int, payload: dict):
        handler = self._make_handler(payload)
        self._httpd = http.server.HTTPServer((HOST, port), handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @staticmethod
    def _make_handler(payload: dict):
        body = json.dumps(payload).encode("utf-8")

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        return Handler

    def close(self):
        self._httpd.shutdown()
        self._thread.join(timeout=2)
        self._httpd.server_close()


def test_port_is_free_reports_true_on_an_unused_port_and_false_once_bound():
    port = TEST_PORT_BASE + 0
    assert launcher.port_is_free(HOST, port) is True

    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        holder.bind((HOST, port))
        holder.listen(1)
        assert launcher.port_is_free(HOST, port) is False
    finally:
        holder.close()


def test_probe_port_reports_silent_when_nothing_is_listening():
    port = TEST_PORT_BASE + 1
    assert launcher.probe_port(HOST, port) == "silent"


def test_probe_port_reports_other_for_a_listener_that_never_answers():
    port = TEST_PORT_BASE + 2
    listener = _SilentListener(port)
    try:
        assert launcher.probe_port(HOST, port) == "other"
    finally:
        listener.close()


def test_probe_port_reports_ragpoc_for_a_server_whose_health_endpoint_says_so():
    port = TEST_PORT_BASE + 3
    server = _HealthServer(port, {"app": "ragpoc", "status": "ok"})
    try:
        assert launcher.probe_port(HOST, port) == "ragpoc"
    finally:
        server.close()


def test_probe_port_reports_other_for_a_server_whose_health_endpoint_says_something_else():
    port = TEST_PORT_BASE + 4
    server = _HealthServer(port, {"app": "some-other-app"})
    try:
        assert launcher.probe_port(HOST, port) == "other"
    finally:
        server.close()


def test_acquire_port_binds_the_preferred_port_when_it_is_free():
    port = TEST_PORT_BASE + 5
    sock, acquired_port = launcher.acquire_port(HOST, port)
    try:
        assert sock is not None
        assert acquired_port == port
        # A second bind attempt on the same port must fail while we hold it -- proof the
        # returned socket actually reserved it rather than just having probed and released it.
        assert launcher.port_is_free(HOST, port) is False
    finally:
        if sock:
            sock.close()


def test_acquire_port_attaches_to_an_existing_ragpoc_on_the_preferred_port():
    # This is the case a naive simplification of probe_port() breaks: if a prior launch fell
    # through to a fallback port and IS a real RAGPoC instance, acquire_port() must recognise
    # it and hand back (None, port) so the caller attaches a window instead of starting a
    # second server against the same sqlite file.
    port = TEST_PORT_BASE + 6
    server = _HealthServer(port, {"app": "ragpoc"})
    try:
        sock, acquired_port = launcher.acquire_port(HOST, port)
        assert sock is None
        assert acquired_port == port
    finally:
        server.close()


def test_acquire_port_skips_a_port_held_by_an_unrelated_server():
    port = TEST_PORT_BASE + 7
    server = _HealthServer(port, {"app": "someone-elses-app"})
    try:
        sock, acquired_port = launcher.acquire_port(HOST, port)
        try:
            assert sock is not None
            assert acquired_port != port
            assert acquired_port in range(port, port + launcher.PORT_SCAN_SPAN)
        finally:
            if sock:
                sock.close()
    finally:
        server.close()


def test_wait_for_server_raises_if_the_server_thread_dies_before_the_port_opens():
    port = TEST_PORT_BASE + 8

    def dies_immediately():
        pass

    dead_thread = threading.Thread(target=dies_immediately)
    dead_thread.start()
    dead_thread.join()  # already dead by the time wait_for_server looks at it

    try:
        launcher.wait_for_server(HOST, port, dead_thread, timeout=2.0)
        raise AssertionError("expected wait_for_server to raise")
    except RuntimeError as exc:
        assert "servidor local se detuvo" in str(exc)


def test_wait_for_server_returns_once_the_port_accepts_connections():
    port = TEST_PORT_BASE + 9
    ready = threading.Event()

    def serve():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST, port))
        sock.listen(1)
        ready.set()
        time.sleep(2)
        sock.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    ready.wait(timeout=2)
    # No exception is the assertion here -- a failure would raise RuntimeError.
    launcher.wait_for_server(HOST, port, thread, timeout=3.0)


def test_show_startup_failure_puts_the_last_error_line_and_log_path_in_the_dialog(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ctypes.windll.user32,
        "MessageBoxW",
        lambda hwnd, text, caption, flags: calls.append((text, caption, flags)),
    )

    traceback_text = (
        "Traceback (most recent call last):\n"
        '  File "desktop_launcher.py", line 1, in main\n'
        "    raise RuntimeError('El servidor local no respondió a tiempo.')\n"
        "RuntimeError: El servidor local no respondió a tiempo."
    )
    launcher._show_startup_failure(traceback_text)

    assert len(calls) == 1
    text, caption, flags = calls[0]
    assert "El servidor local no respondió a tiempo." in text
    assert str(launcher._log_path()) in text
    assert "RAGPoC" in caption
    MB_ICONERROR = 0x10
    assert flags & MB_ICONERROR


def test_show_startup_failure_never_raises_even_if_the_dialog_itself_fails(monkeypatch):
    def boom(*a):
        raise OSError("no user32 available")

    monkeypatch.setattr(ctypes.windll.user32, "MessageBoxW", boom)
    # A broken dialog must never mask (or re-raise as) the original startup error.
    launcher._show_startup_failure("something failed")
