from __future__ import annotations

import ssl

import httpx
import truststore

# Verify TLS certs against the OS trust store instead of certifi's bundled CA
# list. Needed behind a TLS-inspecting corporate proxy/antivirus (Zscaler,
# Netskope, Kaspersky, ...): the OS store already trusts the proxy's injected
# root CA (that's why browsers work fine there), certifi's bundle doesn't.
SSL_CONTEXT = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def new_async_client(**kwargs) -> httpx.AsyncClient:
    kwargs.setdefault("verify", SSL_CONTEXT)
    return httpx.AsyncClient(**kwargs)


def get(url: str, **kwargs) -> httpx.Response:
    kwargs.setdefault("verify", SSL_CONTEXT)
    return httpx.get(url, **kwargs)
