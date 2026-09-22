"""Allow the app's own HTTPS host, not arbitrary cross-origin browser pages."""
from urllib.parse import urlsplit


def allowed_origin(origin, host, extra_origins=''):
    if origin is None:  # Native test clients do not send Origin.
        return True
    if origin in {'http://127.0.0.1:8765', 'http://localhost:8765'}:
        return True
    if origin in {s.strip() for s in extra_origins.split(',') if s.strip()}:
        return True
    try:
        parsed = urlsplit(origin)
        return (parsed.scheme == 'https' and bool(parsed.hostname)
                and parsed.netloc.lower() == (host or '').lower()
                and not parsed.username and not parsed.password
                and not parsed.path and not parsed.query and not parsed.fragment)
    except ValueError:
        return False
