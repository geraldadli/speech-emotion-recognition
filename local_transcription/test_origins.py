import unittest
from origins import allowed_origin


class OriginTests(unittest.TestCase):
    def test_local_and_native(self):
        self.assertTrue(allowed_origin(None, 'localhost:8765'))
        self.assertTrue(allowed_origin('http://127.0.0.1:8765', '127.0.0.1:8765'))

    def test_https_tunnel_same_host(self):
        self.assertTrue(allowed_origin('https://example.ngrok-free.dev', 'example.ngrok-free.dev'))
        self.assertFalse(allowed_origin('https://other.ngrok-free.dev', 'example.ngrok-free.dev'))
        self.assertFalse(allowed_origin('http://example.ngrok-free.dev', 'example.ngrok-free.dev'))

    def test_invalid_origin(self):
        for origin in ['null', 'https://example.com/path', 'https://user@example.com', 'https://[']:
            self.assertFalse(allowed_origin(origin, 'example.com'))

    def test_explicit_proxy_override(self):
        self.assertTrue(allowed_origin('https://example.com', 'localhost:8765', 'https://example.com'))
        self.assertFalse(allowed_origin('https://other.com', 'localhost:8765', 'https://example.com'))
