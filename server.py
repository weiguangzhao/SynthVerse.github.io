"""
Simple threaded HTTP server for local development.
Usage: python server.py
Then open http://localhost:8000
"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


if __name__ == "__main__":
    port = 8000
    server = ThreadingHTTPServer(("", port), SimpleHTTPRequestHandler)
    print(f"Serving at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
