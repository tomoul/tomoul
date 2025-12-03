#!/usr/bin/env python3
"""
Silero VAD WebAssembly Demo Server

Serves the example demo and zig-out/ directory for WebAssembly development.
Adds proper CORS and MIME type headers for Wasm files.

Usage:
    python3 examples/silero-vad/server.py [port]

Default port is 8000.
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path


class TomoulHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler with Wasm MIME type and CORS support."""

    def __init__(self, *args, directory=None, **kwargs):
        # Set up the root directory
        self.root_directory = directory or os.getcwd()
        super().__init__(*args, directory=directory, **kwargs)

    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        # Required for SharedArrayBuffer (if needed later)
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def guess_type(self, path):
        """Add .wasm MIME type."""
        if path.endswith('.wasm'):
            return 'application/wasm'
        return super().guess_type(path)

    def translate_path(self, path):
        """Route requests to appropriate directories."""
        # Default behavior for SimpleHTTPRequestHandler
        result = super().translate_path(path)

        # Check if the file exists, if not try examples/silero-vad/ prefix
        if not os.path.exists(result):
            example_path = os.path.join(self.root_directory, 'examples/silero-vad', path.lstrip('/'))
            if os.path.exists(example_path):
                return example_path

        return result


def run_server(port=8000):
    """Start the development server."""
    # Find project root (examples/silero-vad/ -> examples/ -> project root)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    os.chdir(project_root)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Silero VAD WebAssembly Demo Server                 ║
╠══════════════════════════════════════════════════════════════╣
║  Serving:                                                    ║
║    - examples/silero-vad/  → http://localhost:{port}/        ║
║    - zig-out/              → http://localhost:{port}/zig-out/║
╠══════════════════════════════════════════════════════════════╣
║  Open http://localhost:{port}/ in your browser               ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Serve from project root, with www/ as fallback for index
    handler = lambda *args, **kwargs: TomoulHandler(*args, directory=str(project_root), **kwargs)

    with socketserver.TCPServer(("", port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
