"""Serve the offline N06 viewer on loopback; no training or external services."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import webbrowser


def main() -> None:
    """Launch the checked-in deterministic viewer; Ctrl+C stops the server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / 'docs'
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    url = f'http://127.0.0.1:{args.port}/vm_n06_demo/'
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    print(f'N06 offline viewer: {url}', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
