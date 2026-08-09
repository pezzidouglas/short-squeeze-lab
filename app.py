"""
WSGI entry point for the dashboard API.

    python app.py                      # development server
    gunicorn app:app                   # production

The application object is built by `squeezelab.api.create_app`, which wires the
source fleet, the scan cache and every endpoint.
"""

import os

from squeezelab.api import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
