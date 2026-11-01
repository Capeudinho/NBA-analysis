"""Entry point: `python app.py` starts the report at http://127.0.0.1:5000."""

import os

from nba_analysis import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
