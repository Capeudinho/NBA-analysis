"""NBA analysis web report (Flask + Jinja)."""

from flask import Flask

from .formatting import register_filters


def create_app():
    app = Flask(__name__)
    app.config["JSON_AS_ASCII"] = False
    register_filters(app)

    from . import config
    from .routes import bp

    app.register_blueprint(bp)

    @app.context_processor
    def inject_globals():
        return {
            "TEAMS": config.TEAMS,
            "TEAM_CODES": config.TEAM_CODES,
            "VARIABLES": config.ANALYSIS_VARIABLES,
            "VARIABLE_KEYS": config.VARIABLE_KEYS,
            "DECIMALS": config.DECIMALS,
        }

    return app
