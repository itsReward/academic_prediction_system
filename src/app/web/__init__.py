from flask import Blueprint

# Create a Blueprint for web routes
web = Blueprint('web', __name__, template_folder='../templates', static_folder='../static')

# Import routes after Blueprint creation to avoid circular imports
from src.app.web import routes

def init_app(app):
    """
    Initialize the web module with the Flask application.
    Register the Blueprint and any other configurations.
    
    Args:
        app (Flask): The Flask application instance
    """
    # Register the web blueprint
    app.register_blueprint(web, url_prefix='/')
    
    # Configure template filters if needed
    from src.app.web.filters import init_filters
    init_filters(app)
    
    # Register error handlers
    from src.app.web.errors import init_error_handlers
    init_error_handlers(app)
    
    # Register context processors
    @app.context_processor
    def utility_processor():
        """
        Add utility functions to template context
        """
        def format_date(date, format='%B %d, %Y'):
            """Format a date to a readable string"""
            if date:
                return date.strftime(format)
            return ''
            
        def now(format='%Y'):
            """Return the current year (used in footer)"""
            from datetime import datetime
            return datetime.now().strftime(format)
            
        return dict(format_date=format_date, now=now)