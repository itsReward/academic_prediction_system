from functools import wraps
from flask import request, g, current_app, abort, redirect, url_for, flash
from flask_login import current_user
import time
import json
import logging
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)

class RequestMiddleware:
    """
    Middleware for processing requests.
    Performs tasks like logging, timing, and security checks.
    """
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize middleware with Flask app"""
        
        # Register before_request handlers
        app.before_request(self.log_request_info)
        app.before_request(self.start_timer)
        app.before_request(self.check_maintenance_mode)
        app.before_request(self.validate_content_type)
        
        # Register after_request handlers
        app.after_request(self.add_security_headers)
        app.after_request(self.log_response_info)
        
        # Register teardown handlers
        app.teardown_request(self.teardown_request)
    
    def log_request_info(self):
        """Log request information"""
        # Skip logging for static files
        if request.path.startswith('/static/'):
            return
        
        # Log the request
        logger.info(f"Request: {request.method} {request.path} - User: {current_user.email if not current_user.is_anonymous else 'Anonymous'} - IP: {request.remote_addr}")
        
        # Log request parameters for debugging (excluding passwords)
        if current_app.debug:
            # Filter out sensitive data
            filtered_form = {}
            if request.form:
                for key, value in request.form.items():
                    if 'password' in key.lower() or 'token' in key.lower():
                        filtered_form[key] = '******'
                    else:
                        filtered_form[key] = value
            
            logger.debug(f"Request Args: {request.args}, Form: {filtered_form}")
    
    def start_timer(self):
        """Start a timer for request processing"""
        g.start_time = time.time()
    
    def check_maintenance_mode(self):
        """Check if the application is in maintenance mode"""
        if current_app.config.get('MAINTENANCE_MODE', False):
            # Allow admins to access the application during maintenance
            if not current_user.is_authenticated or not current_user.is_admin():
                return redirect(url_for('web.maintenance'))
    
    def validate_content_type(self):
        """Validate content type for POST/PUT/PATCH requests"""
        if request.method in ['POST', 'PUT', 'PATCH'] and request.is_json:
            try:
                # Attempt to parse JSON
                _ = request.get_json()
            except Exception as e:
                logger.warning(f"Invalid JSON in request: {str(e)}")
                return abort(400, description="Invalid JSON format")
    
    def add_security_headers(self, response):
        """Add security headers to response"""
        # Content Security Policy
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; img-src 'self' data:; font-src 'self' https://cdnjs.cloudflare.com;"
        
        # Prevent browsers from MIME-sniffing
        response.headers['X-Content-Type-Options'] = 'nosniff'
        
        # Prevent clickjacking
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        
        # Enable browser XSS protection
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Referrer policy
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        return response
    
    def log_response_info(self, response):
        """Log response information and timing"""
        # Skip logging for static files
        if request.path.startswith('/static/'):
            return response
        
        # Calculate request duration
        duration = time.time() - g.get('start_time', time.time())
        
        # Log response info
        logger.info(f"Response: {request.method} {request.path} - Status: {response.status_code} - Duration: {duration:.4f}s")
        
        # Record response time in metrics
        if hasattr(g, 'metrics'):
            g.metrics['response_time'] = duration
        
        return response
    
    def teardown_request(self, exception):
        """Clean up after request is processed"""
        # Log exceptions
        if exception:
            logger.error(f"Request error: {str(exception)}")
        
        # Clean up any resources
        if hasattr(g, 'start_time'):
            delattr(g, 'start_time')

class UserActivityMiddleware:
    """
    Middleware for tracking user activity.
    Records page views, button clicks, and other user actions.
    """
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize middleware with Flask app"""
        
        # Register before_request handlers
        app.before_request(self.track_page_view)
        
        # Register after_request handlers
        app.after_request(self.track_api_usage)
    
    def track_page_view(self):
        """Track page views"""
        # Skip for API endpoints, static files, and non-GET requests
        if (request.path.startswith('/api/') or 
            request.path.startswith('/static/') or
            request.method != 'GET'):
            return
        
        # Skip for AJAX requests
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return
        
        # Record the page view
        if current_user.is_authenticated:
            try:
                # In production, save to database:
                # page_view = PageView(
                #     user_id=current_user.id,
                #     url=request.path,
                #     method=request.method,
                #     timestamp=datetime.utcnow()
                # )
                # db.session.add(page_view)
                # db.session.commit()
                
                # For debug, just log
                logger.debug(f"Page view: {request.path} by user {current_user.id}")
            except Exception as e:
                logger.error(f"Failed to record page view: {str(e)}")
    
    def track_api_usage(self, response):
        """Track API usage"""
        # Only track API endpoints with successful responses
        if request.path.startswith('/api/') and 200 <= response.status_code < 300:
            try:
                # In production, save to database:
                # api_call = APICall(
                #     user_id=current_user.id if current_user.is_authenticated else None,
                #     endpoint=request.path,
                #     method=request.method,
                #     status_code=response.status_code,
                #     timestamp=datetime.utcnow()
                # )
                # db.session.add(api_call)
                # db.session.commit()
                
                # For debug, just log
                user_id = current_user.id if current_user.is_authenticated else 'Anonymous'
                logger.debug(f"API call: {request.method} {request.path} by user {user_id} - Status: {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to record API call: {str(e)}")
        
        return response

def validate_json_middleware():
    """Middleware to validate JSON content"""
    if request.is_json:
        try:
            _ = request.get_json()
        except Exception as e:
            logger.warning(f"Invalid JSON in request: {str(e)}")
            abort(400, description="Invalid JSON format")

def inject_system_stats():
    """Middleware to inject system stats into every template"""
    # In production, fetch real stats
    # from src.app.models.system_stats import get_current_stats
    # return get_current_stats()
    
    # For development
    return {
        'active_users': 150,
        'system_version': '1.0.0',
        'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def rate_limiter(max_requests=100, window=60):
    """
    Decorator to limit request rate
    
    Args:
        max_requests (int): Maximum number of requests per window
        window (int): Time window in seconds
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Skip rate limiting for development environment
            if current_app.config.get('ENVIRONMENT') == 'development':
                return f(*args, **kwargs)
            
            # Get client IP
            client_ip = request.remote_addr
            
            # In production, use Redis or similar to track request rates
            # from src.app.utils.rate_limit import is_rate_limited
            # if is_rate_limited(client_ip, max_requests, window):
            #     abort(429, description="Too many requests")
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def init_middleware(app):
    """Initialize all middleware"""
    # Initialize request middleware
    request_middleware = RequestMiddleware()
    request_middleware.init_app(app)
    
    # Initialize user activity middleware
    user_activity_middleware = UserActivityMiddleware()
    user_activity_middleware.init_app(app)
    
    # Register template context processor
    app.context_processor(lambda: {'system_stats': inject_system_stats()})