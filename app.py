from flask import Flask, jsonify, request
from flask_cors import CORS
from pipeline import rag
import logging
import re
from datetime import datetime, timezone
from functools import wraps
import traceback
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Configuration
class Config:
    MAX_QUERY_LENGTH = 500
    DEFAULT_TOP_K = 5
    MAX_TOP_K = 20
    REQUEST_TIMEOUT = 60  # seconds


# ===== MIDDLEWARE / DECORATORS =====
def log_request(f):
    """Decorator to log all requests and responses."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Log request
        request_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'method': request.method,
            'path': request.path,
            'remote_addr': request.remote_addr,
            'data': request.get_json(silent=True)
        }
        logger.info(f"Incoming request: {request_data}")
        
        try:
            # Execute the route function
            response = f(*args, **kwargs)
            logger.info(f"Request successful: {request.path}")
            return response
        except Exception as e:
            logger.error(f"Request failed: {request.path} - {str(e)}")
            raise
    
    return decorated_function


def validate_json(required_fields: list):
    """Decorator to validate JSON request body."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Check if request has JSON
            if not request.is_json:
                return jsonify({
                    "error": "Content-Type must be application/json",
                    "status": "error"
                }), 400
            
            data = request.get_json()
            
            # Check if data exists
            if not data:
                return jsonify({
                    "error": "Request body cannot be empty",
                    "status": "error"
                }), 400
            
            # Check required fields
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return jsonify({
                    "error": f"Missing required fields: {', '.join(missing_fields)}",
                    "status": "error"
                }), 400
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


# ===== HELPER FUNCTIONS =====
def validate_query(query: str) -> Tuple[bool, str]:
    """
    Validate user query.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query:
        return False, "Query cannot be empty"
    
    query = query.strip()
    
    if not query:
        return False, "Query cannot be only whitespace"
    
    if len(query) > Config.MAX_QUERY_LENGTH:
        return False, f"Query too long (max {Config.MAX_QUERY_LENGTH} characters)"
    
    return True, ""


def validate_top_k(top_k: Any) -> Tuple[bool, str, int]:
    """
    Validate top_k parameter.
    
    Returns:
        Tuple of (is_valid, error_message, sanitized_value)
    """
    try:
        k = int(top_k)
    except (ValueError, TypeError):
        return False, "top_k must be an integer", Config.DEFAULT_TOP_K
    
    if k < 1:
        return False, "top_k must be at least 1", Config.DEFAULT_TOP_K
    
    if k > Config.MAX_TOP_K:
        return False, f"top_k cannot exceed {Config.MAX_TOP_K}", Config.MAX_TOP_K
    
    return True, "", k


def clean_html_response(html_string: str) -> str:
    """
    Clean HTML response from LLM by removing markdown code blocks and prefixes.
    
    Args:
        html_string: Raw HTML string from LLM
    
    Returns:
        Cleaned HTML string
    """
    # Remove markdown code blocks (```html ... ``` or ```...```)
    html_string = re.sub(r'^```html\s*\n?', '', html_string, flags=re.IGNORECASE)
    html_string = re.sub(r'^```\s*\n?', '', html_string)
    html_string = re.sub(r'\n?```$', '', html_string)
    
    # Remove standalone "html" at the start (case insensitive)
    html_string = re.sub(r'^\s*[`\'"]?html[`\'"]?\s*\n?', '', html_string, flags=re.IGNORECASE)
    
    # Remove any leading/trailing whitespace
    html_string = html_string.strip()
    
    return html_string


def create_success_response(data: Dict[str, Any], status_code: int = 200) -> Tuple[Dict, int]:
    """Create standardized success response."""
    response = {
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **data
    }
    return jsonify(response), status_code


def create_error_response(error: str, status_code: int = 400, details: str = None) -> Tuple[Dict, int]:
    """Create standardized error response."""
    response = {
        "status": "error",
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    if details:
        response["details"] = details
    
    return jsonify(response), status_code


# ===== ROUTES =====
@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    
    Returns:
        200: Service is healthy
    """
    return create_success_response({
        "message": "RAG API is running",
        "version": "1.0.0"
    })


@app.route('/generate/', methods=['POST'])
@log_request
@validate_json(['query'])
def generate():
    """
    Generate answer using RAG pipeline.
    
    Request Body:
        {
            "query": str (required) - User's question,
            "use_openai": bool (optional) - Use OpenAI instead of Ollama,
            "top_k": int (optional) - Number of documents to retrieve (default: 5)
        }
    
    Returns:
        200: Success with answer
        400: Bad request (validation error)
        500: Internal server error
    """
    try:
        data = request.get_json()
        
        # Extract and validate query
        query = data.get("query", "").strip()
        is_valid, error_msg = validate_query(query)
        if not is_valid:
            return create_error_response(error_msg, 400)
        
        # Extract and validate use_openai
        use_openai = data.get("use_openai", False)
        if not isinstance(use_openai, bool):
            return create_error_response("use_openai must be a boolean", 400)
        
        # Extract and validate top_k
        top_k = data.get("top_k", Config.DEFAULT_TOP_K)
        is_valid, error_msg, top_k = validate_top_k(top_k)
        if not is_valid:
            logger.warning(f"Invalid top_k value, using default: {error_msg}")
            # Don't return error, just use default
        
        # Log the request
        logger.info(f"Processing query: '{query[:100]}...' (use_openai={use_openai}, top_k={top_k})")
        
        # Call RAG pipeline
        answer = rag(query, use_openai=use_openai, k=top_k)
        
        # Clean the answer - remove markdown code blocks and "html" prefix
        answer = clean_html_response(answer)
        
        # Return success response
        return create_success_response({
            "query": query,
            "answer": answer,
            "backend": "openai" if use_openai else "ollama",
            "top_k": top_k
        })
    
    except ValueError as e:
        # Handle validation errors from pipeline
        logger.error(f"Validation error: {e}")
        return create_error_response(str(e), 400)
    
    except RuntimeError as e:
        # Handle model errors from pipeline
        logger.error(f"Model error: {e}")
        return create_error_response(
            "Failed to generate answer",
            500,
            details=str(e)
        )
    
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return create_error_response(
            "Internal server error",
            500,
            details=str(e) if app.debug else None
        )


@app.route('/query', methods=['POST'])
@log_request
@validate_json(['query'])
def query():
    """
    Alias for /generate/ endpoint (for backward compatibility).
    """
    return generate()


@app.route('/stats', methods=['GET'])
def stats():
    """
    Get API statistics.
    
    Returns:
        200: Statistics about the RAG system
    """
    from pipeline import DF, EVENT_EMBEDDINGS, INDEX
    
    return create_success_response({
        "total_documents": len(DF),
        "embedding_dimensions": EVENT_EMBEDDINGS.shape[1],
        "index_size": INDEX.ntotal,
        "model": "all-MiniLM-L6-v2",
        "backends": ["ollama (phi3:mini)", "openai (gpt-4-turbo)"]
    })


# ===== ERROR HANDLERS =====
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return create_error_response(
        "Endpoint not found",
        404,
        details=f"The requested URL {request.path} was not found"
    )


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return create_error_response(
        "Method not allowed",
        405,
        details=f"Method {request.method} is not allowed for {request.path}"
    )


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    logger.error(traceback.format_exc())
    return create_error_response(
        "Internal server error",
        500,
        details=str(error) if app.debug else None
    )


# ===== STARTUP =====
@app.before_request
def before_request():
    """Log all incoming requests."""
    logger.debug(f"{request.method} {request.path} from {request.remote_addr}")


if __name__ == '__main__':
    logger.info("="*60)
    logger.info("Starting RAG API Server")
    logger.info("="*60)
    logger.info(f"Max query length: {Config.MAX_QUERY_LENGTH}")
    logger.info(f"Default top_k: {Config.DEFAULT_TOP_K}")
    logger.info(f"Max top_k: {Config.MAX_TOP_K}")
    logger.info("="*60)
    
    # Run the Flask application
    app.run(
        host='127.0.0.1',
        port=8000,
        debug=True,  # Set to False in production
        threaded=True  # Handle multiple requests concurrently
    )