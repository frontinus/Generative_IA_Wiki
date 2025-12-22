from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import logging
import traceback
import re
from functools import wraps
from typing import Dict, Any, Tuple, Optional, List

# Import the RAG Pipeline class
from pipeline import RAGPipeline

# ===== CONFIGURATION =====
class Config:
    MAX_QUERY_LENGTH = 500
    DEFAULT_TOP_K = 5
    MAX_TOP_K = 20
    CORS_ORIGINS = ["*"]  # In production, restrict this to specific domains

# ===== APP SETUP =====
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": Config.CORS_ORIGINS}})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize RAG Pipeline
# We initialize it here so it's loaded when the app starts
try:
    rag_pipeline = RAGPipeline()
    logger.info("RAG Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG Pipeline: {e}")
    rag_pipeline = None

# ===== HELPER FUNCTIONS =====

def create_success_response(data: Any, status_code: int = 200):
    """Standardized success response."""
    return jsonify({
        "status": "success",
        **data
    }), status_code

def create_error_response(message: str, status_code: int = 400, details: Optional[str] = None):
    """Standardized error response."""
    response = {
        "status": "error",
        "error": message
    }
    if details:
        response["details"] = details
    return jsonify(response), status_code

def validate_query(query: str) -> Tuple[bool, Optional[str]]:
    """Validate user query."""
    if not query:
        return False, "Query cannot be empty"
    if len(query) > Config.MAX_QUERY_LENGTH:
        return False, f"Query too long (max {Config.MAX_QUERY_LENGTH} characters)"
    return True, None

def validate_top_k(k: Any) -> Tuple[bool, Optional[str], int]:
    """Validate top_k parameter."""
    try:
        k_int = int(k)
        if k_int < 1:
            return False, "top_k must be at least 1", Config.DEFAULT_TOP_K
        if k_int > Config.MAX_TOP_K:
            return False, f"top_k must be at most {Config.MAX_TOP_K}", Config.MAX_TOP_K
        return True, None, k_int
    except (ValueError, TypeError):
        return False, "top_k must be an integer", Config.DEFAULT_TOP_K

def clean_html_response(html_content: str) -> str:
    """Clean up the HTML response from the LLM."""
    if not html_content:
        return ""
    
    # Remove markdown code blocks if present (despite instructions)
    cleaned = re.sub(r'```html\s*', '', html_content)
    cleaned = re.sub(r'```\s*', '', cleaned)
    
    # Remove any "html" prefix that might have slipped in
    if cleaned.lower().startswith('html'):
        cleaned = cleaned[4:].strip()
        
    return cleaned.strip()

# ===== DECORATORS =====

def log_request(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        logger.info(f"Request: {request.method} {request.path}")
        return f(*args, **kwargs)
    return decorated_function

def validate_json(required_fields: List[str] = None):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return create_error_response("Content-Type must be application/json", 415)
            
            data = request.get_json()
            if required_fields:
                missing = [field for field in required_fields if field not in data]
                if missing:
                    return create_error_response(f"Missing required fields: {', '.join(missing)}", 400)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ===== ROUTES =====

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    status = "healthy" if rag_pipeline else "degraded"
    return create_success_response({
        "status": status,
        "pipeline_initialized": rag_pipeline is not None
    })

@app.route('/generate/', methods=['POST'])
@log_request
@validate_json(['query'])
def generate():
    """Generate answer for a query."""
    if not rag_pipeline:
        return create_error_response("RAG pipeline is not initialized", 500)
        
    try:
        data = request.get_json()
        query = data.get("query", "").strip()
        
        # Validate query
        is_valid, error_msg = validate_query(query)
        if not is_valid:
            return create_error_response(error_msg, 400)
            
        # Get optional parameters
        use_openai = data.get("use_openai", False)
        if not isinstance(use_openai, bool):
            return create_error_response("use_openai must be a boolean", 400)
            
        top_k = data.get("top_k", Config.DEFAULT_TOP_K)
        is_valid, error_msg, top_k = validate_top_k(top_k)
        if not is_valid:
            logger.warning(f"Invalid top_k value, using default: {error_msg}")
            
        logger.info(f"Processing query: '{query[:100]}...' (use_openai={use_openai}, top_k={top_k})")
        
        # Run pipeline
        answer = rag_pipeline.run(query, use_openai=use_openai, k=top_k)
        
        # Clean response
        answer = clean_html_response(answer)
        
        return create_success_response({
            "query": query,
            "answer": answer,
            "backend": "openai" if use_openai else "ollama",
            "top_k": top_k
        })
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return create_error_response(str(e), 400)
    except RuntimeError as e:
        logger.error(f"Model error: {e}")
        return create_error_response(
            "Failed to generate answer", 
            500, 
            details=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return create_error_response(
            "Internal server error", 
            500, 
            details=str(e) if app.debug else None
        )

@app.route('/stats', methods=['GET'])
def stats():
    """Get RAG system statistics."""
    if not rag_pipeline:
        return create_error_response("RAG pipeline is not initialized", 500)
    
    return create_success_response(rag_pipeline.get_stats())

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