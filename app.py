from flask import Flask, jsonify, request
from flask_cors import CORS
from pipeline import rag, rag_openai

app = Flask(__name__)
CORS(app)

# Simple health check route
@app.route('/generate/', methods=['POST'])
def health():

    data = request.get_json()
    prompt = data["query"]
    use_openai = data.get("use_openai", False)
    if use_openai:
        res = rag_openai(prompt)
    else:
        res = rag(prompt)
    print(res)
    return jsonify({"answer":res,"query":prompt}), 200

if __name__ == '__main__':
    # Run the Flask application
    app.run(host='127.0.0.1', port=8000, debug=True)
