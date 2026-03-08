from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

API_BASE = "http://localhost:8000"


@app.route("/")
def home():
    return render_template("index.html")


# POST query
@app.route("/query", methods=["POST"])
def query():

    try:
        data = request.json

        response = requests.post(
            f"{API_BASE}/query",
            json=data,
            headers={"Content-Type": "application/json"}
        )

        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": str(e)})


# GET cache stats
@app.route("/cache/stats", methods=["GET"])
def cache_stats():

    try:
        response = requests.get(f"{API_BASE}/cache/stats")
        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": str(e)})


# DELETE cache
@app.route("/cache", methods=["DELETE"])
def flush_cache():

    try:
        response = requests.delete(f"{API_BASE}/cache")
        return jsonify(response.json())

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)