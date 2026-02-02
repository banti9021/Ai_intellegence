from flask import Flask, request, jsonify
from pi import AIPipeline

app = Flask(__name__)

pipeline = None   # 🔑 lazy global


@app.route("/")
def home():
    return {"status": "AI Customer Intelligence API is running"}


# ------------------------------
# Main AI Endpoint
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    global pipeline

    if pipeline is None:
        print("🚀 Loading AI Pipeline...")
        pipeline = AIPipeline()
        print("✅ Pipeline Loaded")

    data = request.json
    if not data or "query" not in data:
        return jsonify({"error": "Query missing"}), 400

    query = data["query"]
    result = pipeline.run(query)

    return jsonify(result)


# ------------------------------
# Churn Prediction API
# ------------------------------
@app.route("/churn", methods=["POST"])
def churn():
    global pipeline

    if pipeline is None:
        pipeline = AIPipeline()

    customer_data = request.json
    result = pipeline.churn_agent.predict(customer_data)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
