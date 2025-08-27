from flask import Flask, render_template, request, jsonify
from utils.graph import TranslateState, translate_graph

app = Flask(__name__, template_folder='templates', static_folder='static')

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json(silent=True)
    message = data.get("message", "").strip() if data else ""
    state = TranslateState(message=message)

    if not message:
        return jsonify({"error": "No text provided"}), 400
    
    result_state = translate_graph.invoke(state)

    response = {
        "message":result_state.get("message"),
        "is_translate_msg":result_state.get("is_translate_msg"),
        "language":result_state.get("language"),
        "extracted_msg":result_state.get("extracted_msg"),
        "translation":result_state.get("translation"),
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(port=5001, debug=True)