import os
import tempfile
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from main_graph import legal_graph, profile_graph

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze_legal', methods=['POST'])
def analyze_legal():
    raw_text = request.form.get('raw_text', '').strip()
    inputs = {}
    temp_path = None

    inputs["legal_category"] = request.form.get('legal_category', 'General')
    inputs["case_status"] = request.form.get('case_status', 'Need Information')

    if raw_text:
        inputs["raw_text"] = raw_text
        inputs["file_path"] = None
    elif 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        if not allowed_file(file.filename):
            return jsonify({"success": False, "error": "Invalid file type. Please upload a PDF, TXT, JPG, or PNG."}), 400

        filename = secure_filename(file.filename)
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(temp_path)
        inputs["file_path"] = temp_path
        inputs["raw_text"] = None
    else:
        return jsonify({"success": False, "error": "No file or text provided."}), 400

    try:
        final_state = legal_graph.invoke(inputs)
        return jsonify({
            "success": True,
            "solution": final_state.get("legal_solution", "Analysis failed."),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/api/analyze_profile', methods=['POST'])
def analyze_profile():
    inputs = {
        "user_age": request.form.get('user_age', ''),
        "user_state": request.form.get('user_state', ''),
        "user_education": request.form.get('user_education', ''),
        "user_specialization": request.form.get('user_specialization', ''),
        "user_gender": request.form.get('user_gender', ''),
        "annual_income": request.form.get('annual_income', ''),
        "current_status": request.form.get('current_status', ''),
        "extracurriculars": request.form.get('extracurriculars', ''),
        "user_goal": request.form.get('user_goal', ''),
    }

    try:
        final_state = profile_graph.invoke(inputs)
        return jsonify({
            "success": True,
            "opportunities": final_state.get("opportunities_json", []),
            "missed_opportunities": final_state.get("missed_opportunities_json", []),
            "roadmap": final_state.get("guiding_roadmap", ""),
            "report": final_state.get("insight_report", "Report generation failed."),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)