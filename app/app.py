from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__, template_folder="../templates")
model = joblib.load("model/student_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analytics")
def analytics():
    # Example: Aggregate from your dataset or predictions
    data = {
        "low": 20,      # number of low-risk students
        "medium": 15,   # medium-risk
        "high": 5       # high-risk
    }
    return jsonify(data)


# suggestion function
def get_suggestions(attendance, marks, study_hours):
    
    suggestions = []
    
    if attendance < 60:
        suggestions.append("Improve attendance")
        
    if marks < 50:
        suggestions.append("Attend extra classes")
        
    if study_hours < 2:
        suggestions.append("Increase daily study time")
        
    if not suggestions:
        suggestions.append("Keep up the good work")
        
    return suggestions


@app.route("/predict", methods=["POST"])
def predict():
    
    data = request.json
    
    attendance = data["attendance"]
    marks = data["internal_marks"]
    assignment = data["assignment_score"]
    study_hours = data["study_hours"]
    gpa = data["previous_gpa"]
    
    features = [[attendance, marks, assignment, study_hours, gpa]]
    
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    
    # call suggestion function
    suggestions = get_suggestions(attendance, marks, study_hours)
    
    return jsonify({
        "risk_level": prediction[0],
        "confidence": float(max(probability[0])),
        "suggestions": suggestions
    })


if __name__ == "__main__":
    app.run(debug=True)