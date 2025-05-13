from flask import Flask, render_template, request
import joblib, numpy as np

app = Flask(__name__)

# Load models and preprocessing artifacts
rf_model  = joblib.load('rf_model.pkl')
gb_model  = joblib.load('gb_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')
scaler    = joblib.load('scaler.pkl')

# Load LabelEncoders
family_encoder   = joblib.load('family_history_encoder.pkl')
meds_encoder     = joblib.load('current_medication_encoder.pkl')
history_encoder  = joblib.load('medical_history_encoder.pkl')

def safe_encode(encoder, label):
    if label in encoder.classes_:
        return encoder.transform([label])[0]
    # unseen: append and re-sort classes
    encoder.classes_ = np.append(encoder.classes_, label)
    return encoder.transform([label])[0]

# Insight & utility functions (unchanged)
def generate_health_insights(age, sex, bmi, smoker):
    insights=[]
    if bmi>30: insights.append("BMI indicates overweight. Consider diet & exercise.")
    elif bmi<18.5: insights.append("BMI indicates underweight. Ensure good nutrition.")
    else: insights.append("BMI is normal. Keep up the healthy lifestyle.")
    insights.append("Quit smoking to lower costs.") if smoker else insights.append("Non-smoker — great for cost & health.")
    return insights

def calculate_health_risk(bmi, smoker):
    if bmi>=30 or smoker: return "High Risk"
    if bmi>=25: return "Moderate Risk"
    return "Low Risk"

def compare_cost_by_region(cost, region_code):
    factors={0:1.2,1:1.1,2:1.0,3:1.15}
    return cost * factors.get(region_code,1)

def calculate_savings(cost, bmi_change=False, smoker_change=False):
    if bmi_change:   cost*=0.9
    if smoker_change: cost*=0.8
    return cost

def suggest_insurance_plan(cost):
    if cost<2000: return "Basic Plan: Low Coverage"
    if cost<5000: return "Standard Plan: Moderate Coverage"
    return "Premium Plan: Comprehensive Coverage"

@app.route("/", methods=["GET","POST"])
def home():
    if request.method=="POST":
        # 1. Collect & encode inputs
        age   = int(request.form['age'])
        sex   = 0 if request.form['sex']=="male" else 1
        bmi   = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker   = 1 if request.form['smoker']=="yes" else 0
        region_code = {"northeast":0,"northwest":1,"southeast":2,"southwest":3}[request.form['region']]

        sleep_quality  = int(request.form['sleep_quality'])
        income_level   = float(request.form['income_level'])
        diet_quality   = int(request.form['diet_quality'])
        activity_code  = {"low":0,"moderate":1,"high":2}[request.form['activity_level']]

        fh = request.form['family_history'].strip().lower()
        cm = request.form['current_medication'].strip().lower()
        mh = request.form['medical_history'].strip().lower()

        fh_code = safe_encode(family_encoder, fh) if fh else family_encoder.transform(['none'])[0]
        cm_code = safe_encode(meds_encoder,   cm) if cm else meds_encoder.transform(['none'])[0]
        mh_code = safe_encode(history_encoder,mh) if mh else history_encoder.transform(['none'])[0]

        # 2. Build feature vector & scale
        X = np.array([[age, sex, bmi, children, smoker, region_code,
                       sleep_quality, income_level, diet_quality, activity_code,
                       fh_code, cm_code, mh_code]])
        Xs = scaler.transform(X)

        # 3. Predict & ensemble
        p1 = rf_model .predict(Xs)
        p2 = gb_model .predict(Xs)
        p3 = xgb_model.predict(Xs)
        final = float((p1 + p2 + p3) / 3)

        # 4. Generate dynamic outputs
        insights = generate_health_insights(age, sex, bmi, smoker)
        risk     = calculate_health_risk(bmi, smoker)
        region_c = compare_cost_by_region(final, region_code)
        saving   = calculate_savings(final, bmi_change=True, smoker_change=True)
        plan     = suggest_insurance_plan(final)

        return render_template("index.html",
            final_prediction=round(final,2),
            insurance_plan=plan,
            health_risk=risk,
            region_cost_comparison=round(region_c,2),
            savings=round(saving,2),
            health_insights=insights
        )
    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)
