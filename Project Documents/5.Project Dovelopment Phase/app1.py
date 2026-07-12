import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model and scaler
# model = pickle.load(open("rdf.pkl", "rb"))
# scale = pickle.load(open("scale1.pkl", "rb"))
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "rdf.pkl")
scale_path = os.path.join(BASE_DIR, "scale1.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scale_path, "rb") as f:
    scale = pickle.load(f)

@app.route("/")
def home():
    return render_template("home.html")


# @app.route("/predict", methods=["GET"])
# def predict():
#     return render_template("input.html")


@app.route("/predict", methods=["POST"])
def submit():

    try:
        gender = int(request.form["Gender"])
        married = int(request.form["Married"])
        dependents = int(request.form["Dependents"])
        education = int(request.form["Education"])
        self_employed = int(request.form["Self_Employed"])
        applicant_income = float(request.form["ApplicantIncome"])
        coapplicant_income = float(request.form["CoapplicantIncome"])
        loan_amount = float(request.form["LoanAmount"])
        loan_amount_term = float(request.form["Loan_Amount_Term"])
        credit_history = int(request.form["Credit_History"])
        property_area = int(request.form["Property_Area"])

        input_data = [[
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_amount_term,
            credit_history,
            property_area
        ]]

        columns = [
            'Gender',
            'Married',
            'Dependents',
            'Education',
            'Self_Employed',
            'ApplicantIncome',
            'CoapplicantIncome',
            'LoanAmount',
            'Loan_Amount_Term',
            'Credit_History',
            'Property_Area'
        ]

        data = pd.DataFrame(input_data, columns=columns)

        # Scale the input
        data = scale.transform(data)

        # Prediction
        prediction = model.predict(data)
        prediction = int(prediction[0])

        if prediction == 1:
            result = "Loan Approved ✅"
        else:
            result = "Loan Not Approved ❌"

        return render_template("output.html", result=result)

    except Exception as e:
        return render_template("output.html", result=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)