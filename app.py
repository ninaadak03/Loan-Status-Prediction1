from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the form data
        data = request.form.to_dict()
        Gender = int(data['Gender'])
        Married = int(data['Married'])
        Dependents = int(data['Dependents'])
        Education = int(data['Education'])
        Self_Employed = int(data['Self_Employed'])
        ApplicantIncome = float(data['ApplicantIncome'])
        CoapplicantIncome = float(data['CoapplicantIncome'])
        LoanAmount = float(data['LoanAmount'])
        Loan_Amount_Term = float(data['Loan_Amount_Term'])
        Credit_History = int(data['Credit_History'])
        Property_Area = int(data['Property_Area'])

        # Process the input data
        Total_Income = ApplicantIncome + CoapplicantIncome
        ApplicantIncomelog = np.log(ApplicantIncome + 1)
        LoanAmountlog = np.log(LoanAmount + 1)
        Loan_Amount_Term_log = np.log(Loan_Amount_Term + 1)
        Total_Income_log = np.log(Total_Income + 1)

        # Create a numpy array with the processed data
        input_data = np.array([[Gender, Married, Dependents, Education, Self_Employed, 
                                Credit_History, Property_Area, ApplicantIncomelog, 
                                LoanAmountlog, Loan_Amount_Term_log, Total_Income_log]])

        # Make a prediction
        prediction = model.predict(input_data)
        prediction_text = 'Approved' if prediction[0] == 1 else 'Rejected'

        return render_template('result.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
