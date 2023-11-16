from flask import Flask, request, render_template
import pandas as pd
import joblib

# Create the Flask app
app = Flask(__name__)

# Load the trained model and scalers
model = joblib.load('./rf_model.pkl')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["GET","POST"])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        fixed_acidity = request.form['fixed_acidity']
        volatile_acidity = request.form['volatile_acidity']
        citric_acid = request.form['citric_acid']
        residual_sugar = request.form['residual_sugar']
        chlorides = request.form['chlorides']
        free_sulfur_dioxide = request.form['free_sulfur_dioxide']
        total_sulfur_dioxide = request.form['total_sulfur_dioxide']
        density = request.form['density']
        pH = request.form['pH']
        sulphates = request.form['sulphates']
        alcohol = request.form['alcohol']

        # Create a DataFrame with the user input
        input_data = pd.DataFrame([[float(fixed_acidity), float(volatile_acidity), float(citric_acid), float(residual_sugar), float(chlorides),
                                  float(free_sulfur_dioxide), float(total_sulfur_dioxide), float(density), float(pH), float(sulphates),
                                  float(alcohol)]])
        input_data = pd.DataFrame(data=input_data.values, columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
       'total sulfur dioxide','density','pH','sulphates','alcohol'])

        prediction = model.predict(input_data)
        print(input_data)
        list1=[fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,prediction]
        
        prediction = model.predict(input_data)[0]

        # Map predicted quality score to categories
        if prediction < 5:
            quality_category = "Bad Quality"
        elif prediction < 7:
            quality_category = "Average Quality"
        elif prediction < 9:
            quality_category = "Good Quality"
        else:
            quality_category = "Excellent Quality"
            
        # Render an HTML template with the prediction
        #return render_template('result.html', data=list1)
        return render_template('results.html',quality_category=quality_category, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')