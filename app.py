from flask import Flask, render_template, request
import numpy as np
import pickle

diabetes_model = pickle.load(open('models/diabetes.pkl', 'rb'))
cancer_model = pickle.load(open('models/breast_cancer.pkl', 'rb'))
heart_model = pickle.load(open('models/heart.pkl', 'rb'))
liver_model = pickle.load(open('models/liver.pkl', 'rb'))
kidney_model = pickle.load(open('models/kidney.pkl', 'rb'))
parkinson_model = pickle.load(open('models/ParkinsonsDetection.pkl', 'rb'))
insurance_model = pickle.load(open('models/insurance_prediction.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/breast_cancer", methods=['GET','POST'])
def cancer():
    return render_template('breast_cancer.html')

@app.route("/diabetes", methods=['GET','POST'])
def diabetes():
    return render_template('diabetes.html')

@app.route("/heart", methods=['GET','POST'])
def heart():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET','POST'])
def kidney():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET','POST'])
def liver():
    return render_template('liver.html')

@app.route("/parkinson", methods=['GET','POST'])
def parkinson():
    return render_template('parkinson.html')

@app.route("/insurance", methods=['GET','POST'])
def insurance():
    return render_template('medical_cost.html')

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        if(len([float(x) for x in request.form.values()])==8):
            Pregnancies = float(request.form['Pregnancies'])
            Glucose = float(request.form['Glucose'])
            BloodPressure = float(request.form['BloodPressure'])
            SkinThickness = float(request.form['SkinThickness'])
            Insulin = float(request.form['Insulin'])
            BMI = float(request.form['BMI'])
            DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
            Age = float(request.form['Age'])
            
            data = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction, Age]])
            my_prediction = diabetes_model.predict(data)
            
            return render_template('predict.html', prediction=my_prediction)
        
        elif(len([float(x) for x in request.form.values()])==6):
            age = int(request.form['age'])
            bmi	= int(request.form['bmi'])
            children = int(request.form['children'])
            smoker_enc = int(request.form['smoker_enc'])
            sex_enc	= int(request.form['sex_enc'])
            region_enc = int(request.form['region_enc'])
            
            data = np.array([[age,bmi,children,smoker_enc,sex_enc,region_enc]])
            my_prediction = insurance_model.predict(data)
            
            return render_template('predict_reg.html', prediction=my_prediction)
            
        elif(len([float(x) for x in request.form.values()])==10):
            Age = int(request.form['Age'])
            Gender_Male = int(request.form['Gender_Male'])
            Total_Bilirubin = float(request.form['Total_Bilirubin'])
            Direct_Bilirubin = float(request.form['Direct_Bilirubin'])
            Alkaline_Phosphotase = int(request.form['Alkaline_Phosphotase'])
            Alamine_Aminotransferase = int(request.form['Alamine_Aminotransferase'])
            Aspartate_Aminotransferase = int(request.form['Aspartate_Aminotransferase'])
            Total_Protiens = float(request.form['Total_Protiens'])
            Albumin = float(request.form['Albumin'])
            Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])
            

            data = np.array([[Age,Gender_Male,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]])
            my_prediction = liver_model.predict(data)
            return render_template('predict.html', prediction=my_prediction)

        elif(len([float(x) for x in request.form.values()])==13):
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])

            data = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
            data1 = np.array(data).reshape(1,-1)
            my_prediction = heart_model.predict(data1)
            return render_template('predict.html', prediction=my_prediction)
        
        elif(len([float(x) for x in request.form.values()])==22):
            MDVP_Fo_Hz = float(request.form['MDVP_Fo_Hz'])
            MDVP_Fhi_Hz = float(request.form['MDVP_Fhi_Hz'])
            MDVP_Flo_Hz = float(request.form['MDVP_Flo_Hz'])
            MDVP_Jitter_Percent = float(request.form['MDVP_Jitter_Percent'])
            MDVP_Jitter_Abs = float(request.form['MDVP_Jitter_Abs'])
            MDVP_RAP = float(request.form['MDVP_RAP'])
            MDVP_PPQ = float(request.form['MDVP_PPQ'])
            Jitter_DDP = float(request.form['Jitter_DDP'])
            MDVP_Shimmer = float(request.form['MDVP_Shimmer'])
            MDVP_Shimmer_dB = float(request.form['MDVP_Shimmer_dB'])
            Shimmer_APQ3 = float(request.form['Shimmer_APQ3'])
            Shimmer_APQ5 = float(request.form['Shimmer_APQ5'])
            MDVP_APQ = float(request.form['MDVP_APQ'])
            Shimmer_DDA = float(request.form['Shimmer_DDA'])
            NHR = float(request.form['NHR'])
            HNR = float(request.form['HNR'])
            RPDE = float(request.form['RPDE'])
            DFA = float(request.form['DFA'])
            spread1 = float(request.form['spread1'])
            spread2 = float(request.form['spread2'])
            D2 = float(request.form['D2'])
            PPE = float(request.form['PPE'])

            data = [MDVP_Fo_Hz,MDVP_Fhi_Hz,MDVP_Flo_Hz,MDVP_Jitter_Percent,MDVP_Jitter_Abs,MDVP_RAP,MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,
       MDVP_Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,MDVP_APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]
            data1 = np.array(data).reshape(1,-1)
            my_prediction = parkinson_model.predict(data1)
            return render_template('predict.html', prediction=my_prediction)

        elif(len([float(x) for x in request.form.values()])==24):
            age = float(int(request.form['age']))
            blood_pressure = float(request.form['blood_pressure'])
            specific_gravity = float(request.form['specific_gravity'])
            albumin = float(request.form['albumin'])
            sugar = float(request.form['sugar'])
            red_blood_cells = int(request.form['red_blood_cells'])
            pus_cell = int(request.form['pus_cell'])
            pus_cell_clumps = int(request.form['pus_cell_clumps'])
            bacteria = int(request.form['bacteria'])
            blood_glucose_random = float(request.form['blood_glucose_random'])
            blood_urea = float(request.form['blood_urea'])
            serum_creatinine = float(request.form['serum_creatinine'])
            sodium = int(request.form['sodium'])
            potassium = float(request.form['potassium'])
            haemoglobin = float(request.form['haemoglobin'])
            packed_cell_volume = float(request.form['packed_cell_volume'])
            white_blood_cell_count = int(request.form['white_blood_cell_count'])
            red_blood_cell_count = int(request.form['red_blood_cell_count'])
            hypertension = int(request.form['hypertension'])
            diabetes_mellitus = int(request.form['diabetes_mellitus'])
            coronary_artery_disease = int(request.form['coronary_artery_disease'])
            appetite = int(request.form['appetite'])
            peda_edema = int(request.form['peda_edema'])
            aanemia = int(request.form['aanemia'])
            
            
            data = [age,blood_pressure,specific_gravity,albumin,sugar,red_blood_cells,pus_cell,pus_cell_clumps,bacteria,blood_glucose_random,blood_urea,serum_creatinine,sodium,potassium,haemoglobin,packed_cell_volume,
       white_blood_cell_count,red_blood_cell_count,hypertension,
       diabetes_mellitus,coronary_artery_disease,appetite,
       peda_edema,aanemia]
            data1 = np.array(data).reshape(1,-1)
            my_prediction = kidney_model.predict(data1)
            return render_template('predict.html', prediction=my_prediction)

        elif(len([float(x) for x in request.form.values()])==20):
        
            smoothness_mean = float(request.form['smoothness_mean'])
            compactness_mean = float(request.form['compactness_mean'])
            symmetry_mean = float(request.form['symmetry_mean'])
            fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
            texture_se = float(request.form['texture_se'])
            area_se = float(request.form['area_se'])
            smoothness_se = float(request.form['smoothness_se'])
            compactness_se = float(request.form['compactness_se'])
            concavity_se = float(request.form['concavity_se'])
            concave_points_se = float(request.form['concave_points_se'])
            symmetry_se = float(request.form['symmetry_se'])
            fractal_dimension_se = float(request.form['fractal_dimension_se'])
            texture_worst = float(request.form['texture_worst'])
            area_worst = float(request.form['area_worst'])
            smoothness_worst = float(request.form['smoothness_worst'])
            compactness_worst = float(request.form['compactness_worst'])
            concavity_worst = float(request.form['concavity_worst'])
            concave_points_worst = float(request.form['concave_points_worst'])
            symmetry_worst = float(request.form['symmetry_worst'])
            fractal_dimension_worst = float(request.form['fractal_dimension_worst'])

            data = [smoothness_mean,compactness_mean,symmetry_mean,fractal_dimension_mean,texture_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,texture_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]
            data1 = np.array(data).reshape(1,-1)
            my_prediction  = cancer_model.predict(data1)
            return render_template('predict.html', prediction=my_prediction)
        
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)