import numpy as np
import pandas as pd
import os 
from flask import Flask,url_for,render_template,redirect,request,session
from flask_wtf import FlaskForm
from wtforms import StringField, TextField, SubmitField,SelectField,IntegerField
from wtforms.validators import DataRequired, Length
from tensorflow.keras.models import load_model
from flask_wtf.csrf import CSRFProtect


app = Flask(__name__)
#CONFIGURING SECRET KEY 
csrf = CSRFProtect(app)
model = load_model('prediction_model.h5')
app.config['SECRET_KEY']= 'Query76'
app.config['SESSION_TYPE'] = 'filesystem'
DESTINATION_COLUMNS = ['age',
 'fnlwgt',
 'education_num',
 'capital_gain',
 'capital_loss',
 'workhour',
 'workclass_Federal-gov',
 'workclass_Local-gov',
 'workclass_Never-worked',
 'workclass_Private',
 'workclass_Self-emp-inc',
 'workclass_Self-emp-not-inc',
 'workclass_State-gov',
 'workclass_Without-pay',
 'education_10th',
 'education_11th',
 'education_12th',
 'education_1st-4th',
 'education_5th-6th',
 'education_7th-8th',
 'education_9th',
 'education_Assoc-acdm',
 'education_Assoc-voc',
 'education_Bachelors',
 'education_Doctorate',
 'education_HS-grad',
 'education_Masters',
 'education_Preschool',
 'education_Prof-school',
 'education_Some-college',
 'marital_Divorced',
 'marital_Married-AF-spouse',
 'marital_Married-civ-spouse',
 'marital_Married-spouse-absent',
 'marital_Never-married',
 'marital_Separated',
 'marital_Widowed',
 'occupation_Adm-clerical',
 'occupation_Armed-Forces',
 'occupation_Craft-repair',
 'occupation_Exec-managerial',
 'occupation_Farming-fishing',
 'occupation_Handlers-cleaners',
 'occupation_Machine-op-inspct',
 'occupation_Other-service',
 'occupation_Priv-house-serv',
 'occupation_Prof-specialty',
 'occupation_Protective-serv',
 'occupation_Sales',
 'occupation_Tech-support',
 'occupation_Transport-moving',
 'relationship_Husband',
 'relationship_Not-in-family',
 'relationship_Other-relative',
 'relationship_Own-child',
 'relationship_Unmarried',
 'relationship_Wife',
 'race_Amer-Indian-Eskimo',
 'race_Asian-Pac-Islander',
 'race_Black',
 'race_Other',
 'race_White',
 'sex_Female',
 'sex_Male',
 'country_Cambodia',
 'country_Canada',
 'country_China',
 'country_Columbia',
 'country_Cuba',
 'country_Dominican-Republic',
 'country_Ecuador',
 'country_El-Salvador',
 'country_England',
 'country_France',
 'country_Germany',
 'country_Greece',
 'country_Guatemala',
 'country_Haiti',
 'country_Holand-Netherlands',
 'country_Honduras',
 'country_Hong',
 'country_Hungary',
 'country_India',
 'country_Iran',
 'country_Ireland',
 'country_Italy',
 'country_Jamaica',
 'country_Japan',
 'country_Laos',
 'country_Mexico',
 'country_Nicaragua',
 'country_Outlying-US(Guam-USVI-etc)',
 'country_Peru',
 'country_Philippines',
 'country_Poland',
 'country_Portugal',
 'country_Puerto-Rico',
 'country_Scotland',
 'country_South',
 'country_Taiwan',
 'country_Thailand',
 'country_Trinadad&Tobago',
 'country_United-States',
 'country_Vietnam',
 'country_Yugoslavia']
CATEGORICAL = ['workclass','education','marital','occupation','relationship','race','sex','country']
NUMERICAL = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'workhour']

#FORM FOR FEATURES


class FEATURES(FlaskForm):
    """features  form."""
    age = IntegerField('Age',[DataRequired('Fill  Age')])
    workclass = SelectField('WorkClass',[DataRequired('Select WorkClass')],choices=[('State-gov', 'State-gov'),
                                                                                    ('Self-emp-not-inc', 'Self-emp-not-inc'),
                                                                                    ('Private', 'Private'),
                                                                                    ('Federal-gov', 'Federal-gov'),
                                                                                    ('Local-gov', 'Local-gov'),
                                                                                    ('Self-emp-inc', 'Self-emp-inc'),
                                                                                    ('Without-pay', 'Without-pay'),
                                                                                    ('Never-worked', 'Never-worked')])

    fnlwgt = IntegerField('Fnlwgt')
    education = SelectField('Education',[DataRequired('Select education')],choices=[('Bachelors', 'Bachelors'),
                                                                                    ('HS-grad', 'HS-grad'),
                                                                                    ('11th', '11th'),
                                                                                    ('Masters', 'Masters'),
                                                                                    ('9th', '9th'),
                                                                                    ('Some-college', 'Some-college'),
                                                                                    ('Assoc-acdm', 'Assoc-acdm'),
                                                                                    ('Assoc-voc', 'Assoc-voc'),
                                                                                    ('7th-8th', '7th-8th'),
                                                                                    ('Doctorate', 'Doctorate'),
                                                                                    ('Prof-school', 'Prof-school'),
                                                                                    ('5th-6th', '5th-6th'),
                                                                                    ('10th', '10th'),
                                                                                    ('1st-4th', '1st-4th'),
                                                                                    ('Preschool', 'Preschool'),
                                                                                    ('12th', '12th')])

    education_num = IntegerField('Number of Education',[DataRequired('Select education')])

    marital_status = SelectField('Marital Status',[DataRequired('Select marital_status')],choices=[('Never-married', 'Never-married'),
                                                                                                    ('Married-civ-spouse', 'Married-civ-spouse'),
                                                                                                    ('Divorced', 'Divorced'),
                                                                                                    ('Married-spouse-absent', 'Married-spouse-absent'),
                                                                                                    ('Separated', 'Separated'),
                                                                                                    ('Married-AF-spouse', 'Married-AF-spouse'),
                                                                                                    ('Widowed', 'Widowed')])
    occupation = SelectField('Occupation',[DataRequired('Select occupation')],choices=[('Adm-clerical', 'Adm-clerical'),
                                                                                        ('Exec-managerial', 'Exec-managerial'),
                                                                                        ('Handlers-cleaners', 'Handlers-cleaners'),
                                                                                        ('Prof-specialty', 'Prof-specialty'),
                                                                                        ('Other-service', 'Other-service'),
                                                                                        ('Sales', 'Sales'),
                                                                                        ('Craft-repair', 'Craft-repair'),
                                                                                        ('Transport-moving', 'Transport-moving'),
                                                                                        ('Farming-fishing', 'Farming-fishing'),
                                                                                        ('Machine-op-inspct', 'Machine-op-inspct'),
                                                                                        ('Tech-support', 'Tech-support'),
                                                                                        ('Protective-serv', 'Protective-serv'),
                                                                                        ('Armed-Forces', 'Armed-Forces'),
                                                                                        ('Priv-house-serv', 'Priv-house-serv')])

    relationship = SelectField('Relationship',[DataRequired('Select relationship')],choices=[('Not-in-family', 'Not-in-family'),
                                                                                            ('Husband', 'Husband'),
                                                                                            ('Wife', 'Wife'),
                                                                                            ('Own-child', 'Own-child'),
                                                                                            ('Unmarried', 'Unmarried'),
                                                                                            ('Other-relative', 'Other-relative')])
    sex = SelectField('Sex',[DataRequired('select sex')],choices=[('Male', 'Male'), ('Female', 'Female')])
    race = SelectField('Race',[DataRequired('select race')],choices=[('White', 'White'),
                                                                    ('Black', 'Black'),
                                                                    ('Asian-Pac-Islander', 'Asian-Pac-Islander'),
                                                                    ('Amer-Indian-Eskimo', 'Amer-Indian-Eskimo'),
                                                                    ('Other', 'Other')])
    capital_gain = IntegerField('Capital-Gain',[DataRequired('Fill  Capital Gain')])
    capital_loss = IntegerField('Capital-Loss',[DataRequired('Fill  Capital Loss')])
    hours_per_week = IntegerField('Work Hour Per Week',[DataRequired('Fill  Work hour per week')])
    native_country = SelectField('Country',[DataRequired('Select Native Country')],choices=[('United-States', 'United-States'),
                                                                                                    ('Cuba', 'Cuba'),
                                                                                                    ('Jamaica', 'Jamaica'),
                                                                                                    ('India', 'India'),
                                                                                                    ('Mexico', 'Mexico'),
                                                                                                    ('South', 'South'),
                                                                                                    ('Puerto-Rico', 'Puerto-Rico'),
                                                                                                    ('Honduras', 'Honduras'),
                                                                                                    ('England', 'England'),
                                                                                                    ('Canada', 'Canada'),
                                                                                                    ('Germany', 'Germany'),
                                                                                                    ('Iran', 'Iran'),
                                                                                                    ('Philippines', 'Philippines'),
                                                                                                    ('Italy', 'Italy'),
                                                                                                    ('Poland', 'Poland'),
                                                                                                    ('Columbia', 'Columbia'),
                                                                                                    ('Cambodia', 'Cambodia'),
                                                                                                    ('Thailand', 'Thailand'),
                                                                                                    ('Ecuador', 'Ecuador'),
                                                                                                    ('Laos', 'Laos'),
                                                                                                    ('Taiwan', 'Taiwan'),
                                                                                                    ('Haiti', 'Haiti'),
                                                                                                    ('Portugal', 'Portugal'),
                                                                                                    ('Dominican-Republic', 'Dominican-Republic'),
                                                                                                    ('El-Salvador', 'El-Salvador'),
                                                                                                    ('France', 'France'),
                                                                                                    ('Guatemala', 'Guatemala'),
                                                                                                    ('China', 'China'),
                                                                                                    ('Japan', 'Japan'),
                                                                                                    ('Yugoslavia', 'Yugoslavia'),
                                                                                                    ('Peru', 'Peru'),
                                                                                                    ('Outlying-US(Guam-USVI-etc)', 'Outlying-US(Guam-USVI-etc)'),
                                                                                                    ('Scotland', 'Scotland'),
                                                                                                    ('Trinadad&Tobago', 'Trinadad&Tobago'),
                                                                                                    ('Greece', 'Greece'),
                                                                                                    ('Nicaragua', 'Nicaragua'),
                                                                                                    ('Vietnam', 'Vietnam'),
                                                                                                    ('Hong', 'Hong'),
                                                                                                    ('Ireland', 'Ireland'),
                                                                                                    ('Hungary', 'Hungary'),
                                                                                                    ('Holand-Netherlands', 'Holand-Netherlands')])
    predict = SubmitField('Predict')
def match_input(destination,numerical,categorical,form_input) : 
    for col in numerical : 
        destination[col] = form_input[col]
    for col in categorical : 
        val = form_input[col][0]
        destination_colname = f'{col}_{val}'
        print(destination_colname)
        #finding match column 
        destination[destination_colname] = 1
        
    return destination

def predict_result(model,content):
    destination = pd.DataFrame({x : 0 for x in DESTINATION_COLUMNS },index=[0])
    form_input = pd.DataFrame(content,index=[0])
    feature_input = match_input(destination=destination,numerical=NUMERICAL,categorical=CATEGORICAL,form_input=form_input)
    #scaling numerical value
    
    for column in NUMERICAL : 
        feature_input[column] == np.log1p(feature_input[column]) 
    feature_input.to_excel('feat.xlsx',index=False)



    array_data = np.asarray(feature_input)
    result = model.predict(array_data)
    return result

@app.route('/',methods=['POST'])
def main():
    #creating form instance
    form = FEATURES()
    return render_template('index.html',form=form)

@app.route('/prediction',methods=['GET','POST'])
def prediction():
    form = FEATURES()
    value = [x for x in request.form.values()]
    content = {}
    content['age'] = int(value[1])
    content['workclass'] = value[2]
    content['fnlwgt'] = float(int(value[3]))
    content['education'] = value[4]
    content['education_num'] = float(int(value[5]))
    content['marital'] = value[6]
    content['occupation'] = value[7]
    content['relationship'] = value[8]
    content['sex'] = value[9]
    content['race'] = value[10]
    content['capital_gain'] = float(int(value[11]))
    content['capital_loss'] = float(int(value[12]))
    content['workhour'] = float(int(value[13]))
    content['country'] = value[14]
    result = predict_result(model=model,content=content)
    return render_template('predict_result.html',value=value,prediction=result)


if __name__ == '__main__' :
    app.run(debug=True)


