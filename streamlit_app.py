import streamlit as st
import pandas as pd
import joblib


st.write("## HR Analysis: Employee Promotion Prediction")

st.write("""
Predict whether an employee will be promoted or not using different
machine learning models.
""")

st.image("images/HR-Tech-landscape.jpg")

st.write("""
## About

<p align="justify">Understanding the dynamics of employee promotion is crucial for maintaining a
motivated and engaged workforce while ensuring the growth and success of an
organization. This web app leverages advanced data preprocessing and machine
learning techniques to predict the likelihoods of an employee promotion.
In addition this web app utilizes historical employee data, such as performance
metrics, tenure, past promotions, educational background, and more, to build
predictive models.</p>
""", unsafe_allow_html=True)

st.markdown("""

Below are the performances of the models on the training data without any
sampling technique

<table>
  <tr>
    <th>Model</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>XGBoost</td>
    <td>0.942438</td>
    <td>0.887955</td>
    <td>0.349119</td>
    <td>0.501186</td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td>0.935504</td>
    <td>0.817035</td>
    <td>0.285242</td>
    <td>0.422857</td>
  </tr>
  <tr>
    <td>Tunned Random Forest</td>
    <td>0.931034</td>
    <td>0.969136</td>
    <td>0.172907</td>
    <td>0.293458</td>
  </tr>
  <tr>
    <td>Logistic Regression</td>
    <td>0.921091</td>
    <td>0.586345</td>
    <td>0.160793</td>
    <td>0.252377</td>
  </tr>
</table>


---

Random Forest was chosen as my sampling technique comparison model. Below is how the different sampling techniques compare

---

<table>
  <tr>
    <th>Random Forest with</th>
    <th>Accuracy</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 Score</th>
  </tr>
  <tr>
    <td>Random Oversampling</td>
    <td>0.929028</td>
    <td>0.626459</td>
    <td>0.354626</td>
    <td>0.452883</td>
  </tr>
  <tr>
    <td>No Under/Oversampling</td>
    <td>0.937694</td>
    <td>0.841945</td>
    <td>0.305066</td>
    <td>0.447858</td>
  </tr>
  <tr>
    <td>SMOTE Oversampling</td>
    <td>0.933224</td>
    <td>0.716749</td>
    <td>0.320485</td>
    <td>0.442922</td>
  </tr>
  <tr>
    <td>Class Weight</td>
    <td>0.937238</td>
    <td>0.845912</td>
    <td>0.296256</td>
    <td>0.438825</td>
  </tr>
  <tr>
    <td>SMOTE + Tomek</td>
    <td>0.932129</td>
    <td>0.705000</td>
    <td>0.310573</td>
    <td>0.431193</td>
  </tr>
</table>

---

""", unsafe_allow_html=True)


st.write("""
The sampling models were not included here due to their enormous sizes - you can check them out on my Github though. Everything you need to know regarding this project including the documentation, notebook, dataset, precision-recall dilemma, etc. can be found in my repository on [Github](https://github.com/Oyebamiji-Micheal/Employee-Status-Prediction-Web-App/tree/main).
""")

predict_promotion = st.button("Predict Employee Promotion")

st.sidebar.header("Select Model")

model_name = st.sidebar.selectbox(
    "Select Model", ("Random Forest", "Logistic Regression", "XGBoost")
)

st.sidebar.header("User Input Features")

no_of_trainings = st.sidebar.slider("No of trainings", min_value=1, max_value=10)

age = st.sidebar.slider("Age", min_value=1, max_value=100)

length_of_service = st.sidebar.slider("Length of Service", min_value=1, max_value=50)

education = st.sidebar.selectbox(
    "Educational Level", ("Master's & above", "Bachelor's", "Below Secondary", "Unknown")
)

gender = st.sidebar.selectbox("Gender", ("Male", "Female"))

awards = st.sidebar.number_input(
    "Awards Won: Enter 1 if employee has won an award and 0 otherwise", min_value=0, max_value=1
)

department = st.sidebar.selectbox(
    "Department",
    (
        "Sales & Marketing", "Operations", "Technology", "Analytics",
        "R&D", "Procurement", "Finance", "HR", "Legal"
    ),
)

recruitment_channel = st.sidebar.selectbox("Recruitment Channel", ("Sourcing", 'Other', 'Referred'))

kpis_met = st.sidebar.number_input(
    "KPIs Met: Enter 1 if employee has met 80% KPI and 0 otherwise", min_value=0, max_value=1
)

avg_training_score = st.sidebar.slider("Employee Average Training Score", min_value=0, max_value=100)

previous_year_rating = st.sidebar.slider("Employee Previous Year Rating", min_value=1, max_value=5)

region = st.sidebar.slider("Employee Region", min_value=1, max_value=34)


def predict_input(single_input):
    hr_pred = joblib.load('hr_models.joblib')
    input_df = pd.DataFrame([single_input])
    categorical_cols = hr_pred['categorical_cols']
    input_df[['gender', 'education']] = hr_pred['ordinal_enc'].transform(input_df[['gender', 'education']])
    input_df[hr_pred['encoded_cols']] = hr_pred['onehot_enc'].transform(input_df[categorical_cols])
    X_input = input_df[['gender', 'education'] + hr_pred['numeric_cols'] + hr_pred['encoded_cols']]
    try:
        pred = hr_pred[model_name].predict(X_input)
    except KeyError:
        pred = hr_pred['RF No Sampling'].predict(X_input)

    return pred


if predict_promotion:
    # Format inputs
    gender_mapping = {'Male': 'm', 'Female': 'f'}
    region = 'region_' + str(region)
    recruitment_channel = recruitment_channel.lower()

    single_input = {
        'department':  department,
        'region': region,
        'education': education,
        'gender': gender_mapping[gender],
        'recruitment_channel': recruitment_channel,
        'no_of_trainings': no_of_trainings,
        'age': age,
        'previous_year_rating': previous_year_rating,
        'length_of_service': length_of_service,
        'KPIs_met >80%': kpis_met,
        'awards_won?': awards,
        'avg_training_score': avg_training_score
    }

    prediction = predict_input(single_input)

    st.write(f'Classifier = {model_name}')

    if prediction[0] == 1:
        st.write('Predicted Status = Promoted')
    else:
        st.write('Predicted Status = Not Promoted')
