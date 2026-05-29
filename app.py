import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("data/diabetes.csv")

model = pickle.load(
    open("models/adaboost_model.pkl", "rb")
)

scaler = pickle.load(
    open("models/scaler.pkl", "rb")
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(
        to right,
        #4b001f,
        #7b1e3a,
        #a63d5c
    );
}

.title{
    text-align:center;
    font-size:55px;
    font-weight:bold;
    color:white;
}

.subtitle{
    text-align:center;
    font-size:20px;
    color:#f8d7da;
}

.card{
    background:rgba(255,255,255,0.15);
    padding:20px;
    border-radius:20px;
    text-align:center;
    color:white;
}

.glass{
    background:rgba(255,255,255,0.15);
    padding:30px;
    border-radius:20px;
    color:white;
}

.result{
    background:white;
    padding:25px;
    border-radius:20px;
    text-align:center;
    font-size:28px;
    font-weight:bold;
}

.stButton > button{
    width:100%;
    height:55px;
    font-size:20px;
    font-weight:bold;
    border-radius:12px;
    background:#800020;
    color:white;
    border:none;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    "<div class='title'>🩺 Diabetes Prediction</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AdaBoost Classification Model</div>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class='card'>
        <h3>📄 Rows</h3>
        <h2>{df.shape[0]}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='card'>
        <h3>📊 Columns</h3>
        <h2>{df.shape[1]}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='card'>
        <h3>❌ Missing</h3>
        <h2>{df.isnull().sum().sum()}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class='card'>
        <h3>🎯 Features</h3>
        <h2>{df.shape[1]-1}</h2>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------

st.subheader("🔍 Dataset Preview")

st.dataframe(df.head())

# --------------------------------------------------
# ONLY ONE PLOT
# --------------------------------------------------

st.subheader("📈 Diabetes Distribution")

fig, ax = plt.subplots(figsize=(4,3))

outcome_counts = df["Outcome"].value_counts()

ax.bar(
    ["Non-Diabetic", "Diabetic"],
    outcome_counts.values
)

ax.set_title("Target Distribution")
ax.set_ylabel("Count")

st.pyplot(fig)

# --------------------------------------------------
# PATIENT DETAILS
# --------------------------------------------------

st.subheader("🩺 Patient Details")

left, center, right = st.columns([1,3,1])

with center:

    st.markdown("<div class='glass'>", unsafe_allow_html=True)

    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:
        pregnancies = st.slider(
            "Pregnancies",
            0, 20, 1
        )

        glucose = st.slider(
            "Glucose",
            0, 200, 120
        )

        blood_pressure = st.slider(
            "Blood Pressure",
            0, 130, 70
        )

    with row1_col2:
        skin_thickness = st.slider(
            "Skin Thickness",
            0, 100, 20
        )

        insulin = st.slider(
            "Insulin",
            0, 900, 100
        )

        bmi = st.slider(
            "BMI",
            0.0, 70.0, 25.0
        )

    with row1_col3:
        dpf = st.slider(
            "Diabetes Pedigree Function",
            0.0, 3.0, 0.5
        )

        age = st.slider(
            "Age",
            1, 100, 30
        )

    predict = st.button(
        "Predict Diabetes",
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict:

    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(
        input_scaled
    )[0]

    if prediction == 1:

        st.markdown("""
        <div class='result'
        style='color:red'>
        ⚠️ Diabetic
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class='result'
        style='color:green'>
        ✅ Non-Diabetic
        </div>
        """, unsafe_allow_html=True)