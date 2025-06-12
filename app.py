import streamlit as st
import pandas as pd
import pickle


st.title("🏦 Loan Approval Prediction - Batch Mode")

@st.cache_resource
def load_model():
    with open("loan_approval_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()


def preprocess_data(df):
    df = df.copy()

    
    df.drop(columns=[col for col in ['Loan_ID', 'Loan_Status'] if col in df.columns], inplace=True, errors='ignore')

    #categorical columns encoding
    mapping_dict = {
        'Gender': {'Male': 1, 'Female': 0},
        'Married': {'Yes': 1, 'No': 0},
        'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
        'Education': {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
    }

    for col, mapping in mapping_dict.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Fill any  missing values
    df.fillna(0, inplace=True)

    # Convert all columns to numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df


uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)
    st.subheader("📋 Raw Uploaded Data")
    st.dataframe(raw_data)

    if st.button("🔍 Predict"):
        try:
            data = preprocess_data(raw_data)
            st.subheader("🛠️ Preprocessed Data")
            st.dataframe(data)

            
            if not all(dtype in ['int64', 'float64'] for dtype in data.dtypes):
                st.error("❌ Data contains non-numeric columns even after preprocessing.")
            else:
                preds = model.predict(data)
                raw_data['Prediction'] = ['Approved' if x == 1 else 'Rejected' for x in preds]
                st.success("✅ Predictions generated successfully!")
                st.subheader("📊 Results")
                st.dataframe(raw_data)
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
