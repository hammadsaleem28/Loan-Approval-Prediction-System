# Loan Approval Prediction System

## Overview

This project is an AI-powered loan approval prediction system that uses machine learning to determine the likelihood of loan approval based on applicant information. The system utilizes an XGBoost classifier model trained on historical loan data to make accurate predictions and provides an interactive web interface built with Streamlit.

## Features

- **Batch Prediction**: Upload CSV files containing multiple applicant data for bulk predictions
- **Individual Prediction**: Enter applicant details manually for single predictions
- **Interactive Dashboard**: User-friendly interface with visualizations
- **Confidence Scores**: Probability estimates for each prediction
- **Data Visualization**: Visual representation of income vs. approval status
- **Dark Theme UI**: Modern and sleek user interface
- **Downloadable Results**: Export prediction results as CSV files

## Project Structure

```
├── Loan_Train.csv         # Training dataset with historical loan data
├── Loan_Test.csv          # Test dataset for model evaluation
├── app.py                 # Streamlit web application
├── loan_approval_model.pkl # Trained XGBoost model
├── main_code.ipynb        # Jupyter notebook with model development
└── requirements.txt       # Project dependencies
```

## Installation

1. Clone this repository or download the project files

2. Create a virtual environment (recommended)
   ```bash
   python -m venv loan_env
   loan_env\Scripts\activate
   ```

3. Install the required dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. Navigate to the project directory
   ```bash
   cd "Loan Approval Prediction project"
   ```

2. Launch the Streamlit app
   ```bash
   streamlit run app.py
   ```

3. The application will open in your default web browser at http://localhost:8501

### Batch Prediction

1. Prepare a CSV file with applicant data (must include required features)
2. Upload the CSV file using the file uploader in the app
3. Click "Run Batch Predictions" button
4. View results and download the predictions as a CSV file

### Individual Prediction

1. Fill in the applicant details in the form
2. Click "Predict" button
3. View the prediction result with confidence score

## Model Information

### Input Features

- **Gender**: Male/Female (categorical)
- **Married**: Yes/No (categorical)
- **Dependents**: Number of dependents (0, 1, 2, 3+)
- **Education**: Graduate/Not Graduate (categorical)
- **Self_Employed**: Yes/No (categorical)
- **ApplicantIncome**: Income of the applicant (numerical)
- **CoapplicantIncome**: Income of the co-applicant (numerical)
- **LoanAmount**: Loan amount in thousands (numerical)
- **Loan_Amount_Term**: Term of loan in months (numerical)
- **Credit_History**: Credit history meets guidelines (categorical: 0/1)
- **Property_Area**: Urban/Semiurban/Rural (categorical)

### Model Type

The system uses an XGBoost classifier model trained on historical loan data. The model is saved as `loan_approval_model.pkl` and loaded by the application at runtime.

### Preprocessing

The application performs the following preprocessing steps:

1. Missing value imputation:
   - Categorical variables: filled with mode (most frequent value)
   - Numerical variables: filled with median

2. Categorical encoding:
   - Gender: Male=1, Female=0
   - Married: Yes=1, No=0
   - Education: Graduate=1, Not Graduate=0
   - Self_Employed: Yes=1, No=0
   - Property_Area: Urban=2, Semiurban=1, Rural=0
   - Dependents: Converted to integers (3+ is treated as 3)

## Troubleshooting

### Common Issues

1. **ValueError with categorical data**: Ensure all categorical variables are properly encoded before prediction. The batch prediction feature requires categorical columns to be encoded as numbers.

2. **Missing columns**: Make sure your input data contains all required features for prediction.

3. **Dependency errors**: Verify all required packages are installed using `pip install -r requirements.txt`.

## Future Improvements

- Add model explainability features (SHAP values)
- Implement more advanced data visualization
- Add user authentication system
- Deploy as a cloud-based service
- Add model retraining capability with new data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Created by Hammad Saleem

        
