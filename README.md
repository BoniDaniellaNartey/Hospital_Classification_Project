# HeartCare – Cardiovascular Disease Prediction System

## 1. Project Overview

HeartCare is a web-based cardiovascular disease prediction system developed using Machine Learning. The system uses patient health information to predict whether a patient is likely to have cardiovascular disease.

The project applies Logistic Regression to a hospital dataset and provides an interactive interface for entering patient information, viewing predictions, and managing prediction records.

## 2. Project Objectives

- Develop a machine learning-based cardiovascular disease prediction system.
- Predict the presence or absence of cardiovascular disease.
- Store and retrieve patient prediction records.
- Present prediction statistics through a dashboard and analytics page.

## 3. Technologies Used

- Python
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- SQLite
- HTML
- CSS
- JavaScript

## 4. Machine Learning Model

The system uses Logistic Regression with StandardScaler in a Scikit-learn Pipeline.

The model uses the following input features:

1. Age
2. Sex
3. Blood Pressure
4. Cholesterol
5. BMI
6. Smoking Status
7. Diabetes Status
8. Physical Activity
9. Family History
10. Stress Level

The target variable is `heart_disease`.

## 5. System Features

- Dashboard
- New Patient Prediction
- Patient History
- Analytics
- About Page
- Automatic storage of prediction records

## 6. Installation and Setup

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

Navigate into the project folder:

```bash
cd Hospital_Classification_Project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
uvicorn app:app --reload
```

Open the application at:

`http://127.0.0.1:8000`

## 7. Project Disclaimer

HeartCare is an academic machine learning project designed to demonstrate the application of classification techniques to healthcare-related data. Its predictions should not replace professional medical assessment.