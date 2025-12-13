# 🏥 AI-Based Disease Prediction Web Application

An end-to-end **machine learning web application** that predicts possible diseases based on user-selected symptoms. The project demonstrates the complete ML lifecycle — from data preprocessing and model training to deployment with a user-friendly web interface.

---

## 📌 Features

* 🔐 User Authentication (Login & Registration)
* 🧠 Disease Prediction using Machine Learning
* 📊 Probability-based confidence visualization
* 🤖 Multiple ML model comparison

  * Logistic Regression
  * Naive Bayes
  * Random Forest (Final model)
* 🗂️ User prediction history tracking
* 🌐 Interactive web interface using Streamlit
* ☁️ Deployment-ready architecture

---

## 🧠 Tech Stack

| Category             | Technology      |
| -------------------- | --------------- |
| Programming Language | Python          |
| Frontend             | Streamlit       |
| Machine Learning     | Scikit-learn    |
| Database             | SQLite          |
| Data Processing      | Pandas, NumPy   |
| Deployment           | Streamlit Cloud |

---

## 📁 Project Structure

```
disease_prediction_app/
│
├── app.py                 # Streamlit web application
├── model.py               # Model loading & prediction logic
├── train_models.py        # Train and compare ML models
├── auth.py                # User authentication logic
├── database.py            # SQLite database setup
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
└── data/
    └── symptoms.csv       # Dataset
```

---

## 📊 Dataset

* **Source**: Kaggle – *Disease Prediction Using Symptoms*
* **Description**: Structured dataset mapping multiple symptoms to diseases
* **Preprocessing**:

  * Missing values handled
  * Binary symptom encoding
  * Label encoding for diseases

---

## 🤖 Machine Learning Models

| Model               | Purpose                                     |
| ------------------- | ------------------------------------------- |
| Logistic Regression | Baseline model                              |
| Naive Bayes         | Probabilistic classification                |
| Random Forest       | Final selected model due to better accuracy |

Model performance is evaluated using **train-test split** and **accuracy score**.

---

## 🚀 How to Run the Project Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/disease-prediction-app.git
cd disease-prediction-app
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Models

```bash
python train_models.py
```

### 4️⃣ Run the Web Application

```bash
streamlit run app.py
```

---

## 🌍 Deployment

The application can be deployed for free using **Streamlit Cloud**:

1. Push the project to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Select the repository and `app.py`
4. Deploy and get a public URL

---

## 🧪 Example Workflow

1. User registers and log
