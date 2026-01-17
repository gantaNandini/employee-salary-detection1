# 💼 Employee Salary Detection & Analytics App

An end-to-end **Machine Learning + Streamlit** web application that performs **employee income prediction** and **exploratory data analytics** using the Adult Income dataset.  
The app predicts whether an employee earns **>50K or ≤50K** based on demographic and work-related features.

---

## 🚀 Features

### 📊 Data Analytics Dashboard
- Dataset preview
- Summary statistics
- Income distribution visualization
- Age vs Income analysis
- Education vs Income comparison
- Working hours vs Income insights

### 💼 Income Prediction System
- User-friendly Streamlit interface
- Real-time salary prediction
- Uses a trained ML model with:
  - Feature scaling
  - Label encoding
- Predicts:
  - **>50K income**
  - **≤50K income**

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Pandas & NumPy**
- **Scikit-learn**
- **Seaborn & Matplotlib**
- **Pickle (Model Serialization)**

---

## 📂 Project Structure

employee-salary-detection/
│
├── app.py # Streamlit application
├── module.py # Model training / helper logic
├── requirements.txt # Project dependencies
├── salary_model.pkl # Trained ML model (ignored in Git)
├── adult 3.csv # Dataset (ignored in Git)
├── .gitignore
└── README.md

1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/employee-salary-detection.git
cd employee-salary-detection
2️⃣ Create & activate virtual environment
bash
Copy code
python -m venv venv
venv\Scripts\activate
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
▶️ Run the Application
bash
Copy code
streamlit run app.py
