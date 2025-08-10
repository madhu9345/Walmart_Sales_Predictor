# Walmart Sales Predictor

A machine learning web application built with **Streamlit** to predict Walmart store sales based on historical sales data and relevant features.

## 📌 Features
- Upload and process Walmart sales datasets.
- Perform **data preprocessing** (cleaning, encoding, scaling).
- Train and evaluate multiple machine learning models.
- Predict future sales using trained models.
- Interactive visualizations for better insights.

## 🛠️ Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- **Streamlit** (Web application framework)
- **Joblib** (Model persistence)
- **Matplotlib & Seaborn** (Data visualization)

## 📂 Project Structure
walmart_sales_predictor/
│── app.py # Main Streamlit application
│── model.pkl # Trained ML model
│── requirements.txt # Python dependencies
│── README.md # Project documentation
│── data/ # Dataset folder



## 🚀 How to Run Locally
1. **Clone this repository**
   
   git clone https://github.com/madhu9345/walmart_sales_predictor.git
   cd walmart_sales_predictor
Create a virtual environment (optional but recommended)

pip install -r requirements.txt
Run the Streamlit app

streamlit run app.py
Open the URL shown in the terminal (usually http://localhost:8501).

📊 Dataset
The dataset used is Walmart sales data containing store, department, date, and sales information.
It includes features like:

Store ID

Department

Weekly Sales

Holiday Flag

Temperature

Fuel Price

CPI

Unemployment Rate

🧠 Model
Data preprocessing (handling missing values, encoding categorical variables, scaling numeric features).

Model training using algorithms like:

Random Forest

Best model selected based on evaluation metrics (R², RMSE, MAE).

📷 Screenshots
(Add screenshots of your app UI here)

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contributing
Contributions are welcome!
