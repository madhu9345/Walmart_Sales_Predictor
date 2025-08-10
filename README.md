Walmart Sales Predictor
📌 Overview
The Walmart Sales Predictor is a machine learning project designed to forecast sales for Walmart stores based on historical data, seasonal patterns, and promotional events. This model helps in decision-making for inventory management, marketing strategies, and store planning.

🚀 Features
Data preprocessing and cleaning of historical sales data

Exploratory Data Analysis (EDA) with visual insights

Feature engineering to improve prediction accuracy

Multiple ML model training and evaluation (e.g., Linear Regression, Random Forest, XGBoost)

Streamlit app for user-friendly prediction interface

📂 Project Structure
Walmart_Sales_Predictor/
│
├── data_preprocessing.py  # Python scripts for preprocessing, modeling, and visualization  
├── app.py                 # Streamlit app  
├── requirements.txt       # Required Python libraries  
└── README.md              # Project documentation  
🛠️ Installation & Setup
Clone the repository
git clone https://github.com/<your-username>/Walmart_Sales_Predictor.git
cd Walmart_Sales_Predictor

Install dependencies

pip install -r requirements.txt
Run the Streamlit app

streamlit run app.py
📊 Dataset
The dataset used is from Kaggle - Walmart Sales Forecasting. It contains weekly sales data for various Walmart stores, along with information on promotions, holidays, and weather.

📈 Model Performance
Algorithm Used: Random Forest Regressor (best performing model)

Evaluation Metrics: RMSE, MAE, R² Score

Achieved an R² score of 0.92 on the test dataset.

🎯 Usage
Open the Streamlit app.

Enter store details, date, and other relevant features.

Get sales prediction instantly.

🤝 Contributing
Contributions are welcome! Please fork the repository and create a pull request with your improvements.

📜 License
This project is licensed under the MIT License.
