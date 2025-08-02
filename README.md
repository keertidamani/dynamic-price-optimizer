# 🚖 Dynamic Price Optimizer – Machine Learning for Ride-Hailing

This project implements a **machine learning-based dynamic pricing system** for ride-hailing platforms, optimizing fares based on demand, time, and trip characteristics. The system simulates real-world scenarios using synthetic ride data and regression models to predict optimal fares in real-time.

---

## 📌 Problem Statement

In a highly dynamic market like ride-hailing, **static pricing models fail to capture real-time demand-supply fluctuations**. The goal of this project is to:

- Build a dynamic fare prediction model
- Use key ride features (like distance, time, demand level, location, etc.)
- Simulate real-world surge pricing behavior
- Enable smarter, responsive pricing decisions using ML

---

## 📊 Dataset Overview

The dataset used for this project is synthetically generated to mimic real ride data. It includes the following features:

- `pickup_time` – Time of day (binned into peak/non-peak)
- `distance_km` – Distance of the trip
- `duration_min` – Duration of the trip in minutes
- `demand_level` – Simulated scale of 1 to 5
- `base_fare` – Flat base fare per ride
- `actual_price` – Final price charged (used as label)

---

## 🧠 ML Models Used

The core of the project involves building and evaluating multiple machine learning models to predict ride fares. The models include:

### 1. **Linear Regression**
- Used as a baseline model.
- Assumes linear relationship between input features and target price.
- Easy to interpret coefficients.

### 2. **Ridge Regression**
- Regularized version of linear regression.
- Helps reduce overfitting when multicollinearity exists among features.

### 3. **Lasso Regression**
- Performs feature selection by penalizing less important features.
- Useful in identifying the most influential ride features.

### 4. **Decision Tree Regressor**
- Captures nonlinear relationships in the data.
- Easy to visualize and interpret.

### 5. **Random Forest Regressor**
- Ensemble model of decision trees.
- Handles feature interactions well.
- Better generalization and robustness than a single tree.

### 6. **Gradient Boosting Regressor**
- Builds trees sequentially and minimizes prediction error iteratively.
- Strong performance in terms of RMSE and R².

---

## 📈 Model Evaluation Metrics

Each model was evaluated using:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Square Error)**
- **R² Score (Coefficient of Determination)**

Results showed that **Gradient Boosting** and **Random Forest** consistently outperformed linear models in capturing pricing patterns.

---

## 🧪 How It Works

1. **Data Preprocessing**
   - Handle categorical time slots
   - Normalize continuous features
   - Train-test split

2. **Model Training**
   - Train multiple regressors
   - Use GridSearchCV for hyperparameter tuning

3. **Prediction**
   - Predict fare based on current ride parameters
   - Compare predicted vs actual prices

4. **Visualization**
   - Correlation matrix
   - Feature importance (for tree-based models)
   - Error distribution plots

---


## ✅ Key Takeaways

- ML models, especially **tree-based ensemble models**, are effective in simulating dynamic pricing.
- Feature importance shows `distance_km`, `demand_level`, and `duration_min` are major contributors.
- Can be extended into a real-time pricing API or dashboard.

---

## 📌 Future Improvements

- Integrate **live ride data APIs**
- Add **geo-location clustering** for region-wise pricing
- Develop **real-time dashboard** using Streamlit or Flask
- Incorporate **surge multipliers** based on peak hours and event triggers

---

## 👩‍💻 Author

**Keerti Damani**  
B.Tech CSE @ VIT | Data Science @ IIT Madras  

---

## 📜 License

MIT License – use, modify, and contribute freely.


