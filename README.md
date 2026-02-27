# 🏠 HOUSE PRICE PREDICTION - MACHINE LEARNING PROJECT

## 📌 PROJECT OVERVIEW

This project predicts the selling price of a house based on various features such as area, number of bedrooms, bathrooms, location, and amenities.

It is a **Supervised Machine Learning Regression Problem** because:

* The dataset contains labeled values (SalePrice)
* The output is a continuous numerical value

The goal is to build a reliable model that can estimate property prices accurately for unseen data.

---

## 🎯 OBJECTIVES

* Understand housing dataset
* Clean and preprocess data
* Engineer useful features
* Train multiple regression models
* Evaluate performance using proper metrics
* Select the best model for prediction

---

## ✅ PROBLEM STATEMENT

In the real estate industry, accurately determining the price of a house is a challenging task. Property prices are often estimated based on manual judgment, market assumptions, or limited comparisons, which can lead to incorrect pricing decisions.

Without a data-driven pricing system:

Houses may be overpriced or underpriced

Buyers struggle to evaluate fair property value

Real estate agencies face difficulty in decision making

The goal of this project is:

👉 **To predict house selling prices based on property characteristics using supervised machine learning techniques**.

By building a house price prediction model, stakeholders can:

--> Estimate accurate market value of properties
--> Support data-driven real estate decisions
--> Help buyers and sellers make informed pricing choices
--> Improve efficiency in property valuation

---

## 📊 DATASET USED

The project uses three files:

| File                  | Purpose                      |
| --------------------- | ---------------------------- |
| train.csv             | Used to train the model      |
| test.csv              | Used to predict house prices |
| sample_submission.csv | Format reference for output  |

**Target Column:** SalePrice

**Features include:**

* Area (square feet)
* Bedrooms
* Bathrooms
* Location
* Amenities
* Property characteristics

---

## ⚙️ PROJECT APPROACH

The House Price Prediction project follows a systematic machine learning workflow to build an accurate predictive model for estimating property prices.

**Step 1: Data Understanding**

* Loaded the housing dataset
* Examined dataset structure and feature descriptions
* Identified independent variables and target variable (SalePrice)
* Understood numerical and categorical features

**Step 2: Exploratory Data Analysis (EDA) & Correlation Analysis**

* Analyzed distribution of house prices
* Visualized feature relationships
* Generated correlation heatmap
* Identified important variables influencing price

**Step 3: Data Cleaning**

* Handled missing values using suitable imputation techniques
* Encoded categorical features into numerical format
* Removed irrelevant columns
* Prepared clean dataset for modeling

**Step 4: Train-Test Split**

* Split dataset into training and testing data
* Applied scaling when required
* Ensured unbiased model validation

**Step 5: Model Training**

Two regression models were trained: 
🔹  Linear Regression
* Used as a baseline model
* Learned linear relationships between features and house price

🔹 Random Forest Regressor
* Tree-based ensemble model
* Captured complex non-linear relationships
* Improved prediction performance

**Step 6: Model Evaluation & Comparison**

* Models were evaluated using:
* Root Mean Squared Error (RMSE)
* R² Score
* Feature Importance Analysis
* The best-performing model was selected based on evaluation results.

**Step 7: Final Prediction Model**

* Selected optimized model
* Generated house price predictions
* Prepared model for real-world usage

---

## BLOCK DIAGRAM

![House Price Prediction Block Diagram](Images/House_Price_Block_Diagram.png)


