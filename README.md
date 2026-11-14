# E-commerce Product Delivery Prediction

A capstone project that predicts whether an e-commerce order will arrive on time or late, using machine learning.

## What's inside
- `data/` - original source dataset
- `notebooks/` - analysis, preprocessing, and model comparison
- `app/` - Streamlit app and the saved model it uses
- `dashboard/` - Power BI dashboard
- `reports/` - final presentation
- `evaluation/` - saved test-set predictions for the four models

## Models compared
Decision Tree, Random Forest, Logistic Regression, and K-Nearest Neighbors.
Random Forest had the best accuracy (about 65.6%).

## Run the app
1. Install the packages: `pip install streamlit pandas scikit-learn joblib`
2. Open a terminal in the `app/` folder
3. Run: `streamlit run app.py`

## Author
Ann Maria Rebeiro
