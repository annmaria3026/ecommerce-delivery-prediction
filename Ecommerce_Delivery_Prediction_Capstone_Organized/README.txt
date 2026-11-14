E-commerce Product Delivery Prediction Capstone

Folder guide
- data/: original source dataset
- notebooks/: analysis, preprocessing, model comparison, and artifact-generation notebook
- app/: Streamlit app and the model/data files it loads
- dashboard/: Power BI dashboard file
- reports/: final presentation
- evaluation/: saved test-set predictions for the four compared models

Run the app
1. Install Python packages: streamlit, pandas, scikit-learn, and joblib.
2. Open a terminal in the app/ folder.
3. Run: streamlit run app.py

Notebook paths
The notebook locates data/E_Commerce.csv whether launched with the project root or notebooks/ as the working directory. Generated app artifacts are written into app/, and test predictions are written into evaluation/.
