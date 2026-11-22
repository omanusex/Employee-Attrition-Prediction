# Predicting Employee Attrition Using Machine Learning

Coursework 2 – Mining and Predictive Analytics (COMP 30044)

This project predicts whether an employee will leave the company (Attrition = Yes/No) using the
**IBM HR Analytics Employee Attrition & Performance** dataset. It follows the **CRISP-DM** methodology
and compares **Logistic Regression** and **Random Forest**.

## Project structure

```
├── Employee_Attrition_Prediction.ipynb   # EDA, preprocessing, modelling, evaluation
├── app.py                                # Streamlit app for predictions
├── data/                                 # put the Kaggle CSV here
├── figures/                              # graphs saved by the notebook
├── model/                                # trained model + results table
├── requirements.txt
├── Dockerfile
└── entrypoint.sh
```

## Dataset

- Source: [Kaggle – IBM HR Analytics Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- 1,470 employees, 35 columns, target = `Attrition`
- Download `WA_Fn-UseC_-HR-Employee-Attrition.csv` and place it in the `data/` folder.

## How to run

### 1. Install the libraries
```bash
pip install -r requirements.txt
```

### 2. Run the notebook
Open `Employee_Attrition_Prediction.ipynb` in Jupyter / VS Code and run all cells.
This creates the graphs in `figures/` and the model in `model/`.

### 3. Run the app
```bash
streamlit run app.py
```
Then open http://localhost:8501

### Run with Docker (optional)
```bash
docker build -t attrition-app .
docker run -p 8501:8501 attrition-app
```
The container trains the model automatically (if needed) and starts the app.

## Steps
1. **Data understanding** – dataset description and EDA (heatmap, boxplots, regression plots, Pearson r and p-values)
2. **Data preparation** – feature removal, outlier capping (IQR), label/one-hot encoding, StandardScaler
3. **Modelling** – Logistic Regression and Random Forest, GridSearchCV with 5-fold stratified CV, `class_weight="balanced"`
4. **Evaluation** – Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix, ROC curve
5. **Deployment** – Streamlit app

## Libraries
pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, joblib, streamlit
