#!/bin/sh
set -e

# 1) train the model if it does not exist yet (runs the whole notebook)
if [ ! -f "model/attrition_model.pkl" ]; then
    if [ ! -f "data/WA_Fn-UseC_-HR-Employee-Attrition.csv" ]; then
        echo "ERROR: dataset not found. Put WA_Fn-UseC_-HR-Employee-Attrition.csv inside the data/ folder."
        exit 1
    fi
    echo "Model not found -> running the notebook to train it..."
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=1800 Employee_Attrition_Prediction.ipynb
fi

# 2) start the Streamlit app
echo "Starting Streamlit app on port 8501..."
exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true
