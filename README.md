# Bank Churn Intelligence

A Streamlit dashboard for predictive modeling and risk scoring using `European_Bank.csv`.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard includes:

- Overview metrics for the scored population and held-out model performance
- Churn probability distribution
- Permutation-based feature importance
- Customer-level churn risk calculator
- What-if simulator for activity and product changes

The app removes identifiers, adds balance-to-salary, product density, engagement-product, and age-tenure interaction features, then trains a balanced random forest on a stratified 80/20 split. A logistic regression pipeline is also trained as an interpretability baseline.
