# Credit Card Fraud Detection with Unsupervised Models

This project uses unsupervised anomaly detection techniques—Isolation Forest, One-Class SVM, and Autoencoder—to identify fraudulent transactions in credit card data. The task involves building models capable of detecting rare and unusual patterns in highly imbalanced data without relying on labeled training data.

This project was completed as part of a **machine learning course assignment** using the **Kaggle Credit Card Fraud Detection dataset**.

## Files

**Included in this repository:**
- `credit_card_fraud_unsupervised.ipynb`: Jupyter notebook containing exploratory data analysis, data preprocessing, model building for Isolation Forest, One-Class SVM, and Autoencoder, evaluation, and performance comparison.
- `credit_card_fraud_unsupervised.pdf`: PDF version of the full notebook run including EDA, model training, evaluation, and final results — ideal for quick review without setting up the environment.

## Models

Three unsupervised models were trained and compared:
1. **Isolation Forest**: Ensemble method isolating anomalies based on random splits; fast and effective on high-dimensional data.
2. **One-Class SVM**: Support vector machine variant modeling the boundary of normal data to detect outliers.
3. **Autoencoder**: Neural network trained to reconstruct input data; anomalies detected by high reconstruction error.

## Results

The **Isolation Forest** model showed the highest recall and ROC AUC, excelling in detecting the majority of fraud cases. The **Autoencoder** achieved the best precision and F1 score, indicating better accuracy with fewer false positives. The **One-Class SVM** performed moderately across all metrics but trailed behind the other models.

Performance metrics, ROC curves, and detailed comparisons are included in the notebook for comprehensive analysis.

## Requirements

- Python 3.8 or higher  
- TensorFlow / Keras  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  

Install dependencies with:

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
