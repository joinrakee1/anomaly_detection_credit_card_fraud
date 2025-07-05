# Anomaly Detection in Credit Card Transactions - Unsupervised Learning Project

This project uses unsupervised anomaly detection techniques—Isolation Forest, One-Class SVM, and Autoencoder—to identify fraudulent transactions in credit card data. The task involves building models capable of detecting rare and unusual patterns in highly imbalanced data without relying on labeled training data.

This project was completed as part of a **machine learning course assignment** using the **Kaggle Credit Card Fraud Detection dataset**.

## Files

**Included in this repository:**
- `anomaly_detection_credit_card_fraud.ipynb`: Jupyter notebook containing exploratory data analysis, data preprocessing, model building for Isolation Forest, One-Class SVM, and Autoencoder, evaluation, and performance comparison.
- `anomaly_detection_credit_card_fraud.pdf`: PDF version of the full notebook run including EDA, model training, evaluation, and final results — ideal for quick review without setting up the environment.
- `model_predictions.csv`: Contains transaction IDs, predicted fraud labels (1 = fraud, 0 = genuine), and anomaly scores from Isolation Forest, One-Class SVM and Autoencoder models.

## Models

Three unsupervised models were trained and compared:
1. **Isolation Forest**: Ensemble method isolating anomalies based on random splits; fast and effective on high-dimensional data.
2. **One-Class SVM**: Support vector machine variant modeling the boundary of normal data to detect outliers.
3. **Autoencoder**: Neural network trained to reconstruct input data; anomalies detected by high reconstruction error.

## Results

The **One-Class SVM** achieved the highest precision (0.10) and F1 score (0.16), indicating better accuracy in identifying fraud cases with fewer false positives compared to the other models. The **Isolation Forest** showed the best recall (0.81) and ROC AUC (0.94), suggesting it detects the largest proportion of frauds while maintaining strong overall discrimination. The **Autoencoder** performed lower in precision and F1 score but maintained a competitive ROC AUC (0.93) and moderate recall (0.68). Overall, the One-Class SVM strikes a balance with the highest precision and F1 score, while Isolation Forest excels in recall and overall detection capability.

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
