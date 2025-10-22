# SleepWell-ML
Machine Learning model to predict sleep quality using health and lifestyle data — includes preprocessing, model comparison, and 10-fold cross-validation.

💤 Sleep Disorder Prediction using Machine Learning
🧠 Overview

This project aims to predict whether a person is likely to have a sleep disorder based on health and lifestyle factors such as sleep duration, physical activity level, stress level, BMI, heart rate, and daily steps.
By analyzing these patterns, the model helps identify individuals who may be at risk of poor sleep quality — a key indicator of overall health.

📂 Dataset

The dataset contains various health-related and lifestyle features.
Key columns include:

Sleep Duration — Average sleep hours per night

Physical Activity Level — Daily activity score

Stress Level — Self-reported stress measure

BMI Category — Underweight, Normal, Overweight, or Obese

Heart Rate — Average heart rate per day

Daily Steps — Number of steps per day

Occupation — Simplified into key sectors (e.g., Healthcare, Management, Finance, etc.)

Gender — Encoded as binary

Sleep Quality Category — Target variable (good / mid)

⚙️ Data Preprocessing

Before training, the following preprocessing steps were applied:

Handled Missing Values
Cleaned the dataset to remove or fill missing entries.

Categorical Encoding
Used one-hot encoding (pd.get_dummies) for categorical columns like occupation and gender.

Normalization
Applied feature scaling using StandardScaler to ensure all numerical features are on the same scale.

Data Splitting
Divided data into training and testing sets for unbiased evaluation.

🧩 Model Development

Several machine learning algorithms were tested and compared for best performance, including:

Logistic Regression

Random Forest Classifier

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Decision Tree Classifier

After evaluation, the model with the best accuracy and generalization was selected as the final model.

📊 Model Evaluation

The models were evaluated using:

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-Score)

You can visualize these results to interpret how well the model performs in distinguishing between good and mid sleep quality.

📈 Results

The best-performing model achieved high accuracy on the test set.

Feature importance analysis showed that Stress Level, Sleep Duration, and Physical Activity were among the most influential factors.

The model demonstrates potential for health monitoring systems and wellness analytics.

🚀 Technologies Used

Python 3

Pandas, NumPy — Data handling

Matplotlib, Seaborn — Visualization

Scikit-learn — Model building and evaluation

💬 Future Improvements

Add deep learning models (e.g., ANN) for comparison

Integrate real-time sleep tracking data

Build a small web interface for user input and predictions

🤝 Contributing

Contributions are welcome! Feel free to fork this repository, improve model performance, or enhance visualizations.

📜 License

This project is released under the MIT License.
