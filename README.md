# Heart Disease Classification Using Machine Learning

## Overview

This project leverages machine learning techniques to classify whether a person is likely to have heart disease based on specific medical attributes. By analyzing a dataset of patient records, the model can predict the likelihood of heart disease with a high degree of accuracy, making it a valuable tool for early diagnosis and prevention.

## Features
- Developed a Logistic Regression-based model achieving **88.5% accuracy** on test data.
- Explored and compared multiple algorithms (Logistic Regression, KNN, Random Forest) to identify the most optimal solution.
- Enhanced model robustness by tuning hyperparameters and employing cross-validation techniques.
- Provided insightful data visualizations to interpret feature importance and correlations.
- Built using Python libraries such as **Scikit-learn**, **Pandas**, **Matplotlib**, and **Seaborn**.

## Dataset
The project uses a publicly available heart disease dataset, which contains 13 key medical attributes of patients, such as:
- **Age**
- **Sex**
- **Chest Pain Type**
- **Resting Blood Pressure**
- **Cholesterol Levels**
- **Maximum Heart Rate Achieved**
- And others...

## Project Workflow
1. **Data Exploration**:
   - Cleaned and preprocessed the dataset to handle missing values and outliers.
   - Performed Exploratory Data Analysis (EDA) to understand feature relationships and distributions.

2. **Feature Engineering**:
   - Encoded categorical features and scaled numerical data for better model performance.
   - Analyzed feature importance to focus on the most critical factors influencing predictions.

3. **Model Development**:
   - Implemented Logistic Regression, KNN, and Random Forest classifiers.
   - Selected the Logistic Regression model as the best performer based on accuracy and F1 score.
   - Optimized the Logistic Regression model using grid search for hyperparameter tuning.

4. **Model Evaluation**:
   - Evaluated the model using metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.
   - Achieved an F1 score of **0.87**, ensuring reliable predictions.

5. **Visualization**:
   - Created plots to illustrate feature distributions, correlation matrices, and model performance.

## Installation and Usage
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Jupyter Notebook (or Google Colab)
- Required Python libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

### Steps to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/MrMaheshwari11/Heart-Disease-Classification.git

2. Navigate to the project directory:
   ```bash
   Navigate to the project directory:
   
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook end-to-end-heart-disease-classification.ipynb

5. Follow the steps in the notebook to run the code and observe the outputs.

## Results
- **Accuracy**: 88.5%
- **F1 Score**: 0.87
- **Key Insights**: Features such as chest pain type, maximum heart rate, and exercise-induced angina are significant predictors of heart disease.

## Key Takeaways
This project demonstrates the power of machine learning in healthcare by building an accurate model for heart disease prediction. With further development, it could be extended into a full-fledged diagnostic tool.

## Future Scope
- Integrate more advanced machine learning models like Gradient Boosting or Neural Networks.
- Expand the dataset with more diverse records for generalization.
- Develop a web application to make the tool more accessible to healthcare professionals.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For any queries or suggestions, feel free to reach out:
- **Name**: Manishkumar Maheshwari  
- **Email**: [manish1111maheshwari@gmail.com](mailto:manish1111maheshwari@gmail.com)  
- **GitHub**: [MrMaheshwari11](https://github.com/MrMaheshwari11)

