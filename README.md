# MLAB (Mortgage Loan Approval ChatBot)

## The goal is to build a chatbot using Streamlit to review all the loan applications and approve loans application based on the information a user inputs in the chat user interface. 
### Loan-based businesses can benefit by using this bot because it automates the loan review process and approves applications based on a set guideline per business risk aversion scale. Individuals can also benefit from using the chatbot as a planning tool.  The chatbot can help loan applicants know what credit score they should aim for, how much down payment, and project future payments.

* Process
  * The mortgage loan approval bot utilizes machine learning techniques to automate the loan approval process, which is a crucial aspect of fintech. By leveraging machine learning models, the chatbot can analyze various factors and make predictions on loan approvals, streamlining the lending process and improving efficiency.

  * The aim is to improve the model's performance in correctly predicting the minority class and potentially increase metrics such as precision, recall, and F1-score for the minority class.

* Machine Learning Model
  * The AdaBoost (Adaptive Boosting) machine learning model for the mortgage loan approval bot. AdaBoost is a popular ensemble learning method that combines multiple weak learners to create a strong predictive model.

* Key Takeaways
  * Accuracy: AdaBoost is known for its high accuracy in classification tasks. By combining multiple weak learners, it can capture complex patterns in the data and make accurate predictions.
  * Handling Imbalanced Data: Mortgage loan approval datasets often suffer from class imbalance, where the number of approved loans is significantly different from the number of rejected loans. AdaBoost handles imbalanced data well by focusing on the misclassified samples in each boosting round, helping to mitigate the impact of class imbalance.
  * Robustness to Overfitting: AdaBoost applies a boosting approach that gives more weight to misclassified samples in subsequent iterations. This process helps to reduce overfitting by focusing on the challenging instances during training.
  * Interpretability: While AdaBoost uses an ensemble of weak learners, it can still provide insights into feature importance. By analyzing the weights assigned to each weak learner and their corresponding features, we can understand the relative importance of different factors in the loan approval process.
  * Availability and Ease of Use: AdaBoost is a widely implemented algorithm available in popular machine learning libraries such as sci-kit-learn. Its straightforward implementation and ease of use make it convenient for our project.

* AdaBoost

  * AdaBoost algorithm, short for Adaptive Boosting, is a Boosting technique used as an Ensemble Method in Machine Learning. It is called Adaptive Boosting as the weights are re-assigned to each instance, with higher weights assigned to incorrectly classified instances.
  * It is a technique in Machine Learning used as an Ensemble Method. The most common estimator used with AdaBoost is decision trees with one level which means Decision trees with only 1 split. These trees are also called Decision Stumps.
  * High Accuracy, by combining weak learners, captures complex patterns to make into a strong learner.
  * Imbalanced Data, in the initial data, the number of approved loans is higher than unapproved. Ada Boost focuses on miscalculated samples in each boosting iteration to mitigate the imbalance.
  * Robustness to Overfitting - Ada Boost gives more weight to miscalculated samples in subsequent iterations, reducing overfitting during training.
  * Interpretability: Ada Boost further analyzes the weights assigned to weak learners and features important rankings.

* Packages to Use/Import
  `Numpy as np`
  `Pandas as pd`
  `Pathlib import Path`
  `Sklearn.metrics import balanced_accuracy_score, confusion_matrix`
  `Imblearn.metrics import classification_report_imbalanced`
  `Sklearn.preprocessing import LabelEncoder, StandardScaler`
  `Sklearn.compose import ColumnTransformer`
  `Skleanr.model_selection import Train_test_split`
  `Sklean.linear_model import LogisticRegression`
  `Sklearn.ensemble import AdaBoostClassifier`
  `Sklearn.model_selection import train_test_split`
  `Sklearn.metrics import classification_report`
  `Sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score`

* Data preparation and model training process

  * Obtained the data from Kaggle: The data set is about loan approval status for different customers, including Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History, and others.
  * Data is put into Pandas Dataframe
  * Collected and preprocessed the data: dropping unnecessary columns, encoding categorical variables, handling missing values, and scaling numerical features.
  * must clean the data, look at outliers or trends, etc.
  * drop unnecessary columns or columns with non-numerical values
  * Separate the target variable (loan status) and the features (X)
  * Perform categorical encoding – use label-encoder
  * Handle missing values
  * Scale the numeric columns

* Model trained using the prepared dataset: splitting of the data into training and testing sets, fitting model using training data, making predictions on testing data, and calculating accuracy score.
  * Split the data into training and testing sets
  * Instantiate the Logistic Regression model
  * Fit the model using training data
  * Make predictions on the test data
  * Evaluate the model
  * Print the evaluation metrics
  * Balanced Accuracy
  * Confusion Matrix
  * Classification Report
  * Split your data into training and testing sets
  * Create an AdaBoost classifier
  * Train the classifier on the training data
  * Make predictions on the test data 
  * Evaluate the performance of the model
  * Classification Report
  * Make predictions on the test data
  * Calculate evaluation metrics
  * Print the evaluation metrics
  * Accuracy, Precision, Recall, F1-Score

* Challenges

  * Data Quality Issues: Missing data can significantly impact the performance of machine learning models. We had to decide on appropriate strategies to handle missing values, such as replacing missing values with column mean. 

  * Imbalanced Dataset: The number of approved loans is significantly higher than the number of rejected loans. Imbalanced data can lead to biased models that favor the majority class. We had to employ the synthetic minority oversampling technique (SMOTE) to address the class imbalance issue and ensure that the model could accurately predict loan approvals for both classes.	


* Streamlit

    It is an open-source app framework, a Python library that makes it easy to create and share custom web apps for machine learning and data science.  Streamlit makes it easy to visualize, mutate, and share data. The API reference is organized by activity type, like displaying data or optimizing performance.
    Each section includes methods associated with the activity type, including examples.
    Streamlit library includes a Get Started Guide, API reference, and more advanced core library features, including caching, theming, and Streamlit Components.
    Streamlit Community Cloud is an open and free platform for the community to deploy, discover, and share Streamlit apps and code with each other. It is as simple as creating a new app, sharing it with the community, getting feedback, iterating quickly with live code updates, and having an impact. 
    Knowledge base is a self-serve library of tips, step-by-step tutorials, and articles to answer questions about creating and deploying Streamlit apps.

  * The boosted model demonstrates improved performance compared to the previous model, with higher precision, recall, and F1-score values. This indicates that the model is better at correctly classifying positive samples while maintaining a reasonable level of correctness and balance between precision and recall.

  * Based on the boosted model, we figured out the range for an if-else algorithm in Streamlit.


* Base Inputs for the bot:
`Loan_ID- Unique Loan ID`
`Gender- Male/ Female`
`Married- Applicant married (Y/N)`
`Dependents- Number of dependents`
`Education- Applicant Education (Graduate/ Under Graduate)`
`Self_Employed- Self employed (Y/N)`
`ApplicantIncome- Applicant income`
`CoapplicantIncome- Coapplicant income`
`LoanAmount- Loan amount in thousands`
`Loan_Amount_Term- Term of loan in months`
`Credit_History- credit history meets guidelines`
`Property_Area- Urban/ Semi Urban/ Rural`
`Loan_Status- (Target) Loan approved (Y/N)`

