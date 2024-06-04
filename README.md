# Online-Shoppers-Purchase-Intention-Prediction-ML-Model


Online Shopping is a huge and growing form of purchasing which caters to huge portion of business revenue in the e-commerce sector. The client is facing the problem of low rate of sales conversion and seeks to increase revenue from its online shopping platform by converting visitors into customers on the basis of the customer’s online activities.Given clickstream and session data of a user who visits an e-commerce website, can we predict whether or not that visitor will make a purchase?

Answering this question is critical for these types of companies in order to ensure that they are able to remain profitable. This information can be used to nudge a potential customer in real-time to complete an online purchase, increasing overall purchase conversion rates.

The dataset consists of 18 features belonging to 12330 sessions of unique users.The dataset was formed so that each session would belong to a different user in a 1-year period to avoid any tendency to a specific campaign, special day, user profile, or period.


# Preliminary Business Problem Scope

Our project is focused on applying Machine Learning classification models to e-commerce website data.The Objective of this project is to build a 'Predictive Machine Learning Model' that can predict customer purchase intention as accurately as possible.This project aims to use the information of customers by tracing their web acitivity when they visit an online shopping site.We intend to develop different machine learning models and evaluate them and identify best model in predicting purchasing intention of a customer.

# Choosing Best Evaluation Metric for the Model 

Our model predicts whether a shopper intends to make a purchase or not .

True Positives (TP): These are instances where the model correctly predicts that a shopper intends to make a purchase. These are our successful predictions. 

True Negatives (TN): These are cases where the model correctly predicts that a shopper does not intend to make a purchase. These represent accurate negative predictions.

False Positives (FP): In these instances, the model wrongly predicts that a shopper intends to make a purchase when they do not. These are also known as Type I errors or 'false alarms.'

False Negatives (FN): Here, the model incorrectly predicts that a shopper does not intend to make a purchase when they actually do. These are Type II errors or 'missed opportunities.'

We have chosen f1 as best evaluation metric for our model.

Initially we have thought of using **recall** as our performance metric as it is ability to identify all relevant instances of the positive class. In the context of online shopper purchase intention, recall can be explained as follows:

**True Positives(TP)**: As mentioned earlier, these are cases where the model correctly predicts potential customers who intend to make a purchase.

**False Negatives(FN)**: These are instances where the model incorrectly predicts that a shopper does not intend to make a purchase when they actually do.

For an online retailer, high recall ensures that as many potential customers as possible are correctly identified, increasing the chances of conversion and revenue.

**Cost of having more False Negatives** :

In this case having more false negatives could lead to lost sales and revenue.The user may not receive relevant product recommendations or incentives, potentially causing them to abandon their shopping session.So we also need to better Recall for the model.

So having high recall indicates lower rate of false negatives.

------------------------------------------------------------------------------------------------------

On the other hand we felt **precision** is also important as it is ability of the model to make accurate positive predictions, out of all the positive predictions it makes.

Precision measures the accuracy of positive predictions made by the model. In the context of online shopper purchase intention, precision can be explained as follows:

**True Positives(TP)**: These are cases where the model correctly predicts that a shopper intends to make a purchase. For an online retailer, this means correctly identifying potential customers who are likely to complete a purchase.

**False Positives(FP)**: These are instances where the model wrongly predicts that a shopper intends to make a purchase when they do not. 

For an online retailer, high precision means accurately identifying and targeting potential customers to increase the chances of conversion while avoiding unnecessary promotional efforts on customers who are unlikely to make a purchase.

**Cost of having more False Postives** :

This can result in annoying or spamming customers, which may lead to dissatisfaction and even loss of potential sales if the user decides to disengage from the platform. It can also incur additional marketing costs and resource wastage.So we need to better Precision for the model.

A higher precision indicates a lower rate of false positives.

# Reasons for choosing F1 as main evaluation metric

We felt in this case both precision and recall might be of equal importance.

Therefore we have decided to stike a balance between recall and precision. We wanted a single measure that considers both the false positives and false negatives in a classification model.That is the reason why we chose 'F1-score' as our perofrmance metric.

The F1 score provides a single number that tells you how well your model is at both finding all the relevant items (high recall) and ensuring those it identified are indeed relevant (high precision). It's a way to balance the trade-off between making sure you catch as many relevant things as possible while avoiding false alarms.

# Results and Discussion Summary

Before Hyper parameter tuning :

| Description  of Model    | Accuracy | Recall   | Precision | f1-score |ROC-AUC|
|--------------------------|----------|----------|-----------|----------|-------|
| Logistic                 | 0.85     | 0.85     | 0.88      | 0.86     |0.9    |
| KNN                      | 0.73     | 0.73     | 0.86      | 0.77     | -     |
| SVM_MODEL(Linear kernel) | 0.88     | 0.88     | 0.90      | 0.88     | -     |
| SVM_MODEL(Rbf kernel)    | 0.86     | 0.86     | 0.89      | 0.87     |0.8    |
| Decision Tree            | 0.86     | 0.86     | 0.87      | 0.87     |0.78   |
| Random Forest            | 0.89     | 0.89     | 0.90      | 0.90     |0.92   |
| Ada Boost                | 0.88     | 0.88     | 0.89      | 0.88     |0.91   |
| Gradient Boost           | 0.89     | 0.89     | 0.90      | 0.90     |0.93   |
| XG Boost                 | 0.89     | 0.89     | 0.89      | 0.89     |0.92   |
| Neural Network           | 0.87     | 0.87     | 0.87      | 0.87     |0.87   |

In our Online Shoppers Purchase Intention Model, maximizing the F1 Score is crucial to avoid losing potential revenue.Its essential to balance both False Positives and False Negatives.As False positives may lead to annoying or spamming customers, which may cause dissatisfaction and even loss of potential sales if the user decides to disengage from the platform. It can also incur additional marketing costs and resource wastage.False negatives can result in missed opportunities.The user may not receive relevant product recommendations or incentives, potentially causing them to abandon their shopping session. Over time, this can impact the business's profitability and customer satisfaction.

Since we are focused on maximizing our F1 score value from the initial stage of the problem,we can observe from the above values that Gradient Boost model is giving us higher F1 score value than the other models.we are gonna consider it as our best model before hyperparameter tuning since we are mainly focused on maximizing our F1 Score.

# After Hyper parameter tuning : 


| Description               | Accuracy | Recall   | Precision | f1-score |
|---------------------------|----------|----------|-----------|----------|
| Logistic (Random Search)  | 0.86     | 0.86     | 0.89      | 0.87     |
| Logistic (Grid Search)    | 0.88     | 0.88     | 0.90      | 0.89     |
| SVM(Rbf)(RandomSearch)    | 0.84     | 0.84     | 0.81      | 0.82     |
| SVM(Rbf)(GridSearch)      | 0.84     | 0.84     | 0.81      | 0.82     |
| DecisionTree(RandomSearch)| 0.87     | 0.87     | 0.90      | 0.88     |
| DecisionTree(GridSearch)  | 0.87     | 0.87     | 0.90      | 0.88     |


We can observe that Gradient Boost Model is giving us higher F1 Score value than other models.Even thouhgh Random forest model also has same F1 Score value=0.9, Area under curve (AUC) is little bit more for Gradient Boost model.So we can choose the best prediction model as Gradient Boost in this problem context.We will deploy Gradient Boost model for production.

We chose our business objective to maintain both False Negatives and False positives we took F1 scrore as our main metric measure.However this might not be the case in every business, Based on the different Business Objectives and requirements one can choose to maximize Recall maintaining a threshold cap on Precison or Vice versa.

# Future Scope

The scope of this project is only limited to predicting customer purchase intention and evaluating and measuring the F1score of these predictions.As for the Future scope,accurately predicting customer purchase intentions can significantly enhance the development of diverse marketing strategies and potentially integrate with an E-COMMERCE WEBSITE PRODUCT RECOMMENDATION SYSTEM. 

For example, if the machine learning solution indicates a high likelihood of a customer's purchase intent, the recommendation system might suggest premium or more expensive products, as it can be inferred that the user is inclined to consider higher-quality or pricier items when their intent to purchase is strong. On the other hand, if the solution predicts a lower purchase intent, the recommendation system could propose discounted products or items with special offers, such as 'Buy one, get one free.'Furthermore, the historical data on how customer intentions change in response to these recommendations can also be studied and applied to enhance the recommendation system's effectiveness.

# Conclusion

This remarkable F1 score demonstrates the model's effectiveness in predicting whether a visitor is likely to make a purchase during their online session. It signifies the potential for businesses to significantly enhance their sales conversion rates by leveraging such predictive models to provide real-time assistance and incentives to potential customers.

In conclusion, our project aimed to address the pressing issue of low sales conversion in online shopping.We developed a predictive model with gboost that proved to be highly successful, achieving a weighted average F1 score of 0.9. By understanding and predicting customer purchase intentions, businesses can proactively engage with potential customers and optimize their online shopping experience, ultimately increasing revenue and profitability. This solution enables companies to remain competitive and thrive in the ever-growing e-commerce sector.
