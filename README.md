# DISEASE DETECTION IN CHILLI PLANTS AND REMOTE MONITORING OF AGRICULTURAL PARAMETERS



### Table of Contents

- [ABSTRACT](#abstract)
- [INTRODUCTION](#introduction)
- [EXISTING SYSTEM](#existing-system)
- [PROPOSED SYSTEM](#proposed-system)


### ABSTRACT
Agriculture, the backbone of nations, provides essential food and raw materials. Detecting chilli plant diseases is crucial due to their impact on food security. Traditional methods, often relying on visual inspection by experts, are commonly used for identifying these diseases

### Data Sources
The dataset is downloaded from the [Kaggle](https://www.kaggle.com/datasets/srolka/ecommerce-customers).Also the dataset is attached in the Repository 👉 [click here](https://github.com/Thadkapally-Saikiran/Ecommerce-Customers/blob/main/Ecommerce%20Customers.csv)

Working with the Ecommerce Customers csv file from the company. It has total 8 columns - Customer info such as Email, Address, and their color Avatar. Then it also has numerical value columns:
 * Avg. Session Length: Average session of in-store style advice sessions.
 * Time on App: Average time spent on App in minutes.
 * Time on Website: Average time spent on Website in minutes.
 * Length of Membership: How many years the customer has been a member ?
 * Yearly Amount Spent: How much amount the customer have spent ?

### Problem statement
1. Analyze customer behavior and spending patterns:
   * The goal is to understand how various customer engagement metrics (time on app, website, session length, membership duration) influence their yearly spending on an e-commerce platform
3. Identify key drivers of customer spending:
   * By visualizing the correlations between engagement metrics and spending, you aim to pinpoint which factors have the strongest positive impact on customer spending
4. Optimize customer engagement strategies:
   * The insights from the data analysis can be used to develop strategies that encourage customer retention (lengthening memberships) and increase app engagement to drive higher spending.
 
### Skills/ Concepts demonstrated

The following Python vs Machine Learning features were incorporated:
- Importing the required libraries i.e. Pandas, Numpy, Matplotlib, Seaborn, Scikit-Learn
- Reading the dataset using "Pandas"
- Exploratory Data Analysis(EDA): Understanding and Cleaning the Data
- Statistical Correlation Analysis
- Visualizing Data using Seaborn library
- Draw Insights from visual Data and conclude data driven decision upon problem-statement
- Machine Learning Model Building
   * Separating the dataset into features and target data
   * Split the data set into train and test
   * Implementing the Linear Regression Model
   * Evaluating the prediction scores.
   * Visualizing the results
   * Repeat this for remaining Models(Decision Trees, RandomForest, K-NearestNeighbour, XGBoost, Adaboost)
- Evaluating the model accuracy.
- Compare the Models Evaluation Scores, Metrics and select the best Model performence

### Visualization:

Heat-Map   
---
![](heat-map.png)  

Pair-Plot
---
![](pairplot.png)

### Recommendations

Based on the analysis, we recommend the following actions:
- Focus on improving app engagement
   * Since there seems to be a strong correlation between time spent on the app and amount of money spent, you can focus on strategies to keep users engaged with the app for longer periods of time. This could include things like gamification, personalized recommendations, and exclusive offers for app users.
- Target website visitors with retargeting campaigns
  * Since time spent on the website also seems to be correlated with spending, you can target website visitors with retargeting campaigns to encourage them to make a purchase.
- Develop strategies to retain members
  * The data suggests that longer memberships lead to higher spending. You can develop strategies to retain members, such as offering loyalty programs and exclusive benefits for long-term members.
- Develop strategies to retain existing members
  *  This could include offering loyalty programs, discounts, or exclusive content to members.
- Encourage customers to sign up for memberships
  * You could offer a free trial or discount on the first year of membership.
- Target your marketing campaigns to high-value customers
  * These are the customers who are most likely to spend a lot of money on your platform.
 
![](ecommerce_img.gif)       |     ![](ecommerce_img2.webp)   |   ![](ecommerce_img3.gif)

### References
1. Popular Machine Learning Algorithms.
   - [Kaggle](https://www.kaggle.com/code/marcovasquez/top-machine-learning-algorithms-beginner)
2. Python Docs for your reference
   - [Python Docs](https://docs.python.org/3/tutorial/)
3. [Pandas](https://pandas.pydata.org/docs/)
4. [Matplotlib](https://matplotlib.org/stable/users/getting_started/)
5. [Seaborn](https://seaborn.pydata.org/tutorial/introduction.html)
6. [scikit-learn](https://scikit-learn.org/stable/getting_started.html)

😄

💻
