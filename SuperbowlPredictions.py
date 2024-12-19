import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/Neel/Desktop/2023-2024/Spring 2024/Big Data and Forecasting/SuperbowlHW.csv")
#print(data)

teams_2024 = data.head(2)
X_2024_data = teams_2024.drop(['Winner','Team','Year'], axis = 1)

#Remove 2024 teams from current data cells
data = data.dropna()

#Preparing data(Remove all 'useless' info/variables) and make your dependent variable (Winner)
X_data = data.drop(['Winner','Team','Year'], axis = 1)
#print(X)
Y_data = data['Winner']
#print(Y)
scale = StandardScaler()
X = scale.fit_transform(X_data)
X_2024 = scale.transform(X_2024_data)
alphas = [10, 20, 25, 50, 100, 200, 500, 1000, 2000]
reg_rid = RidgeCV(alphas = alphas, store_cv_values = True)

#Set the training data
model = reg_rid.fit(X, Y_data)

# Get the best alpha selected through cross-validation
best_alpha = reg_rid.alpha_
cv_errors_mean = np.mean(reg_rid.cv_values_, axis=0)
plt.plot(alphas, cv_errors_mean, marker='o')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Mean Error')
plt.title('Alpha vs. Mean Error')
plt.show()

print("Best alpha:", best_alpha)

#Calculate error for the predicted values
Pred_y = model.predict(X)
ridge_mse = np.sqrt(mean_squared_error(Y_data, Pred_y))

#Predicted values for Superbowl 2024
y_2024 = model.predict(X_2024)

#Standardize prediction values for Superbowl 2024
prob_2024 = np.exp(y_2024)/ np.sum(np.exp(y_2024))


# Create a DataFrame to associate teams with probabilities
teams_prob_df = pd.DataFrame({'Team': teams_2024['Team'], 'Probability': prob_2024})
print(teams_prob_df)

