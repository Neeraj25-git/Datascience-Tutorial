import numpy as np
import pandas as pd


file_path = "Advertising.csv" 
df = pd.read_csv(file_path)

X = df[['TV', 'newspaper', 'radio']].values  
y = df['sales'].values  

n = len(y)

X = np.column_stack((np.ones(n), X))

p = X.shape[1] - 1

beta = np.linalg.inv(X.T @ X) @ X.T @ y #X.T @ X(matrix multi)   np.linalg.inv(inverse the given matrix)      @ X.T @ y(get beta coeff)

y_pred = X @ beta

RSS = np.sum((y - y_pred) ** 2)

y_mean = np.mean(y)
TSS = np.sum((y - y_mean) ** 2)

R2 = 1 - (RSS / TSS)

RSE = np.sqrt(RSS / (n - p - 1))

F_stat = ((TSS - RSS) / p) / (RSS / (n - p - 1))

print("Residual Standard Error (RSE):", RSE)
print("R² (coefficient of determination):", R2)
print("F-statistic:", F_stat)


#RSE of 1.68 means the predicted sales differ by 1.68 from the actual value
#R2 of 0.897 means 89.7% variation is explained by TV Radio and Newspaper
#F-stat of 570.27 suggests strong relationship between the budget and the sales