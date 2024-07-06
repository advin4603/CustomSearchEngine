import csv
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

words = []
frequencies = []
probabilities = []
theoretical_values = []

with open("metrics.csv", mode="r") as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        words.append(row["Word"])
        frequencies.append(int(row["Frequency"]))
        probabilities.append(float(row["Probability"]))
        theoretical_values.append(float(row["Theoretical Value"]))

ranks = list(range(1, len(words) + 1))
inverse_ranks = [1 / rank for rank in ranks]

X_rank = np.array(ranks).reshape(-1, 1)
Y = np.array(probabilities).reshape(-1, 1)
X_rank = sm.add_constant(X_rank)  # Add a constant for the intercept
model_rank = sm.OLS(Y, X_rank).fit()

X_inverse_rank = np.array(inverse_ranks).reshape(-1, 1)
X_inverse_rank = sm.add_constant(X_inverse_rank)
model_inverse_rank = sm.OLS(Y, X_inverse_rank).fit()

residuals_rank = model_rank.resid
residuals_inverse_rank = model_inverse_rank.resid

plt.figure(figsize=(12, 6))
plt.scatter(ranks, probabilities, label="Probability", s=10)
plt.scatter(ranks, theoretical_values, label="Theoretical Value", s=10)
plt.plot(ranks, model_rank.fittedvalues, label="Linear Regression (Rank)", color='blue')
plt.xlabel("Rank")
plt.ylabel("Probability / Theoretical Value")
plt.legend()

r_squared_rank = model_rank.rsquared
p_value_rank = model_rank.f_pvalue
plt.text(0.1, 0.9, f"R-squared (Rank) = {r_squared_rank:.4f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.85, f"p-value (Rank) = {p_value_rank:.4f}", transform=plt.gca().transAxes)

plt.figure(figsize=(12, 6))
plt.scatter(inverse_ranks, probabilities, label="Probability", s=10)
plt.scatter(inverse_ranks, theoretical_values, label="Theoretical Value", s=10)
plt.plot(inverse_ranks, model_inverse_rank.fittedvalues, label="Linear Regression (1/Rank)", color='red')
plt.xlabel("1/Rank")
plt.ylabel("Probability / Theoretical Value")
plt.legend()

r_squared_inverse_rank = model_inverse_rank.rsquared
p_value_inverse_rank = model_inverse_rank.f_pvalue
plt.text(0.1, 0.9, f"R-squared (1/Rank) = {r_squared_inverse_rank:.4f}", transform=plt.gca().transAxes)
plt.text(0.1, 0.85, f"p-value (1/Rank) = {p_value_inverse_rank:.4f}", transform=plt.gca().transAxes)

plt.figure(figsize=(12, 6))
plt.scatter(ranks, residuals_rank, s=10)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel("Rank")
plt.ylabel("Residuals (Rank)")

plt.figure(figsize=(12, 6))
plt.scatter(inverse_ranks, residuals_inverse_rank, s=10)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel("1/Rank")
plt.ylabel("Residuals (1/Rank)")

plt.show()
