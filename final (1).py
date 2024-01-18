import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy

def read_clean_transpose_data(file_path):
    """
    Reads data from a CSV file, performs basic cleaning, and returns original, cleaned, and transposed dataframes.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - original_data (pd.DataFrame): Original data read from the CSV file.
    - cleaned_data (pd.DataFrame): Data after basic cleaning.
    - transposed_data (pd.DataFrame): Transposed version of the cleaned data.
    """
    # Read the original data
    original_data = pd.read_csv(file_path)

    # Perform basic cleaning (you can customize this based on your data cleaning needs)
    cleaned_data = original_data.dropna()  # Example: Remove rows with missing values

    # Transpose the cleaned data
    transposed_data = cleaned_data.transpose()

    return original_data, cleaned_data, transposed_data


# Calculate confidence interval using err_ranges function
def err_ranges(x, params, covariance, confidence=0.95):
    """
        Calculate the confidence interval for the given x values based on the parameters and covariance from curve fitting.

        Parameters:
        - x (array-like): The x values for which to calculate the confidence interval.
        - params (array-like): Parameters obtained from curve fitting.
        - covariance (2D array): Covariance matrix obtained from curve fitting.
        - confidence (float, optional): Confidence level for the interval. Default is 0.95.

        Returns:
        - lower_bound (array-like): Lower bounds of the confidence interval for each x value.
        - upper_bound (array-like): Upper bounds of the confidence interval for each x value.
        """
    p_err = np.sqrt(np.diag(covariance))
    z_score = scipy.stats.norm.ppf((1 + confidence) / 2)
    upper_bound = simple_model(x, *params) + z_score * p_err[0]
    lower_bound = simple_model(x, *params) - z_score * p_err[0]
    return lower_bound, upper_bound


def simple_model(x, a, b):
    """
        Computes a simple linear model.

        Parameters:
        - x (array-like): Independent variable values.
        - a (float): Slope of the linear model.
        - b (float): Intercept of the linear model.

        Returns:
        - y (array-like): Dependent variable values predicted by the linear model.
        """
    return a * x + b


# Call the function to get the dataframes
data_file_path = 'DATA.csv'
original_data, cleaned_data, transposed_data = read_clean_transpose_data(data_file_path)

# Select relevant columns for clustering
columns_for_clustering = ["Forest area (% of land area)",
                          "GDP growth (annual %)",
                          "Unemployment, total (% of total)", "Water productivity, total ",
                          "Population, total ", "Net migration",
                          "Current health expenditure (% of GDP) ",
                          "Domestic general government health expenditure (% of GDP)"]

# Normalize the data
scaler = StandardScaler()
df_normalized = scaler.fit_transform(cleaned_data[columns_for_clustering])

# Apply KMeans clustering
num_clusters = 3  # Adjust as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cleaned_data["Cluster"] = kmeans.fit_predict(df_normalized)

# Calculate Silhouette Score
silhouette_avg = silhouette_score(df_normalized, cleaned_data["Cluster"])
print(f"Silhouette Score: {silhouette_avg}")

# Visualize the clusters and cluster centers
plt.scatter(cleaned_data["GDP growth (annual %)"], cleaned_data["Forest area (% of land area)"], c=cleaned_data["Cluster"], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], s=300, c='red', marker='X', label='Cluster Centers')
plt.title("Clusters of Countries")
plt.xlabel("GDP Growth (annual %)")
plt.ylabel("Forest area (% of land area)")
plt.legend()
plt.show()

# Example usage for fitting GDP growth and Forest area
x_data = cleaned_data["GDP growth (annual %)"]
y_data = cleaned_data["Forest area (% of land area)"]

params, covariance = curve_fit(simple_model, x_data, y_data)
y_fit = simple_model(x_data, params[0], params[1])

# Plot the data, the fitted curve, and confidence interval
plt.scatter(x_data, y_data, label="Original Data")
plt.plot(x_data, y_fit, label="Fitted Curve", color='red')

# Confidence interval
lower_bound, upper_bound = err_ranges(x_data, params, covariance)
plt.fill_between(x_data, lower_bound, upper_bound, color='gray', alpha=0.2, label='Confidence Interval')

plt.title("Fitting GDP Growth and Forest Area with Confidence Interval",fontsize=16,fontweight='bold')
plt.xlabel("GDP Growth (annual %)")
plt.ylabel("Forest area (% of land area)")
plt.legend()
plt.show()

# Predict for the year 2023 for all countries
x_predict = np.linspace(min(x_data), max(x_data), num=len(cleaned_data))
y_predict = simple_model(x_predict, params[0], params[1])

# Plot the prediction for 2023
plt.scatter(x_data, y_data, label="Original Data")
plt.plot(x_predict, y_predict, label="Prediction for 2023", color='green', linestyle='--')
plt.title("Prediction for 2023 - GDP Growth and Forest Area",fontsize=16,fontweight='bold')
plt.xlabel("GDP Growth (annual %)")
plt.ylabel("Forest area (% of land area)")
plt.legend()
plt.show()
