import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("unemployment analysis.csv")

# Convert wide format to long format
data_long = data.melt(
    id_vars=["Country Name", "Country Code"],
    var_name="Year",
    value_name="Unemployment Rate"
)

# Clean data
data_long["Year"] = pd.to_numeric(data_long["Year"], errors="coerce")
data_long = data_long.dropna()

# Select India data
india_data = data_long[data_long["Country Name"] == "India"]

# Graph 1 - Overall trend
plt.figure()
plt.plot(india_data["Year"], india_data["Unemployment Rate"])
plt.xlabel("Year")
plt.ylabel("Unemployment Rate")
plt.title("Unemployment Trend in India")
plt.show()

# Graph 2 - Covid impact
covid_data = india_data[india_data["Year"] >= 2020]

plt.figure()
plt.plot(covid_data["Year"], covid_data["Unemployment Rate"])
plt.xlabel("Year")
plt.ylabel("Unemployment Rate")
plt.title("Impact of Covid-19 on Unemployment in India")
plt.show()