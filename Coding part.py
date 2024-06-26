import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Load the datasets
access_electricity_df = pd.read_csv('Access to Electricity (% of Population).csv', skiprows=4)
co2_emissions_df = pd.read_csv('CO2 Emissions (Metric Tons per Capita).csv', skiprows=4)
electric_power_df = pd.read_csv('Electric Power Consumption (kWh per Capita).csv', skiprows=4)
total_energy_use_df = pd.read_csv('Total Energy Use (kg of Oil Equivalent per Capita).csv', skiprows=4)

# Drop unnecessary columns
for df in [access_electricity_df, co2_emissions_df, electric_power_df, total_energy_use_df]:
    df.drop(columns=['Unnamed: 66', 'Unnamed: 67', 'Unnamed: 68'], inplace=True, errors='ignore')

# Forward fill missing values
for df in [access_electricity_df, co2_emissions_df, electric_power_df, total_energy_use_df]:
    df.fillna(method='ffill', inplace=True)

# Ensure numeric data for the relevant columns (1990-2020)
years = list(map(str, range(1990, 2021)))
for df in [access_electricity_df, co2_emissions_df, electric_power_df, total_energy_use_df]:
    df[years] = df[years].apply(pd.to_numeric, errors='coerce')

# Function to plot time series data
def plot_time_series(df, title, ylabel, countries):
    df_filtered = df[df['Country Name'].isin(countries)]
    df_filtered = df_filtered.set_index('Country Name')[years].T
    df_filtered.index = pd.to_numeric(df_filtered.index, errors='coerce')
    df_filtered = df_filtered.apply(pd.to_numeric, errors='coerce')
    df_filtered.dropna(how='all', inplace=True)
    
    plt.figure(figsize=(14, 8))
    for column in df_filtered.columns:
        plt.plot(df_filtered.index, df_filtered[column], label=column, marker='o', linestyle='-', linewidth=2)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# Selected countries for visualization
selected_countries = ['United States', 'China', 'India', 'Brazil']

# Plot access to electricity
plot_time_series(access_electricity_df, 'Access to Electricity (% of Population) Over Time', 'Access to Electricity (%)', selected_countries)

# Plot CO2 emissions
plot_time_series(co2_emissions_df, 'CO2 Emissions (Metric Tons per Capita) Over Time', 'CO2 Emissions (Metric Tons per Capita)', selected_countries)

# Plot electric power consumption
plot_time_series(electric_power_df, 'Electric Power Consumption (kWh per Capita) Over Time', 'Electric Power Consumption (kWh per Capita)', selected_countries)

# Plot total energy use
plot_time_series(total_energy_use_df, 'Total Energy Use (kg of Oil Equivalent per Capita) Over Time', 'Total Energy Use (kg of Oil Equivalent per Capita)', selected_countries)

# Combine the datasets for correlation analysis
combined_df = access_electricity_df.merge(co2_emissions_df, on=['Country Name', 'Country Code'], suffixes=('_access', '_co2')).merge(
    electric_power_df, on=['Country Name', 'Country Code']).merge(
    total_energy_use_df, on=['Country Name', 'Country Code'], suffixes=('_electric', '_energy'))

# Select a few key years for correlation analysis to reduce clutter
key_years = ['1990', '2000', '2010', '2020']
key_cols = [f"{year}_access" for year in key_years] + [f"{year}_co2" for year in key_years] + [f"{year}_electric" for year in key_years] + [f"{year}_energy" for year in key_years]

# Extract the numeric data for correlation for key years
combined_numeric = combined_df.set_index(['Country Name', 'Country Code'])[key_cols]

# Ensure no non-numeric columns are present
combined_numeric = combined_numeric.apply(pd.to_numeric, errors='coerce')

# Calculate the correlation matrix
correlation_matrix = combined_numeric.corr()

# Enhanced visualization of the correlation matrix
plt.figure(figsize=(18, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, linecolor='black', cbar_kws={'shrink': 0.75, 'aspect': 30, 'pad': 0.02})
plt.title('Correlation Between these Indicators By Heatmap', fontsize=20, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()
plt.show()

# Summary statistics for each dataset
access_electricity_summary = access_electricity_df[key_years].describe()
co2_emissions_summary = co2_emissions_df[key_years].describe()
electric_power_summary = electric_power_df[key_years].describe()
total_energy_use_summary = total_energy_use_df[key_years].describe()

# Display summary statistics with a nice heading using tabulate
print("\nSummary Statistics for Access to Electricity (% of Population)\n")
print(tabulate(access_electricity_summary, headers='keys', tablefmt='pretty'))

print("\nSummary Statistics for CO2 Emissions (Metric Tons per Capita)\n")
print(tabulate(co2_emissions_summary, headers='keys', tablefmt='pretty'))

print("\nSummary Statistics for Electric Power Consumption (kWh per Capita)\n")
print(tabulate(electric_power_summary, headers='keys', tablefmt='pretty'))

print("\nSummary Statistics for Total Energy Use (kg of Oil Equivalent per Capita)\n")
print(tabulate(total_energy_use_summary, headers='keys', tablefmt='pretty'))
