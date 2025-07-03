


from gc import collect; 
from warnings import filterwarnings; 

import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
import seaborn as sns 

plt.style.use("fivethirtyeight")

from datetime import datetime  

from scipy import stats 

filterwarnings('ignore'); 
from IPython.display import display_html, clear_output; 


clear_output();
print();
collect();



try:
    
    df = pd.read_csv('Walmart_Sales.csv')
    print("Dataset loaded successfully.")
    
except FileNotFoundError:
    
    print("Error: File not found. Please check the file path.")

except Exception as e:
    
    print("An error occurred while loading the dataset:", e)

print();
collect();

df.columns

# Rename columns to lowercase
df.columns = df.columns.str.lower()

# Verify the new column names
print("\nNew column names:")
print(df.columns)

print();
collect();

# checking for null values in the dataset
print(df.isna().sum())
# information about dataset
df.info()

# checking if the date format is correct
try:
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    print("All values in 'date' column are valid dates.")
except ValueError as e:
    print("Error:", e)
    print("There are non-date values present in the 'date' column.")

# Checking the duplicate values in the data
duplicate_values=df.duplicated().sum()
print(f'The data contains {duplicate_values} duplicate values')

# Checking the data shape
print(f'The dataset contains {df.shape[0]} rows and {df.shape[1]} columns')

# Statistics about the data set
all_stats = df.describe(include='all')
print(all_stats)

# Function to map dates to seasons
def date_to_season(date):
    # Extract month from date string and convert it to integer
    month = datetime.strptime(date, "%Y-%m-%d").month
    
    # Map months to seasons
    if month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn"
    else:
        return "Winter"
    
# Apply date_to_season function to create a new column
df["season"] = df["date"].apply(lambda x: date_to_season(x.strftime("%Y-%m-%d")))

# Extract year from the 'date' column
df['year'] = df['date'].dt.year

# Extract month from the 'date' column
df['month'] = df['date'].dt.month

# Extract month name from the 'date' column
df['month_name'] = df['date'].dt.month_name()

# Extract day from the 'date' column
df['day'] = df['date'].dt.day

# Extract day of the week (0 = Monday, 1 = Tuesday, ..., 6 = Sunday) from the 'date' column
df['day_of_week'] = df['date'].dt.dayofweek

df.sample(5)

# List of features
features = ['weekly_sales', 'temperature', 'fuel_price', 'cpi', 'unemployment']

# Set the figure size
plt.figure(figsize=(18, 20))

# Loop through each column in your dataset
for i, col in enumerate(features):
    # Create subplots
    plt.subplot(3, 3, i+1)
    
    # Plot histogram for the current column
    sns.histplot(data=df, x=col, kde=True)

plt.tight_layout()
plt.show()

correlation_matrix = df[['weekly_sales', 'temperature', 'fuel_price', 'cpi', 'unemployment']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

pd.pivot_table(data = df,
              index = 'year',
              columns = 'season',
              values = 'weekly_sales',
              aggfunc = 'sum')

# Create a dictionary to store season-wise weekly sales for each year
seasonwise_weekly_sales = {}

# Iterate over unique seasons
for season in df['season'].unique():
    # Group by year and sum the weekly sales for the current season
    season_sales = df[df['season'] == season].groupby('year')['weekly_sales'].sum()
    # Store the season-wise weekly sales in the dictionary
    seasonwise_weekly_sales[season] = season_sales

# Create an empty list to store data
plot_data = []

# Populate the list with data
for season, sales in seasonwise_weekly_sales.items():
    for year, weekly_sales in sales.items():
        plot_data.append({'Year': year, 'Season': season, 'Weekly Sales': weekly_sales})

# Convert the list to a DataFrame
plot_data = pd.DataFrame(plot_data)

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the stacked bar plot using seaborn
sns.barplot(data=plot_data, x='Year', y='Weekly Sales', hue='Season', ax=ax, ci=None)

# Set labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Total Weekly Sales')
ax.set_title('Yearly Season-wise Total Weekly Sales')
# Adjust legend position to prevent it from going outside the plot
ax.legend(title='Season', loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()

# Group by year and holiday_flag to get counts
holiday_counts = df.groupby(['year', 'holiday_flag']).size().unstack(fill_value=0).reset_index()

# Melt DataFrame to long format
holiday_counts_melted = pd.melt(holiday_counts, id_vars='year', var_name='Holiday Flag', value_name='Count')

# Plot using Seaborn
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Bar plot
sns.barplot(data=holiday_counts_melted, x='year', y='Count', hue='Holiday Flag', ax=ax[0])
ax[0].set_title('Holiday Distribution Over the Years')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('Count')

# Get legend handles
handles, _ = ax[0].get_legend_handles_labels()

ax[0].legend(handles=handles, labels=['Not Holiday', 'Holiday'], title='Holiday Flag', loc='upper left', bbox_to_anchor=(1, 1))

ax[1].pie(df['holiday_flag'].value_counts().values, labels=['Not Holiday', 'Holiday'], autopct='%1.2f%%')
ax[1].set_title('Overall Holiday Distribution')

plt.tight_layout()
plt.show()

# Calculate the count of each year
year_counts = df['year'].value_counts()

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Countplot for the distribution of years
sns.countplot(data=df, x='year', ax=ax[0])
ax[0].set_xlabel('Year')
ax[0].set_ylabel('Count')

# Pie chart for the distribution of years
ax[1].pie(year_counts.values, labels=year_counts.index, autopct='%1.2f%%')

# Set a single title for the entire figure
plt.suptitle('Distribution of Years', fontsize=16, y=1.05)

plt.show()