import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Excel file
file_path = 'Region.xlsx'
#file_path = 'D:\\Data Analysis\\Region.xlsx'  # Replace with your file path
sheet_name = 'Sheet1'    # Replace with your sheet name if needed
df = pd.read_excel(file_path, sheet_name=sheet_name)
print(df)
print()

#Trend Analysis
# Aggregate by Date
overall_trends = df.groupby('Date').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()

# Line plot for Sales and Profit
plt.figure(figsize=(10, 6))
plt.plot(overall_trends['Date'], overall_trends['Sales'], label='Sales', marker='o', color='skyblue')
plt.plot(overall_trends['Date'], overall_trends['Profit'], label='Profit', marker='s', color='orange')
plt.title('Overall Trends: Sales and Profit')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
print()

# Group by Date and Region
regional_trends = df.groupby(['Date', 'Region']).agg({'Sales': 'sum'}).reset_index()

# Line plot for Sales by Region
plt.figure(figsize=(10, 6))
for region in regional_trends['Region'].unique():
    region_data = regional_trends[regional_trends['Region'] == region]
    plt.plot(region_data['Date'], region_data['Sales'], label=f'Region {region}', marker='o')

plt.title('Regional Sales Trends')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend(title='Region')
plt.grid(True)
plt.show()
print()

# Group by Date and Product
product_trends = df.groupby(['Date', 'Product']).agg({'Sales': 'mean'}).reset_index()

# Line plot for Sales by Product
plt.figure(figsize=(10, 6))
for product in product_trends['Product'].unique():
    product_data = product_trends[product_trends['Product'] == product]
    plt.plot(product_data['Date'], product_data['Sales'], label=f'Product {product}', marker='s')

plt.title('Product Sales Trends')
plt.xlabel('Date')
plt.ylabel('Average Sales')
plt.legend(title='Product')
plt.grid(True)
plt.show()
print()

# Custom aggregation: Percentage increase
df['Prev_Sales'] = df.groupby('Region')['Sales'].shift(1)
df['Percent_Change'] = ((df['Sales'] - df['Prev_Sales']) / df['Prev_Sales']) * 100

custom_trends = df.groupby('Date').agg({'Percent_Change': 'mean'}).reset_index()

# Line plot for percentage change
plt.figure(figsize=(10, 6))
plt.plot(custom_trends['Date'], custom_trends['Percent_Change'], marker='^', color='purple')
plt.title('Average Percentage Change in Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Percent Change (%)')
plt.grid(True)
plt.show()
print()

# Pivot table for Sales
pivot_trends = df.pivot_table(
    index='Date',
    columns='Region',
    values='Sales',
    aggfunc='sum',
    fill_value=0
)

# Heatmap for regional sales trends
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_trends, annot=True, fmt=".1f", cmap='coolwarm', cbar=True)
plt.title('Regional Sales Trends Over Time')
plt.xlabel('Region')
plt.ylabel('Date')
plt.show()
print()

