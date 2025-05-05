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

# Aggregate sum and mean for Sales
df_agg = df.agg({
    'Sales': ['sum', 'mean'],
    'Profit': ['sum', 'mean']
})
print(df_agg)
print()

# Aggregate sum and mean for Sales
agg_data = df.agg({
    'Sales': ['sum', 'mean']
}).reset_index()
agg_data.columns = ['Metric', 'Value']

# Bar plot for aggregated metrics
plt.figure(figsize=(8, 5))
plt.bar(agg_data['Metric'], agg_data['Value'], color=['skyblue', 'orange'])
plt.title('Sales Aggregation (Sum and Mean)')
plt.ylabel('Value')
plt.show()

# Group by Region and calculate total Sales and Profit
grouped = df.groupby('Region').agg({
    'Sales': 'sum',
    'Profit': 'sum'
})
print(grouped)
print()

# Group by Region and calculate total Sales and Profit
grouped = df.groupby('Region').agg({
    'Sales': 'sum',
    'Profit': 'sum'
}).reset_index()

# Bar plot for Sales and Profit by Region
plt.figure(figsize=(8, 5))
x = grouped['Region']
plt.bar(x, grouped['Sales'], width=0.4, label='Sales', align='center', color='skyblue')
plt.bar(x, grouped['Profit'], width=0.4, label='Profit', align='edge', color='orange')
plt.title('Total Sales and Profit by Region')
plt.ylabel('Value')
plt.xlabel('Region')
plt.legend()
plt.show()

# Group by Region and Product, calculate mean Sales and Profit
grouped_multi = df.groupby(['Region', 'Product']).agg({
    'Sales': 'mean',
    'Profit': 'mean'
}).reset_index()
print(grouped_multi)
print()

# Group by Region and Product
grouped_multi = df.groupby(['Region', 'Product']).agg({
    'Sales': 'mean'
}).reset_index()

# Bar plot for Sales by Region and Product
plt.figure(figsize=(10, 6))
for product in grouped_multi['Product'].unique():
    product_data = grouped_multi[grouped_multi['Product'] == product]
    plt.bar(product_data['Region'], product_data['Sales'], label=f'Product {product}', alpha=0.7)

plt.title('Average Sales by Region and Product')
plt.ylabel('Average Sales')
plt.xlabel('Region')
plt.legend(title='Product')
plt.show()

# Custom function for percentage increase in Sales
def percent_increase(x):
    return (x.max() - x.min()) / x.min() * 100

custom_agg = df.groupby('Region')['Sales'].agg(percent_increase)
print(custom_agg)
print()

# Custom aggregation: Percentage increase in sales
def percent_increase(x):
    return (x.max() - x.min()) / x.min() * 100

custom_agg = df.groupby('Region')['Sales'].agg(percent_increase).reset_index()
custom_agg.columns = ['Region', 'Percent Increase']

# Line plot for percentage increase
plt.figure(figsize=(8, 5))
plt.plot(custom_agg['Region'], custom_agg['Percent Increase'], marker='o', color='purple')
plt.title('Percentage Increase in Sales by Region')
plt.ylabel('Percent Increase')
plt.xlabel('Region')
plt.grid(True)
plt.show()

pivot = df.pivot_table(
    index='Region', 
    columns='Product', 
    values='Sales', 
    aggfunc='sum', 
    fill_value=0
)
print(pivot)
print()

# Pivot table for Sales
pivot = df.pivot_table(
    index='Region',
    columns='Product',
    values='Sales',
    aggfunc='sum',
    fill_value=0
)

# Heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap='coolwarm')
plt.title('Sales by Region and Product')
plt.ylabel('Region')
plt.xlabel('Product')
plt.show()
