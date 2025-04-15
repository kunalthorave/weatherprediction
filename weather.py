import pandas as pd
import zipfile

# Unzip and load the dataset
with zipfile.ZipFile('/content/weatherHistory.csv.zip', 'r') as zip_ref:
    zip_ref.extractall('/mnt/data/')

df = pd.read_csv('/content/weatherHistory.csv.zip')
df.head()
# Rename columns for ease
df.rename(columns={
    'Formatted Date': 'datetime',
    'Temperature (C)': 'temp',
    'Humidity': 'humidity',
    'Pressure (millibars)': 'pressure'
}, inplace=True)

# Convert datetime
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

# Drop unused columns
df = df[['datetime', 'temp', 'humidity', 'pressure']]

# Drop missing values and duplicates
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Optional: Filter out outliers
df = df[(df['temp'] > -30) & (df['temp'] < 50)]

print(df.describe())
print(df.corr())

# Resample to daily average
daily_df = df.set_index('datetime').resample('D').mean()

import matplotlib.pyplot as plt
import seaborn as sns

# Line plot of temperature
plt.figure(figsize=(12, 5))
plt.plot(daily_df.index, daily_df['temp'], label='Temperature (°C)', color='blue')
plt.title('Daily Temperature Trend')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.grid(True)
plt.legend()
plt.show()

# Boxplot of Temperature
sns.boxplot(x=df['temp'])
plt.title('Temperature Distribution')
plt.show()

# Scatter plot: Humidity vs Temperature
sns.scatterplot(x='humidity', y='temp', data=df)
plt.title('Humidity vs Temperature')
plt.show()

from statsmodels.tsa.arima.model import ARIMA

# Use the daily average temperature for modeling
ts = daily_df['temp'].dropna()

# Fit ARIMA model (you can tune order as needed)
model = ARIMA(ts, order=(2, 1, 2))
model_fit = model.fit()

# Forecast next 7 days
forecast = model_fit.forecast(steps=7)
print("Forecasted Temperatures:")
print(forecast)
