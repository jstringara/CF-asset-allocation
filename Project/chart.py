import pandas as pd

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

# import the data, read the date column as date (american format)
df = pd.read_csv('HistoricalPrices.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')

# sort the data by date
df = df.sort_values(by=['Date'])

# plot the data all on one graph
fig = plt.figure()
fig.suptitle('Historical Prices of the S&P100', fontsize=14, fontweight='bold')

# plot the data
plt.plot(df['Date'], df['Open'], color='blue', label='Price')


# add the x-axis label
plt.xlabel('Date')
# tilt the x-axis label 45 degrees
plt.xticks(
    rotation=45,
    # ticks are every 30 days, first and last are included
    ticks = df['Date'][::30].tolist() + [df['Date'].iloc[-1]] + [df['Date'].iloc[0]]
)
# plt.xlim(df['Date'].iloc[0], df['Date'].iloc[-1])
# format the x-axis dates as %d-%m-%y
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))

# add the y-axis label
plt.ylabel('Close')

# add two points on the first and last date
plt.plot(df['Date'].iloc[0], df['Open'].iloc[0], 'ro', label= f'Start price {df["Open"].iloc[0]}')
plt.plot(df['Date'].iloc[-1], df['Open'].iloc[-1], 'ro', label=f'End price {df["Open"].iloc[-1]}')


# add the legend
plt.legend(loc='upper left')

# show the plot
plt.show()
