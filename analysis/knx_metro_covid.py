#!/usr/bin/env python
# coding: utf-8

import json
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def logifunc(x, a, x0, k):
    return a / (1. + np.exp(-k * (x - x0)))


# matplotlib plotting style
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Hack'

# seaborn plotting style
sns.set()
sns.set_style('whitegrid')

# reusable data dir
datadir = '../data/'

# read fips county data
fips_datafile = datadir + 'csv/fips.csv'
fips_df = pd.read_csv(fips_datafile, encoding="ISO-8859-1")

# cleanup dataframe
fips_df['county_name'] = fips_df['county_name'].str.lower()
fips_df['state_abbr'] = fips_df['state_abbr'].str.lower()
fips_df['county_name'] = fips_df['county_name'].str.replace(' county', '')

# read hand-defined "metro area"
metro_datafile = datadir + 'json/metro.json'

# use the 60min catchment area
drive_time = '60'
with open(metro_datafile) as f:
    metro_data = json.load(f)

# assign metro to fips
knx_metro_fips = []
for county in metro_data[drive_time]:
    knx_metro_fips.append(
        fips_df[(fips_df['county_name'] == county)
                & (fips_df['state_abbr'] == 'tn')]['fips'].values[0])
print('Knoxville Metro FIPS: {}'.format(knx_metro_fips))

# read NY Times covid data set
ny_times_datafile = datadir + 'ny-times/us-counties.csv'
ny_times_df = pd.read_csv(ny_times_datafile)

# remove NA's
ny_times_df.fillna(0, inplace=True)

# convert to py datetime
ny_times_df['date'] = pd.to_datetime(ny_times_df['date'], errors='coerce')

# conert fips from string to int
ny_times_df['fips'] = ny_times_df['fips'].astype('int')

# filter for data in KNX metro fips
knx_df = ny_times_df[ny_times_df['fips'].isin(knx_metro_fips)]

# plot cases per county per day
plt.figure(figsize=(14, 9))
ax = sns.lineplot(x='date', y='cases', hue='county', markers=True, dashes=False, data=knx_df)
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.xlabel('Date [YYYY-MM-DD]')
plt.ylabel('Total Confirmed Cases')
plt.title('Knoxville Metro COVID19 Cases by County')
plt.savefig('../imgs/metro-county-cases.png')

# plot aggregate cases per day for KNX metro
plt.figure(figsize=(14, 9))
case_series = knx_df.groupby(knx_df.date.dt.date)['cases'].sum()
case_series.plot(kind='bar')
plt.xlabel('Date [YYYY-MM-DD]')
plt.ylabel('Total Confirmed Cases')
plt.title('Knoxville Metro COVID19 Cases')
plt.savefig('../imgs/metro-all.png')

# select dates and cases as arrays
x = [n for n, _ in enumerate(case_series.index)]
y = case_series.values

# scale case number for fitting
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))
y_scaled = y_scaled.reshape(1, -1)[0]

# use scipy opt to fit logistic
popt, pcov = opt.curve_fit(logifunc, x, y_scaled, maxfev=100000)
x_fit = np.linspace(0, 30, num=200)
y_fit = logifunc(x_fit, *popt)

# reverse the scaling
y_fit = y_fit.reshape(-1, 1)
y_fit = scaler.inverse_transform(y_fit).reshape(1, -1)[0]

# plot projected cases
fig, ax = plt.subplots(figsize=(14, 9))
plt.scatter(x, y, label='Case Data')
plt.plot(x_fit, y_fit, 'r-', label='Fitted function')
plt.legend()
plt.xlabel('Days from first reported case')
plt.ylabel('Total Confirmed Cases')
plt.title('Knoxville Metro COVID19 Cases')
ax.annotate('Max Cases: ~{}'.format(round(max(y_fit))),
            xy=(x_fit[-1], y_fit[-1]),
            xycoords='data',
            xytext=(25, 85),
            arrowprops=dict(facecolor='black', shrink=0.05))
ratio = max(y_fit) / max(y)
ax.text(25, 80, "Approx. {}x current".format(round(ratio, 1)))
plt.savefig('../imgs/metro-all-fit.png')
