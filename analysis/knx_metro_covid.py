#!/usr/bin/env python
# coding: utf-8

import sys
import datetime as dt
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
case_series = knx_df.groupby(knx_df.date.dt.date)['cases'].sum()

# select dates and cases as arrays
x = [n for n, _ in enumerate(case_series.index)]
y = case_series.values

# project an extra n days of linear growth
ndays = 0
growth_rate = 1.2
for _ in range(0, ndays):
    x = np.append(x, x[-1] + 1)
    y = np.append(y, y[-1] * growth_rate)

# scale case number for fitting
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))
y_scaled = y_scaled.reshape(1, -1)[0]

# use scipy opt to fit logistic
popt, pcov = opt.curve_fit(logifunc, x, y_scaled, maxfev=100000)
days_out = int(x[-1] + 30)
x_fit = np.linspace(0, days_out, num=days_out)
y_fit = logifunc(x_fit, *popt)

# reverse the scaling
y_fit = y_fit.reshape(-1, 1)
y_fit = scaler.inverse_transform(y_fit).reshape(1, -1)[0]
ratio = max(y_fit) / max(y)

# when do new cases fall below ~1 (i.e., < 0.5)?
no_new_cases = 0
case_diff = np.diff(y_fit)
for i, ndiff in enumerate(case_diff):
    if ndiff < 0.5 and i > x[-1]:
        no_new_cases = i
        break
end_date = case_series.index[-1] + dt.timedelta(days=no_new_cases)

# updated date for labeling figures
now = dt.datetime.now().replace(microsecond=0).isoformat()

# plot cases per county per day
plt.figure(figsize=(14, 9))
ax = sns.lineplot(x='date', y='cases', hue='county', markers=True, dashes=False, data=knx_df)
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.xlabel('Date [YYYY-MM-DD]')
plt.ylabel('Total Confirmed Cases')
plt.title('Knoxville Metro COVID19 Cases by County -- Updated: {}'.format(now))
plt.tight_layout()
plt.savefig('../imgs/metro-county-cases.png')

# plot aggregate cases per day for KNX metro
plt.figure(figsize=(14, 9))
case_series.plot(kind='bar')
plt.xlabel('Date [YYYY-MM-DD]')
plt.ylabel('Total Confirmed Cases')
plt.title('Knoxville Metro COVID19 Cases -- Updated: {}'.format(now))
plt.tight_layout()
plt.savefig('../imgs/metro-all.png')

# plot projected cases
fig, ax = plt.subplots(figsize=(14, 9))
plt.scatter(x, y, label='Confirmed')
plt.plot(x_fit, y_fit, 'r-', label='Projected')
plt.xticks(np.arange(min(x_fit), max(x_fit)+1, 7.0))
plt.legend()
plt.xlabel('Days from first reported case')
plt.ylabel('Total Cases')
plt.title('Projected Knoxville Metro COVID19 Cases -- Updated: {}'.format(now))
ax.annotate('Max Cases: {:.0f}\nApprox. {:.1f}x current\nRollover Date: {}'.format(max(y_fit), ratio, end_date),
            xytext=(0.75, 0.75), textcoords='figure fraction',
            horizontalalignment='right', verticalalignment='top',
            xy=(x_fit[no_new_cases], y_fit[no_new_cases]), xycoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.tight_layout()
plt.savefig('../imgs/metro-all-fit.png')
