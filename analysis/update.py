#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from fbprophet import Prophet

import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# import traceback

# setup file logger
start_time = time.time()
logfile = os.path.splitext(os.path.basename(__file__))[0] + '.log'
msgfmt = '%(asctime)s %(name)-22s %(levelname)-8s %(message)s'
dtefmt = '%d-%m-%Y %H:%M:%S'
logging.basicConfig(level=logging.DEBUG,
                    format=msgfmt,
                    datefmt=dtefmt,
                    filename=logfile,
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(msgfmt)
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
log = logging.getLogger()

# matplotlib plotting style
log.info("# Loading fonts")
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Open Sans'

# seaborn plotting style
sns.set()
sns.set_style('whitegrid')


class FullPaths(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        """
        Expand user- and relative-paths
        :param parser:
        :param namespace:
        :param values:
        :param option_string:
        """
        setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))


def time_now():
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()


def is_dir(dirname):
    """
    Checks if a path is an actual directory
    :param dirname: a directory name
    :return: the directory name
    """
    if not os.path.isdir(dirname):
        msg = "{0} is not a directory".format(dirname)
        raise argparse.ArgumentTypeError(msg)
    else:
        return dirname


def parse_arguments():
    """
    Parses arguments from command line
    :return: array of parsed arguments
    """
    parser = argparse.ArgumentParser(description="What's the story with COVID19 cases in Knoxville Metro?",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datadir", type=is_dir, help="Data directory", action=FullPaths, required=True)
    parser.add_argument("-i", "--imgdir", type=is_dir, help="Images directory", action=FullPaths, required=True)
    parser.add_argument("-v", "--verbose", help="increase verbosity", action="store_true")
    parser.add_argument("-dt", "--drive-time", type=str, default="60", help="Drive time to use for metro definition")
    parser.add_argument("-nd", "--num-days", type=int, default=0, help="Number of days to add linear growth")
    parser.add_argument("-gr", "--growth-rate", type=float, default=1.15, help="Linear growth rate")
    parser.add_argument("-th", "--time-horizon", type=int, default=30, help="Number of days to look forward")
    args = parser.parse_args()
    return args


def logifunc(x, a, x0, k):
    return a / (1. + np.exp(-k * (x - x0)))


def main():
    """
    The main program that updates projections
    :rtype: object
    """

    # parse arguments
    args = parse_arguments()

    # set verbosity
    if args.verbose:
        log.setLevel(logging.DEBUG)

    # use some args
    datadir = args.datadir
    imgdir = args.imgdir
    drive_time = args.drive_time
    ndays = args.num_days
    growth_rate = args.growth_rate
    time_horizon = args.time_horizon

    # file definitions
    fips_datafile = os.path.join(datadir, 'csv/fips.csv')
    metro_datafile = os.path.join(datadir, 'json/metro.json')
    ny_times_datafile = os.path.join(datadir, 'ny-times/us-counties.csv')

    # read fips county data
    log.info("# Processing FIPS")
    fips_df = pd.read_csv(fips_datafile, encoding="ISO-8859-1")

    # cleanup dataframe
    fips_df['county_name'] = fips_df['county_name'].str.lower()
    fips_df['state_abbr'] = fips_df['state_abbr'].str.lower()
    fips_df['county_name'] = fips_df['county_name'].str.replace(' county', '')

    # use the 60min catchment area
    with open(metro_datafile) as f:
        metro_data = json.load(f)

    # assign metro to fips
    knx_metro_fips = []
    for county in metro_data[drive_time]:
        knx_metro_fips.append(
            fips_df[(fips_df['county_name'] == county)
                    & (fips_df['state_abbr'] == 'tn')]['fips'].values[0])

    # read NY Times covid data set
    log.info("# Processing NY Times data")
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
    log.info("# Fitting logistic function")
    x = [n for n, _ in enumerate(case_series.index)]
    y = case_series.values

    # project an extra n days of linear growth
    for _ in range(0, ndays):
        x = np.append(x, x[-1] + 1)
        y = np.append(y, y[-1] * growth_rate)

    # scale case number for fitting
    scaler = MinMaxScaler()
    y_scaled = scaler.fit_transform(y.reshape(-1, 1))
    y_scaled = y_scaled.reshape(1, -1)[0]

    # use scipy opt to fit logistic
    popt, pcov = opt.curve_fit(logifunc, x, y_scaled, maxfev=100000)
    days_out = int(x[-1] + time_horizon)
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

    # forecast daily cases
    daily_cases = knx_df.groupby(knx_df.date.dt.date)['cases'].sum().diff()
    prophet_df = daily_cases.to_frame().reset_index().fillna(0)
    prophet_df.columns = ['ds', 'y']
    m = Prophet()
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=days_out)
    forecast = m.predict(future)

    # plot cases per county per day
    log.info("# Creating figures")
    plt.figure(figsize=(14, 9))
    sns.lineplot(x='date', y='cases', hue='county', markers=True, dashes=False, data=knx_df)
    plt.legend(bbox_to_anchor=(1, 1), loc=2)
    plt.xlabel('Date [YYYY-MM-DD]')
    plt.ylabel('Total Confirmed Cases')
    plt.title('Knoxville Metro COVID19 Cumulative Cases by County -- Updated: {}'.format(time_now()))
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, 'metro-county-cases.png'))

    # plot aggregate cases per day for KNX metro
    plt.figure(figsize=(14, 9))
    case_series.plot(kind='bar')
    plt.xlabel('Date [YYYY-MM-DD]')
    plt.ylabel('Total Confirmed Cases')
    plt.title('Knoxville Metro COVID19 Cumulative Cases -- Updated: {}'.format(time_now()))
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, 'metro-all.png'))

    # plot projected cases
    fig, ax = plt.subplots(figsize=(14, 9))
    plt.scatter(x, y, label='Confirmed')
    plt.plot(x_fit, y_fit, 'r-', label='Projected')
    plt.xticks(np.arange(min(x_fit), max(x_fit) + 1, 7.0))
    plt.legend()
    plt.xlabel('Days from first reported case')
    plt.ylabel('Total Cases')
    plt.title('Knoxville Metro COVID19 Projected Cumulative Cases -- Updated: {}'.format(time_now()))
    ax.annotate('Max Cases: {:.0f}\nApprox. {:.1f}x current\nRollover Date: {}'.format(max(y_fit), ratio, end_date),
                xytext=(0.75, 0.75), textcoords='figure fraction',
                horizontalalignment='right', verticalalignment='top',
                xy=(x_fit[no_new_cases], y_fit[no_new_cases]), xycoords='data',
                arrowprops=dict(facecolor='black', shrink=0.05))
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, 'metro-all-fit.png'))

    # plot forecasted daily cases
    fig1 = m.plot(forecast)
    plt.xlabel('Date [YYYY-MM-DD]')
    plt.ylabel('Total Cases')
    plt.title('Knoxville Metro COVID19 Forecasted Daily New Cases -- Updated: {}'.format(time_now()))
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, 'metro-all-forecasted.png'))

    # plot forecasted daily cases
    fig2 = m.plot_components(forecast)
    plt.title('Knoxville Metro COVID19 Forecasted Daily New Cases Components -- Updated: {}'.format(time_now()))
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, 'metro-all-forecasted-components.png'))


if __name__ == "__main__":
    log.info("#### run-start: {}".format(time_now()))
    main()
    log.info("#### run-end: {}".format(time_now()))
    log.info("#### run-elapsed: {}".format(time.time() - start_time))
    sys.exit()
