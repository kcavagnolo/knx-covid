#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time

import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import seaborn as sns
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
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
        setattr(namespace, self.dest, os.path.abspath(
            os.path.expanduser(values)))


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
    parser.add_argument("-d", "--datadir", type=is_dir,
                        help="Data directory", action=FullPaths, required=True)
    parser.add_argument("-i", "--imgdir", type=is_dir,
                        help="Images directory", action=FullPaths, required=True)
    parser.add_argument("-v", "--verbose",
                        help="increase verbosity", action="store_true")
    parser.add_argument("-dt", "--drive-time", type=str, default="60",
                        help="Drive time to use for metro definition")
    parser.add_argument("-nd", "--num-days", type=int,
                        default=0, help="Number of days to add linear growth")
    parser.add_argument("-gr", "--growth-rate", type=float,
                        default=1.15, help="Linear growth rate")
    parser.add_argument("-th", "--time-horizon", type=int,
                        default=30, help="Number of days to look forward")
    parser.add_argument("-pc", "--population", type=float,
                        default=1e6, help="KNX metro population")
    args = parser.parse_args()
    return args


def logifunc(x, a, x0, k):
    return a / (1. + np.exp(-k * (x - x0)))


def daily_fb_forecast(df, d):
    cases = df.groupby(df.date.dt.date)['cases'].sum().diff()
    df = cases.to_frame().reset_index().fillna(0)
    df.columns = ['ds', 'y']
    m = Prophet(interval_width=0.95)
    m.fit(df)
    future = m.make_future_dataframe(periods=d)
    pred = m.predict(future)
    return m, pred


def worst_fb_forecast(df, c, d):
    cases = df.groupby(df.date.dt.date)['cases'].sum()
    df = cases.to_frame().reset_index().fillna(0)
    df.columns = ['ds', 'y']
    df['y'] = np.log(df['y'])
    df = df.replace([np.inf, -np.inf], 0)
    capacity = np.log(c)
    df['cap'] = capacity
    df['floor'] = 0.0
    m = Prophet(growth='logistic', interval_width=0.95)
    m.fit(df)
    future = m.make_future_dataframe(periods=d)
    future['cap'] = capacity
    future['floor'] = 0.0
    pred = m.predict(future)
    return m, pred


def line_format(date):
    """
    Convert a datetime obj to graphic friendly MMMDD format
    """
    label = date.strftime("%b%d")
    if label == 'Jan':
        label += f'\n{date.strftime("%Y")}'
    return label


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
    knx_capacity = args.population

    # file definitions
    fips_datafile = os.path.join(datadir, 'csv/fips.csv')
    metro_datafile = os.path.join(datadir, 'json/metro.json')
    ny_times_datafile = os.path.join(datadir, 'ny-times/us-counties.csv')
    readme_file = os.path.join(os.path.dirname(__file__), '../README.md')

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
    daily_model, daily_forecast = daily_fb_forecast(knx_df, days_out)

    # forecast unabated cases with logistic growth
    worst_model, worst_forecast = worst_fb_forecast(
        knx_df, knx_capacity, days_out)

    # figure setup
    figsize = (14, 9)
    attribution = 'Data from The New York Times, based on reports from state and local health agencies.'
    attr_fontsize = 10
    attr_color = '#000000'
    attr_alpha = 0.33

    # plot cases per county per day
    log.info("# Creating figures")
    plt.subplots(figsize=figsize)
    sns.lineplot(x='date', y='cases', hue='county',
                 markers=True, dashes=False, data=knx_df)
    plt.legend(bbox_to_anchor=(1, 1), loc=2)
    plt.xlabel('Date [YYYY-MM-DD]')
    plt.ylabel('Total Confirmed Cases')
    plt.title(
        'Knoxville Metro COVID19 Cumulative Cases by County -- Updated: {}'.format(time_now()))
    plt.annotate(attribution, (0, 0), (0, -60),
                 xycoords='axes fraction',
                 textcoords='offset points',
                 fontsize=attr_fontsize, color=attr_color, alpha=attr_alpha)
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, 'metro-cases-county.png'))

    # plot aggregate cases per day for KNX metro
    plt.figure(figsize=figsize)
    ax = case_series.plot(kind='bar', rot=0)
    plt.xlabel('Date')
    ax.set_xticklabels(map(lambda xt: line_format(xt), case_series.index))
    plt.ylabel('Total Confirmed Cases')
    plt.title(
        'Knoxville Metro COVID19 Cumulative Cases -- Updated: {}'.format(time_now()))
    plt.annotate(attribution, (0, 0), (0, -60),
                 xycoords='axes fraction',
                 textcoords='offset points',
                 fontsize=attr_fontsize, color=attr_color, alpha=attr_alpha)
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, 'metro-cases-all.png'))

    # plot best scenario
    plt.subplots(figsize=figsize)
    plt.scatter(x, y, label='Confirmed')
    plt.plot(x_fit, y_fit, 'r-', label='Projected')
    plt.xticks(np.arange(min(x_fit), max(x_fit) + 1, 7.0))
    plt.legend()
    plt.xlabel('Days from first reported case')
    plt.ylabel('Total Cases')
    plt.title(
        'Knoxville Metro COVID19 Projected Cumulative Cases -- Updated: {}'.format(time_now()))
    plt.annotate('Max Cases: {:.0f}\nApprox. {:.1f}x current\nRollover Date: {}'.format(max(y_fit), ratio, end_date),
                xytext=(0.75, 0.75), textcoords='figure fraction',
                horizontalalignment='right', verticalalignment='top',
                xy=(x_fit[no_new_cases], y_fit[no_new_cases]), xycoords='data',
                arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate(attribution, (0, 0), (0, -60),
                 xycoords='axes fraction',
                 textcoords='offset points',
                 fontsize=attr_fontsize, color=attr_color, alpha=attr_alpha)
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, 'metro-cases-all-fit-best.png'))

    # plot forecasted daily cases
    plt.figure(figsize=figsize)
    fig = daily_model.plot(daily_forecast)
    _ = add_changepoints_to_plot(fig.gca(), daily_model, daily_forecast)
    plt.xlabel('Date [YYYY-MM-DD]')
    plt.ylabel('Total Cases')
    plt.title(
        'Knoxville Metro COVID19 Forecasted Daily New Cases -- Updated: {}'.format(time_now()))
    plt.annotate(attribution, (0, 0), (0, -60),
                 xycoords='axes fraction',
                 textcoords='offset points',
                 fontsize=attr_fontsize, color=attr_color, alpha=attr_alpha)
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, 'metro-cases-all-daily-forecasted.png'))

    # plot worst scenario
    plt.figure(figsize=figsize)
    fig = worst_model.plot(worst_forecast)
    _ = add_changepoints_to_plot(fig.gca(), worst_model, worst_forecast)
    plt.xlabel('Date [YYYY-MM-DD]')
    plt.ylabel('ln(Total Cases)')
    plt.title(
        'Knoxville Metro COVID19 Forecasted Cumulative Cases -- Updated: {}'.format(time_now()))
    plt.annotate(attribution, (0, 0), (0, -60),
                 xycoords='axes fraction',
                 textcoords='offset points',
                 fontsize=attr_fontsize, color=attr_color, alpha=attr_alpha)
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, 'metro-cases-all-fit-worst.png'))

    # update the readme
    log.info("# Updating README")
    current = '**Updated on'
    updated = "**Updated on {}**\n".format(time_now())
    with open(readme_file, 'r') as f:
        text = f.readlines()
        new_text = "".join(
            [line if current not in line else updated for line in text])
    with open(readme_file, 'w') as f:
        f.write(new_text)


if __name__ == "__main__":
    log.info("#### run-start: {}".format(time_now()))
    main()
    log.info("#### run-end: {}".format(time_now()))
    log.info("#### run-elapsed: {}".format(time.time() - start_time))
    sys.exit()
