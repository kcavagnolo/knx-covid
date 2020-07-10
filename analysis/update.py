#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime as dt
import glob
import json
import logging
import os
import sys
import time

import geopandas as gpd
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import seaborn as sns
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

pd.options.mode.chained_assignment = None

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


class SuppressStdout(object):
    """
    https://github.com/facebook/prophet/issues/223#issuecomment-326455744
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.

    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]

        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)

        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


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
    parser.add_argument("-d", "--datadir",
                        required=True,
                        type=is_dir,
                        help="Data directory",
                        action=FullPaths)
    parser.add_argument("-i", "--imgdir",
                        required=True,
                        type=is_dir,
                        help="Images directory",
                        action=FullPaths)
    parser.add_argument("-v", "--verbose",
                        help="increase verbosity",
                        action="store_true")
    parser.add_argument("-dt", "--drive-time",
                        type=str,
                        default="60",
                        help="Drive time to use for metro definition")
    parser.add_argument("-nd", "--num-days",
                        type=int,
                        default=0,
                        help="Number of days to add linear growth")
    parser.add_argument("-gr", "--growth-rate",
                        type=float,
                        default=1.15,
                        help="Linear growth rate")
    parser.add_argument("-th", "--time-horizon",
                        type=int,
                        default=30,
                        help="Number of days to look forward")
    parser.add_argument("-pc", "--population",
                        type=float,
                        default=1e6,
                        help="KNX metro population")
    parser.add_argument("-f", "--forecast",
                        type=bool,
                        default=False,
                        help="Run Prophet forecasting")
    args = parser.parse_args()
    return args


def remove_tmp_files(imgdir):
    log.info("# Removing images")
    imtypes = ["wc*", "palette*"]
    for imtype in imtypes:
        imgfiles = glob.glob(os.path.join(imgdir, imtype))
        for imgfile in imgfiles:
            try:
                os.remove(imgfile)
            except OSError as e:
                log.error(e)
                pass


def process_tn_data(fips_datafile, metro_datafile, hospitals_datafile, nytimes_datafile, jhu_datafiles, drive_time):
    # process fips
    log.info("# Processing TN Data")
    fips_df = pd.read_csv(fips_datafile, encoding="ISO-8859-1")
    fips_df['county_name'] = fips_df['county_name'].str.lower()
    fips_df['state_abbr'] = fips_df['state_abbr'].str.lower()
    fips_df['county_name'] = fips_df['county_name'].str.replace(' county', '')

    # open metro ref file
    with open(metro_datafile) as f:
        metro_data = json.load(f)
    metro_counties = [c for c in metro_data[drive_time]]

    # assign metro to fips
    knx_metro_fips = []
    for county in metro_counties:
        knx_metro_fips.append(
            fips_df[(fips_df['county_name'] == county)
                    & (fips_df['state_abbr'] == 'tn')]['fips'].values[0])

    # read tn hospital data
    hospitals_df = gpd.read_file(hospitals_datafile)
    knx_hospitals_df = hospitals_df[hospitals_df['County'].str.lower().isin(metro_counties)]
    knx_hospitals_df.loc[knx_hospitals_df['icu_beds'] < 0, 'icu_beds'] = 0

    # read NY Times covid data set
    nytimes_df = pd.read_csv(nytimes_datafile)
    nytimes_df.fillna(0, inplace=True)
    nytimes_df.columns = [c.lower().replace(' ', '_') for c in nytimes_df.columns]
    nytimes_df.rename(columns={'cases': 'ncases', 'deaths': 'ndeaths'}, inplace=True)
    nytimes_df['date'] = pd.to_datetime(nytimes_df['date'], errors='coerce')
    nytimes_df['date'] = nytimes_df.date.dt.date
    nytimes_df['fips'] = nytimes_df['fips'].astype('int')
    nytimes_df = df_clean_string(nytimes_df)
    nytimes_df = nytimes_df.sort_values(by=['date', 'state', 'county'])
    nknx_df = nytimes_df[nytimes_df['fips'].isin(knx_metro_fips)]

    # read JHU covid data set
    jhu_df = (pd.read_csv(f) for f in jhu_datafiles)
    jhu_df = pd.concat(jhu_df, ignore_index=True)
    threshold = 0.75 * len(jhu_df)
    jhu_df = jhu_df.dropna(thresh=threshold, axis=1)
    threshold = 0.50 * len(jhu_df.columns)
    jhu_df = jhu_df.dropna(thresh=threshold, axis=0)
    jhu_df = jhu_df.reset_index().drop('index', axis=1)
    jhu_df.fillna(0, inplace=True)
    jhu_df.columns = [c.lower().replace(' ', '_') for c in jhu_df.columns]
    jhu_df.rename(columns={'admin2': 'county', 'confirmed': 'jcases', 'deaths': 'jdeaths'}, inplace=True)
    jhu_df['date'] = pd.to_datetime(jhu_df['last_update'], errors='coerce')
    jhu_df['date'] = jhu_df.date.dt.date
    jhu_df['fips'] = jhu_df['fips'].astype('int')
    jhu_df = df_clean_string(jhu_df)
    jhu_df = jhu_df.sort_values(by=['date', 'country_region', 'province_state', 'county'])
    jknx_df = jhu_df[jhu_df['fips'].isin(knx_metro_fips)]

    # combine the data and take max of either
    knx_df = pd.merge(nknx_df, jknx_df, on=['date', 'county', 'fips'], how="outer")
    knx_df.fillna(0, inplace=True)
    knx_df["cases"] = knx_df[["jcases", "ncases"]].values.max(1)
    knx_df["deaths"] = knx_df[["jdeaths", "ndeaths"]].values.max(1)

    # filter
    # knx_df = knx_df[knx_df['date'] >= dt.date(2020, 5, 22)]

    return knx_df


def process_midas_data(midas_datafile):
    # load midas param estimates
    midas_params = pd.read_csv(midas_datafile)

    # remove whitespace and lowercase everything
    midas_params = midas_params.stack().str.replace(' ', '_').unstack()
    midas_params = midas_params.stack().str.lower().unstack()

    # progression model params
    params = {}

    # select for peer reviewed values
    midas_params_pr = midas_params.dropna(subset=['peer_review'])
    midas_params_pr = midas_params_pr[midas_params_pr['peer_review'] == 'positive']

    # R_0 -- https://en.wikipedia.org/wiki/Basic_reproduction_number
    p = 'basic_reproduction_number'
    params['r0'] = float(midas_params_pr[midas_params_pr['name'] == p]['value'].max())

    # incubation period -- https://en.wikipedia.org/wiki/Incubation_period
    p = 'incubation_period'
    params['ip'] = float(midas_params_pr[midas_params_pr['name'] == p]['value'].max())

    # transmission rate -- https://en.wikipedia.org/wiki/Transmission_risks_and_rates
    p = 'transmission_rate'
    params['tr'] = float(midas_params_pr[midas_params_pr['name'] == p]['value'].min())

    # select for not peer reviewed values
    midas_params_npr = midas_params[midas_params['peer_review'] != 'positive']

    # time from symptoms to Hospitalization (tr)
    p = 'time_from_symptom_onset_to_hospitalization'
    params['soh'] = midas_params_npr[midas_params_npr['name'] == p]['value'].astype('float').mean()

    # those that go to icu (icu)
    p = "proportion_of_hospitalized_cases_admitted_to_icu"
    params['icu'] = midas_params_npr[midas_params_npr['name'] == p]['value'].astype('float').mean()

    # case hospitalization rate (chr)
    # percent confirmed cases requiring hospitalization: 15% (China)
    # https://www.cdc.gov/coronavirus/2019-ncov/hcp/clinical-guidance-management-patients.html
    params['chr'] = 0.15

    return params


def logifunc(x, a, x0, k):
    return a / (1. + np.exp(-k * (x - x0)))


def logistic_forecast(knx_df, ndays=0, growth_rate=0, time_horizon=0):
    # select data to fit
    log.info("# Fitting logistic function")
    case_series = knx_df.groupby(knx_df.date)['cases'].sum()

    # save param outputs
    params = {}

    # transform x from date to int
    x = [n for n, _ in enumerate(case_series.index)]

    # transform from np array to arr
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
    params["days_out"] = int(x[-1] + time_horizon)
    x_fit = np.linspace(0, params["days_out"], num=params["days_out"])
    y_fit = logifunc(x_fit, *popt)

    # reverse the scaling
    y_fit = y_fit.reshape(-1, 1)
    y_fit = scaler.inverse_transform(y_fit).reshape(1, -1)[0]
    params["ratio"] = max(y_fit) / max(y)

    # when do new cases fall below ~1 (i.e., < 0.5)?
    no_new_cases = 0
    case_diff = np.diff(y_fit)
    for i, ndiff in enumerate(case_diff):
        if ndiff < 0.5 and i > x[-1]:
            no_new_cases = i
            break
    params["rollover_date"] = case_series.index[-1] + dt.timedelta(days=no_new_cases)
    params["rollover_date_coords"] = (x_fit[no_new_cases], y_fit[no_new_cases])

    return x_fit, y_fit, params


def daily_cases_fb_forecast(df, d, imgdir, attribution, figsize=(14, 9)):
    # fit model
    log.info("# Fitting daily cases using Prophet")
    cases = df.groupby(df.date)['cases'].sum().diff()
    df = cases.to_frame().reset_index().fillna(0)
    df.columns = ['ds', 'y']
    m = Prophet(interval_width=0.95)
    with SuppressStdout():
        m.fit(df)
    future = m.make_future_dataframe(periods=d)
    pred = m.predict(future)

    # plot result
    plt.figure(figsize=figsize)
    fig = m.plot(pred)
    _ = add_changepoints_to_plot(fig.gca(), m, pred)
    plt.xlabel('Date [YYYY-MM-DD]')
    plt.ylabel('Total Cases')
    plt.title('Knoxville Metro COVID19 Forecasted Daily New Cases -- Updated: {}'.format(time_now()))
    plt.annotate(attribution['text'], (0, 0), (0, -60),
                 xycoords='axes fraction',
                 textcoords='offset points',
                 fontsize=attribution['fsize'],
                 color=attribution['color'],
                 alpha=attribution['alpha']
                 )
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, 'metro-cases-all-daily-forecasted.png'))


def worst_case_fb_forecast(df, capacity, days_out, imgdir, attribution, figsize=(14, 9)):
    log.info("# Fitting full worst case using Prophet")
    cases = df.groupby(df.date)['cases'].sum()
    df = cases.to_frame().reset_index().fillna(0)
    df.columns = ['ds', 'y']
    df['y'] = np.log(df['y'])
    df = df.replace([np.inf, -np.inf], 0)
    capacity = np.log(capacity)
    df['cap'] = capacity
    df['floor'] = 0.0
    m = Prophet(growth='logistic', interval_width=0.95)
    with SuppressStdout():
        m.fit(df)
    future = m.make_future_dataframe(periods=days_out)
    future['cap'] = capacity
    future['floor'] = 0.0
    pred = m.predict(future)

    # plot current results
    plt.figure(figsize=figsize)
    fig = m.plot(pred)
    _ = add_changepoints_to_plot(fig.gca(), m, pred)
    plt.xlabel('Date [YYYY-MM-DD]')
    plt.ylabel('ln(Total Cases)')
    plt.title('Knoxville Metro COVID19 Forecasted Cumulative Cases -- Updated: {}'.format(time_now()))
    plt.annotate(attribution['text'], (0, 0), (0, -60),
                 xycoords='axes fraction',
                 textcoords='offset points',
                 fontsize=attribution['fsize'],
                 color=attribution['color'],
                 alpha=attribution['alpha']
                 )
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, 'metro-cases-all-fit-worst.png'))

    # initialize an img counter
    imgctr = 0
    xmin = dt.date(2020, 3, 1)
    xmax = dt.date.today() + dt.timedelta(days=days_out)

    # create an animation starting w/ first 3 data points
    for i in tqdm(range(3, len(cases) + 1), desc="Fitting FB Prophet df[:i]"):
        ani_df = df.iloc[:i]
        m = Prophet(growth='logistic', interval_width=0.95)
        with SuppressStdout():
            m.fit(ani_df)
        future = m.make_future_dataframe(periods=90)
        future['cap'] = capacity
        future['floor'] = 0.0
        pred = m.predict(future)
        plt.figure(figsize=figsize)
        fig = m.plot(pred)
        _ = add_changepoints_to_plot(fig.gca(), m, pred)
        plt.xlabel('Date [YYYY-MM-DD]')
        plt.ylabel('ln(Total Cases)')
        plt.title('Knoxville Metro COVID19 Forecasted Cumulative Cases -- Updated: {}'.format(time_now()))
        plt.annotate(attribution['text'], (0, 0), (0, -60),
                     xycoords='axes fraction',
                     textcoords='offset points',
                     fontsize=attribution['fsize'],
                     color=attribution['color'],
                     alpha=attribution['alpha']
                     )
        plt.xlim(xmin, xmax)
        plt.ylim(-1, 15)
        plt.tight_layout()
        plt.savefig(os.path.join(imgdir, 'wc_{:04d}'.format(imgctr)))
        plt.close('all')
        imgctr += 1

    # create an animation trimming early data successively
    j = round(0.75 * len(cases))
    for i in tqdm(range(0, j), desc="Fitting FB Prophet df[i:]"):
        ani_df = df.iloc[i:]
        m = Prophet(growth='logistic', interval_width=0.95)
        with SuppressStdout():
            m.fit(ani_df)
        future = m.make_future_dataframe(periods=90)
        future['cap'] = capacity
        future['floor'] = 0.0
        pred = m.predict(future)
        plt.figure(figsize=figsize)
        fig = m.plot(pred)
        _ = add_changepoints_to_plot(fig.gca(), m, pred)
        plt.xlabel('Date [YYYY-MM-DD]')
        plt.ylabel('ln(Total Cases)')
        plt.title('Knoxville Metro COVID19 Forecasted Cumulative Cases -- Updated: {}'.format(time_now()))
        plt.annotate(attribution['text'], (0, 0), (0, -60),
                     xycoords='axes fraction',
                     textcoords='offset points',
                     fontsize=attribution['fsize'],
                     color=attribution['color'],
                     alpha=attribution['alpha']
                     )
        plt.xlim(xmin, xmax)
        plt.ylim(-1, 15)
        plt.tight_layout()
        plt.savefig(os.path.join(imgdir, 'wc_{:04d}'.format(imgctr)))
        plt.close('all')
        imgctr += 1


def reorder_legend(df, fig):
    latest = df[df['date'] == df['date'].max()].sort_values(by=['cases'], ascending=False)
    latest = zip([x.lower() for x in latest['county'].unique()], [x for x in latest['cases']])
    handles, labels = fig.get_legend_handles_labels()
    legend_entries = {}
    for lh in zip(labels, handles):
        legend_entries[lh[0].lower()] = lh[1]
    labels = []
    handles = []
    for c in latest:
        labels.append(c[0].capitalize() + ': {:.0f}'.format(c[1]))
        handles.append(legend_entries[c[0]])
    return handles, labels


def line_format(date):
    """
    Convert a datetime obj to graphic friendly MMMDD format
    """
    label = date.strftime("%b%d")
    if label == 'Jan':
        label += f'\n{date.strftime("%Y")}'
    return label


def df_clean_string(df):
    for col in df.select_dtypes(include=['object']).columns:
        if isinstance(df[col][0], str):
            df[col] = df[col].str.lower().replace(' ', '_')
    return df


def plot_county_cases_per_day(df, imgdir, attribution, figsize=(14, 9)):
    plt.subplots(figsize=figsize)
    fig = sns.lineplot(x='date', y='cases',
                       hue='county',
                       # markers=True,
                       # marker='o',
                       dashes=False,
                       data=df)
    handles, labels = reorder_legend(df, fig)
    plt.legend(handles, labels, bbox_to_anchor=(1, 1), loc=2)
    plt.xlabel('Date [YYYY-MM-DD]')
    plt.ylabel('Cumulative Total Confirmed Cases')
    plt.title('Knoxville Metro COVID19 Cumulative Cases by County -- Updated: {}'.format(time_now()))
    plt.annotate(attribution['text'], (0, 0), (0, -60),
                 xycoords='axes fraction',
                 textcoords='offset points',
                 fontsize=attribution['fsize'],
                 color=attribution['color'],
                 alpha=attribution['alpha']
                 )
    safer_home_knx = dt.date(2020, 3, 23)
    safer_home_tn = dt.date(2020, 4, 2)
    knx_phase1 = dt.date(2020, 5, 1)
    plt.axvline(safer_home_knx, color='red', linewidth=2, linestyle=':')
    plt.annotate(' Knox Cty\n Closes', (safer_home_knx, 450))
    plt.axvline(safer_home_tn, color='orange', linewidth=2, linestyle=':')
    plt.annotate(' State\n Safer@Home', (safer_home_tn, 450))
    plt.axvline(knx_phase1, color='green', linewidth=2, linestyle=':')
    plt.annotate(' KnoxCty \n Reopens', (knx_phase1, 450))
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, 'metro-cases-county.png'))


def plot_metro_cases_per_day(df, imgdir, attribution, figsize=(14, 9)):
    data = df.groupby(df.date)['cases'].sum().reset_index()
    plt.subplots(figsize=figsize)
    _ = sns.lineplot(x='date', y='cases',
                     markers=True,
                     marker='o',
                     markersize=5,
                     dashes=False,
                     data=data)
    plt.xlabel('Date [YYYY-MM-DD]')
    plt.ylabel('Total Confirmed Cases')
    plt.title('Knoxville Metro COVID19 Cumulative Cases')
    plt.annotate(attribution['text'], (0, 0), (0, -60),
                 xycoords='axes fraction',
                 textcoords='offset points',
                 fontsize=attribution['fsize'],
                 color=attribution['color'],
                 alpha=attribution['alpha']
                 )
    # data = df.groupby(df.date)['cases'].sum()
    # plt.figure(figsize=figsize)
    # ax = data.plot(kind='bar')
    # plt.xlabel('Date')
    # ax.set_xticklabels(map(lambda xt: line_format(xt), data.index))
    # plt.ylabel('Total Confirmed Cases')
    # plt.title('Knoxville Metro COVID19 Cumulative Cases -- Updated: {}'.format(time_now()))
    # plt.annotate(attribution['text'], (0, 0), (0, -60),
    #              xycoords='axes fraction',
    #              textcoords='offset points',
    #              fontsize=attribution['fsize'],
    #              color=attribution['color'],
    #              alpha=attribution['alpha']
    #              )
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, 'metro-cases-all.png'))


def plot_logistic_model(df, log_model_x, log_model_y, log_model_params, imgdir, attribution, figsize=(14, 9)):
    data = df.groupby(df.date)['cases'].sum()
    basedate = data.index[0]
    x = data.index
    y = data.values
    log_model_x = np.array([basedate + dt.timedelta(days=i) for i in range(len(log_model_x))])
    fig, ax = plt.subplots(figsize=figsize)
    plt.scatter(x, y, label='Confirmed', marker='o', s=5)
    plt.plot(log_model_x, log_model_y, 'r-', label='Projected')
    ticks = np.arange(min(log_model_x), max(log_model_x), step=7)
    plt.xticks(ticks, rotation=90)
    ax.set_xticklabels(map(lambda xt: line_format(xt.astype(dt.datetime)), ticks))
    plt.xlabel('Date')
    plt.ylabel('Total Cases')
    plt.title('Knoxville Metro COVID19 Projected Cumulative Cases -- Updated: {}'.format(time_now()))
    plt.legend()
    # plt.annotate(
    #     'Max Cases: {:.0f}\n'
    #     'Approx. {:.1f}x current\n'
    #     'Rollover Date: {}'.format(max(log_model_y),
    #                                log_model_params['ratio'],
    #                                log_model_params['rollover_date']),
    #     xytext=(0.75, 0.75), textcoords='figure fraction',
    #     horizontalalignment='right', verticalalignment='top',
    #     xy=(log_model_params['rollover_date'], log_model_params['rollover_date_coords'][1]), xycoords='data',
    #     arrowprops=dict(facecolor='black', shrink=0.05))
    plt.annotate(attribution['text'], (0, 0), (0, -70),
                 xycoords='axes fraction',
                 textcoords='offset points',
                 fontsize=attribution['fsize'],
                 color=attribution['color'],
                 alpha=attribution['alpha']
                 )
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, 'metro-cases-all-fit-best.png'))


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
    forecast = args.forecast

    # file definitions
    fips_datafile = os.path.join(datadir, 'tn/fips.csv')
    metro_datafile = os.path.join(datadir, 'tn/metro.json')
    hospitals_datafile = os.path.join(datadir, 'tn/tn-hospitals.geojson')
    nytimes_datafile = os.path.join(datadir, 'ny-times/us-counties.csv')
    jhu_datafiles = glob.glob(os.path.join(datadir, 'jhu_csse/csse_covid_19_data/csse_covid_19_daily_reports/*.csv'))
    midas_datafile = os.path.join(datadir, 'midas/parameter_estimates/2019_novel_coronavirus/estimates.csv')
    readme_file = os.path.join(os.path.dirname(__file__), '../README.md')

    # remove old files
    remove_tmp_files(imgdir)

    # suppress Stan/FB Prophet verbose output
    logging.getLogger('fbprophet').setLevel(logging.WARNING)

    # figure data attribution
    attribution = {
        'text': 'Data from the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University (JHU)',
        'fsize': 10,
        'color': '#000000',
        'alpha': 0.33
    }

    # process local TN data
    knx_df = process_tn_data(fips_datafile, metro_datafile, hospitals_datafile, nytimes_datafile, jhu_datafiles,
                             drive_time)

    # plot cases per county per day
    plot_county_cases_per_day(knx_df, imgdir, attribution)

    # plot aggregate cases per day for KNX metro
    plot_metro_cases_per_day(knx_df, imgdir, attribution)

    # best case logistic fit
    log_model_x, log_model_y, log_model_params = logistic_forecast(knx_df, ndays, growth_rate, time_horizon)

    # plot logistic model best case scenario
    plot_logistic_model(knx_df, log_model_x, log_model_y, log_model_params, imgdir, attribution)

    # run forecasts
    if forecast:
        # prophet forecast of daily cases
        daily_cases_fb_forecast(knx_df, log_model_params['days_out'], imgdir, attribution)

        # process midas data
        midas_params = process_midas_data(midas_datafile)

        # worst case prophet model
        worst_case_fb_forecast(knx_df, knx_capacity, log_model_params['days_out'], imgdir, attribution)

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
