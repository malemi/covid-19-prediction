#!/usr/bin/env python3
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
import json
import subprocess
import argparse
import time
import re


def make_name(text): return ''.join(re.findall(r'\w+', text.lower()))


def r_curve(day, par0, par1):
    # return par1-0.925 + day * par0
    return par1*np.exp(par0*day) + 0.9


def const(day, par1):
    return par1


def growth(prev, rate):
    return prev * rate


def abs_error_rate(cov, day, par, rate, curve: str):
    """
    General:
    var_

    Linear:
    r = par0*day + par1 = r(par0, par1)
    var_r = cov_00*(dr/d0)^2 + cov_11*(dr/d1)^2 + 2cov_10*dr/d0*dr/d1 =
          = cov_00*day^2 + cov_11 + 2cov_10*day


    """
    if curve == "linear":
        result = (cov[0, 0]*day**2 + 2.*cov[0, 1]*day + cov[1, 1])**0.5
    elif curve == "exponential":
        result = (np.exp(2*par[0]*day)*cov[1, 1] +
                  par[1]*day*np.exp(par[0]*day)*cov[0, 0] +
                  2*cov[1, 0]*par[1]*day*np.exp(3*par[0]*day))**0.5
    else:
        result = None
    # print(f"ERROR for day {day} at rate {rate}: {result}. COV: {repr(cov)}")
    return result


def future_rate_with_error(par, cov, start_day, days_forecast):
    """

    :param par:
    :param cov:
    :param start_day:
    :param days_forecast:
    :return:
    """
    future_rate_best = []
    future_rate_p_sigma =[]
    future_rate_m_sigma = []
    for day in range(start_day, start_day + days_forecast):
        day_rate = r_curve(day, par[0], par[1])
        day_err = abs_error_rate(cov=cov, day=day, par=par, rate=day_rate, curve="exponential")
        max_err = day_rate + day_err
        min_err = day_rate - day_err
        future_rate_best.append(day_rate)
        future_rate_p_sigma.append(max_err)
        future_rate_m_sigma.append(min_err)
    return np.array(future_rate_best), np.array(future_rate_p_sigma), np.array(future_rate_m_sigma)


def future_infectious_with_error(yesterday_infectious,
                                 future_rate_best,
                                 future_rate_p_sigma,
                                 future_rate_m_sigma,
                                 start_day,
                                 days_forecast):
    """
    Computes for each day after start_day the forecast number of ospedalizzati
    with upper and lower limit.

    future_rate_best[0] is the forecasted rate for today (for which we don't have data yet)

    yesterday_ospedalizzati: last observed "ospedalizzati" the day before start_day
    n_days: number of days we want to forecast
    """

    future_infectious = [growth(yesterday_infectious, future_rate_best[0])]
    future_infectious_p_limit = [growth(yesterday_infectious, future_rate_p_sigma[0])]
    future_infectious_m_limit = [growth(yesterday_infectious, future_rate_m_sigma[0])]

    for day in range(start_day+1, start_day + days_forecast):
        rate_error = future_rate_p_sigma[day-start_day]-future_rate_best[day-start_day]
        future_infectious_now = growth(future_infectious[day-1-start_day], future_rate_best[day-start_day])
        future_infectious.append(future_infectious_now)
        # I=N*exp(r'*day) with r'=r-1
        # dI = day*I*dr
        future_infectious_error = 3*(day-start_day)**0.5*future_infectious_now*rate_error
        future_infectious_p_limit.append(future_infectious_now+future_infectious_error)
        future_infectious_m_limit.append(future_infectious_now-future_infectious_error)
        # future_infectious_p_limit.append(growth(future_infectious[day - 1], future_rate_best[day] + rate_error))
        # future_infectious_m_limit.append(growth(future_infectious[day - 1], future_rate_best[day] - rate_error))
    return np.array(future_infectious), np.array(future_infectious_p_limit), np.array(future_infectious_m_limit)


def make_prediction(url,
                    name_dataset,
                    regions_field,
                    date_field,
                    variable_name,
                    start_days_ago,
                    days_analyzed=14,  # last days analyzed
                    days_forecasted=60,
                    filter_regions=None,
                    smoothing=7,
                    json_name="all_data.json",
                    debug=False,
                    average_results=0):

    max_y_axis = {}  # record the maximum to plot forecasts
    infected_img_names = {}
    datum = {"dataset": name_dataset, "url": url}
    df = pd.read_csv(url)

    regions = set(df[regions_field].to_list())
    if filter_regions is not None:
        regions = regions.intersection(set(filter_regions))
        if len(regions) == 0:
            exit(f"No region found: {filter_regions}")

    for r in regions:
        infected_img_names[r] = []

    global_future_infected = {}

    for days_ago in range(start_days_ago-1, -1, -1):

        assert days_ago >= 0

        if smoothing > 1:
            dir_prefix = name_dataset + "_smooth_"
        else:
            dir_prefix = name_dataset + "_real_"

        for region in regions:
            print(f"Region: {region}")
            infected = df[df[regions_field] == region][variable_name].to_numpy()
            maxy = max(infected)*5.
            total_days_available = len(infected)
            # We need at least days_ago days for the fit:
            if total_days_available <= days_ago:
                print(f"Not enough days: total_days_available={total_days_available}, days_ago={days_ago}")
                continue
            infected = infected[:total_days_available-days_ago]

            orig_infected = infected.copy()
            if smoothing > 1:
                smoothed_infected = np.zeros(len(infected))
                for i in range(len(infected)):
                    if i == 0:
                        smoothed_infected[i] = infected[i]
                    if infected[i] > 0:
                        days_smoothed = min(i, smoothing)
                        #not_norm_w = np.array([2 ** (-i) for i in range(0, days_smoothed)][::-1])
                        not_norm_w = np.array([1./np.sqrt(i) for i in range(1, days_smoothed+1)][::-1])
                        weights = not_norm_w / not_norm_w.sum()
                        smoothed_infected[i] = (infected[i-days_smoothed+1:i+1] * weights).sum()

                smoothed_infected = smoothed_infected * infected.sum() / smoothed_infected.sum()
                not_smoothed_infected = infected.copy()
                infected = smoothed_infected.copy()

            csv_dates = df[df[regions_field] == region][date_field].to_numpy()
            csv_dates = csv_dates[:total_days_available-days_ago]
            giorni = [datetime.datetime.fromisoformat(d).strftime("%d %b") for d in csv_dates]

            first_day_str = datetime.datetime.fromisoformat(csv_dates[0])
            last_day_str = datetime.datetime.fromisoformat(csv_dates[-1]).strftime("%Y%m%d")
            dir_name = dir_prefix + last_day_str
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            print(f"last day: {last_day_str}")

            all_dates_strings = [(first_day_str + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(giorni) + days_forecasted)]

            datum.setdefault(region, {})[last_day_str] = {}

            datum[region][last_day_str]["smoothing"] = smoothing

            datum[region][last_day_str] [variable_name] = infected.tolist()
            if smoothing:
                datum[region][last_day_str]["not_smoothed_" + variable_name] = not_smoothed_infected.tolist()
            datum[region][last_day_str]["all_dates"] = all_dates_strings

            dates = [datetime.datetime.fromisoformat(d) for d in df[df[regions_field] == region][date_field].to_numpy()[:total_days_available-days_ago]]

            rate = [0.0]  # Rate=0 are not considered anyway. But so we have values since day=0

            for i, h in enumerate(infected):
                if i == 0:
                    continue
                if infected[i - 1] == 0:
                    rate.append(0)
                else:
                    rate.append(h / infected[i - 1])

            sigma = [100.0]  # in reality is inf.
            for i, r in enumerate(rate[1:]):
                yesterday = infected[i - 1]
                today = infected[i]
                if today == 0 or yesterday == 0:
                    error = 100.0  # in reality is inf.
                else:
                    # r = y/x  ==> dr = dy/x + y/x^2 dx
                    # error = (today**0.5/yesterday) + (today/yesterday**2 * yesterday**0.5)
                    error = (1./today + 1./yesterday)**0.5
                    # the number behind hospitalised (i.e. infected) is high, so the error is actually small even
                    # when we have a few units. Having the "true" stat error gives too much weight to recent data

                sigma.append(error)

            # beginning of epidemic
            buffer = []
            beginning_epidemic = len(rate) - days_analyzed
            found_start = False
            for i, r in enumerate(rate):
                buffer.append(r)
                if min(buffer[-4:]) > 1.1:
                    beginning_epidemic = max(i, beginning_epidemic)
                    found_start = True
                    break

            if len(rate) - beginning_epidemic < max(7, smoothing) or not found_start:
                print(f"Less than 7 days for {region}, or beginning not found. Skipping")

            else:

                try:
                    par, cov = curve_fit(f=r_curve,
                                         xdata=range(beginning_epidemic, len(rate)),
                                         ydata=rate[beginning_epidemic:],
                                         sigma=sigma[beginning_epidemic:],
                                         bounds=[(-0.5, 0), (1, 100)],
                                         maxfev=1000)
                except Exception:
                    print("===============Minimization does not converge=================")
                    par = np.array([0., np.array(rate[beginning_epidemic:]).mean()-0.9])
                    cov = np.array([[0., 0.], [0., 0.]])

                if par[0] > 0:
                    par = np.array([0., np.array(rate[beginning_epidemic:]).mean()-0.9])
                    cov = np.array([[0., 0.], [0., 0.]])

                future_rate_best, future_rate_p_sigma, future_rate_m_sigma = future_rate_with_error(par=par,
                                                                                                    cov=cov,
                                                                                                    start_day=len(rate),
                                                                                                    days_forecast=days_forecasted)

                future_infected_best, future_infected_p_sigma, future_infected_m_sigma = future_infectious_with_error(
                    yesterday_infectious=orig_infected[-1],
                    future_rate_best=future_rate_best,
                    future_rate_p_sigma=future_rate_p_sigma,
                    future_rate_m_sigma=future_rate_m_sigma,
                    start_day=len(rate),
                    days_forecast=days_forecasted)

                ave_done = False
                if average_results > 0:
                    global_future_infected.setdefault(region, {})[days_ago] = future_infected_best
                    if start_days_ago-1-days_ago > average_results:
                        try:
                            tmp_f_i_b = global_future_infected[region][days_ago][:-(average_results-1)]
                            for d in range(1, average_results-1):
                                tmp_f_i_b = tmp_f_i_b + global_future_infected[region][days_ago + d][d:-(average_results-1-d)]
                            tmp_f_i_b = tmp_f_i_b + global_future_infected[region][days_ago + average_results-1][average_results-1:]
                            tmp_f_i_b = tmp_f_i_b / average_results
                            future_infected_best_ave = np.concatenate((tmp_f_i_b, future_infected_best[-2:]))
                            ave_done = True
                        except KeyError:
                            print("NOT ENGOUGH")
                            pass
                if debug:
                    print(f"Infectious (original): {repr(orig_infected)}")
                    print(f"Infectious (smooth, if): {repr(infected)}")
                    print(f"Future Infectious for {len(future_infected_best)} days: {future_infected_best}")
                    if average_results > 0 and ave_done:
                        print(f"Future Infectious (averaged): {repr(future_infected_best_ave)}")
                    print(f"Error on future Infectious: {future_infected_p_sigma-future_infected_best}")
                    print(f"Rate: {repr(rate)}")
                    print(f"Future rate: {future_rate_best}")
                    print(f"Error on future rate: {future_rate_p_sigma-future_rate_best}")
                    print(f"Fit done: par={repr(par)}, cov={repr(cov)}, len(rate)={len(rate)}, len(future rate)={len(future_rate_best)}")

                present = len(infected)  # first forecasted day
                far_future = present + days_forecasted
                peak = np.argmax(future_infected_best)
                peak_date_string = (dates[-1] + datetime.timedelta(days=int(peak))).strftime("%d %b")
                axes = plt.gca()
                axes.set_ylim([0.0, max(future_infected_best) * 1.2])

                datum[region][last_day_str]["forecasts_" + variable_name] = future_infected_best.tolist()
                plt.ylim(0, maxy)
                plt.plot(giorni, orig_infected, 'b-')
                if ave_done:
                    plt.plot(range(present, far_future), future_infected_best_ave, 'r-')
                else:
                    plt.plot(range(present, far_future), future_infected_best, 'r-')
                plt.plot(range(present, far_future), future_infected_p_sigma, 'g-')
                plt.plot(range(present, far_future), future_infected_m_sigma, 'g-')
                plt.fill_between(range(present, far_future), future_infected_m_sigma, future_infected_p_sigma, facecolor="gray", alpha=0.15)
                plt.xticks(range(0, far_future), [giorni[0]] + ['' for i in range(1, len(giorni)-1)] + [giorni[-1]] +
                               ['' for i in range(peak)] + [peak_date_string] +
                               ['' for i in range(far_future-present-peak)], rotation=45)

                plt.title("Forecasts " + variable_name.replace("_", " ") + " " + region)
                forecasts_filename = dir_name + "/forecasts_" + make_name(region) + last_day_str + ".png"
                if os.path.exists(forecasts_filename):
                    os.remove(forecasts_filename)
                plt.savefig(forecasts_filename)
                infected_img_names[region].append(forecasts_filename)
                plt.clf()
                plt.xticks([])

                datum[region][last_day_str]["rate"] = rate
                datum[region][last_day_str]["forecasts_rate"] = future_rate_best.tolist()
                plt.ylim(0.75, 2.5)
                plt.plot(giorni, rate, 'b-')
                plt.plot(range(present, far_future), future_rate_best, 'r-')
                plt.plot(range(present, far_future), future_rate_p_sigma, 'g-')
                plt.plot(range(present, far_future), future_rate_m_sigma, 'g-')
                plt.title("Rate " + variable_name.replace("_", " ") + " " + region)
                rate_filename = dir_name + "/rate_" + make_name(region) + last_day_str + ".png"
                if os.path.exists(rate_filename):
                    os.remove(rate_filename)
                plt.savefig(rate_filename)
                plt.clf()
                plt.xticks([])
                plt.close()
    with open(json_name, "w") as f:
        json.dump(datum, f)

    animated_dir = name_dataset + "_animated"
    if not os.path.exists(animated_dir):
        os.makedirs(animated_dir)

    for region in regions:
        animated_filename = f"{animated_dir}/{name_dataset}_{make_name(region)}.gif"
        if os.path.exists(animated_filename):
            os.remove(animated_filename)
        print(f"Creating animated image {animated_filename}")
        cmd = f"convert  -reverse -delay 100 -loop 0 {' '.join(infected_img_names[region][::-1])} {animated_filename}"
        time.sleep(0.5)
        try:
            subprocess.call(cmd, shell=True)
        except FileNotFoundError:
            print("'convert' (ImageMagick) not installed?")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, help="url",
                        default="https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv")
    parser.add_argument("--name-dataset", type=str, help="", default="protezione_civile")
    parser.add_argument("--regions-field", type=str, help="", default="denominazione_regione")
    parser.add_argument("--date-field", type=str, help="", default="data")
    parser.add_argument("--variable-name", type=str, help="", default="totale_ospedalizzati")
    parser.add_argument("--start-days-ago", type=int, help="", default="7")
    parser.add_argument("--days-analyzed", type=int, help="", default="14")
    parser.add_argument("--days-forecasted", type=int, help="", default="60")
    parser.add_argument("--smoothing",  type=int, help="", default="7")
    parser.add_argument("--json-name", type=str, help="", default="all_data.json")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    make_prediction(url=args.url,
                    name_dataset=args.name_dataset,
                    regions_field=args.regions_field,
                    date_field=args.date_field,
                    variable_name=args.variable_name,
                    start_days_ago=args.start_days_ago,
                    days_analyzed=args.days_analyzed,  # last days analyzed
                    days_forecasted=args.days_forecasted,
                    filter_regions=["Hubei China"], #["Italy"], #["Hubei China"], ["Lombardia"],
                    smoothing=args.smoothing,
                    json_name=args.json_name,
                    debug=args.debug,
                    average_results=0)
