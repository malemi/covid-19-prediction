import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import os
import json

url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv"
df = pd.read_csv(url)

debug = False
smoothing = True
days_analyzed = 210  # last days analyzed
days_forecasted = 60
asymptotic_rate = 0.925
max_asymptotic_rate = 0.95
min_asymptotic_rate = 0.9


def r_curve(day, par0, par1):
    # return par1-0.925 + day * par0
    return par1*np.exp(par0*day) + 0.9


def const(day, par1):
    return par1


def growth(prev, rate):
    return prev * rate


def abs_error_rate(cov, day, par, rate, curve: str):
    """
    TODO it was for linear fit
    r = par0*day + par1 = r(par0, par1)
    var_r = cov_00*(dr/d0)^2 + cov_11*(dr/d1)^2 + 2cov_10*dr/d0*dr/d1 =
          = cov_00*day^2 + cov_11 + 2cov_10*day
    """
    if curve == "linear":
        result = (cov[0, 0]*day**2 + 2.*cov[0, 1]*day + cov[1, 1])**0.5
    elif curve == "exponential":
        result = np.sqrt(par[0]**2 * day**2 * cov[0, 0])
    else:
        result = None
    # print(f"ERROR for day {day} at rate {rate}: {result}. COV: {repr(cov)}")
    return result


def future_rate_with_error(par, cov, start_day, last_rate, days_forecast):
    """

    :param par:
    :param cov:
    :param start_day:
    :param last_rate: the rate today will be a linear approx from last_rate (no abrupt variations)
    :param days_forecast:
    :return:
    """
    future_rate_best = []
    future_rate_p_sigma =[]
    future_rate_m_sigma = []
    for day in range(start_day, start_day + days_forecast):
        # For linear, so that it is close to the last rate
        # day_rate = (day - start_day + 1) * par[0] + last_rate
        day_rate = r_curve(day, par[0], par[1])
        day_err = abs_error_rate(cov=cov, day=start_day, par=par, rate=day_rate, curve="exponential")
        max_err = day_rate + day_err
        min_err = day_rate - day_err
        # if day_rate < asymptotic_rate:  # for the linear one
        #     day_rate = asymptotic_rate
        future_rate_best.append(day_rate)
        # if max_err < 0.95:
        #     max_err = 0.95
        future_rate_p_sigma.append(max_err)
        # if min_err < 0.9:
        #     min_err = 0.9
        future_rate_m_sigma.append(min_err)
        #future_rate_p_sigma.append(r_curve(day, par[0]+cov[0, 0]**0.5, par[1]+cov[1, 1]**0.5))
        #future_rate_m_sigma.append(r_curve(day, par[0]-cov[0, 0]**0.5, par[1]-cov[1, 1]**0.5))
    return np.array(future_rate_best), np.array(future_rate_p_sigma), np.array(future_rate_m_sigma)


def future_ricoverati_with_error(yesterday_ricoverati,
                                 future_rate_best,
                                 future_rate_p_sigma,
                                 future_rate_m_sigma,
                                 days_forecast):
    """
    Computes for each day after start_day the forecast number of ospedalizzati
    with upper and lower limit.

    future_rate_best[0] is the forecasted rate for today (for which we don't have data yet)

    yesterday_ospedalizzati: last observed "ospedalizzati" the day before start_day
    n_days: number of days we want to forecast
    """

    future_ricoverati = [growth(yesterday_ricoverati, future_rate_best[0])]
    future_ricoverati_p_limit = [growth(yesterday_ricoverati, future_rate_p_sigma[0])]
    future_ricoverati_m_limit = [growth(yesterday_ricoverati, future_rate_m_sigma[0])]

    for day in range(1, days_forecast):
        future_ricoverati.append(growth(future_ricoverati[day-1], future_rate_best[day]))
        future_ricoverati_p_limit.append(growth(future_ricoverati_p_limit[day-1],
                                                future_rate_p_sigma[day]))
        future_ricoverati_m_limit.append(growth(future_ricoverati_m_limit[day-1],
                                                future_rate_m_sigma[day]))
    return np.array(future_ricoverati), np.array(future_ricoverati_p_limit), np.array(future_ricoverati_m_limit)

regioni = ["Piemonte", "Valle d'Aosta", "Lombardia", "P.A. Bolzano", "P.A. Trento",
           "Veneto", "Friuli Venezia Giulia", "Liguria", "Emilia Romagna",
           "Toscana", "Umbria", "Marche", "Lazio", "Abruzzo", "Molise",
           "Campania", "Puglia", "Basilicata", "Calabria", "Sicilia", "Sardegna"]

variable_name = 'totale_ospedalizzati' #'ricoverati_con_sintomi'
max_y_axis = {}  # record the maximum to plot forecasts

for days_ago in range(0, 7):

    assert days_ago >= 0
    #delta = datetime.timedelta(days=last_day)
    #str_date = (datetime.datetime.now() - delta).strftime("%Y%m%d")

    if smoothing:
        dir_prefix = "smooth_"
    else:
        dir_prefix = "real_"

    for denominazione_regione in regioni:
        datum = {}
        datum["smoothing"] = smoothing

        datum["denominazione_regione"] = denominazione_regione
        infected = df[df['denominazione_regione'] == denominazione_regione][variable_name].to_numpy()
        total_days_available = len(infected)
        infected = infected[:total_days_available-days_ago]
        datum[variable_name] = infected.tolist()

        orig_infected = infected.copy()
        if smoothing:
            new_infected = np.zeros(len(infected))
            weights3 = (.15, .25, .6)
            weights2 = (.3, .7)
            for i in range(len(infected)):
                if i == 0:
                    new_infected[i] = infected[0]
                elif i == 1:
                    new_infected[i] = infected[i - 1] * weights2[0] + \
                                      infected[i] * weights2[1]
                else:
                    new_infected[i] = \
                        infected[i - 2] * weights3[0] + \
                        infected[i - 1] * weights3[1] + \
                        infected[i] * weights3[2]
            new_infected = new_infected * infected.sum() / new_infected.sum()
            infected = new_infected.copy()

        csv_dates = df[df['denominazione_regione'] == denominazione_regione]['data'].to_numpy()
        csv_dates = csv_dates[:total_days_available-days_ago]
        giorni = [datetime.datetime.fromisoformat(d).strftime("%d %b") for d in csv_dates]

        first_day_str = datetime.datetime.fromisoformat(csv_dates[0])
        last_day_str = datetime.datetime.fromisoformat(csv_dates[-1]).strftime("%Y%m%d")
        dir_name = dir_prefix + last_day_str
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        print(f"Regione: {denominazione_regione}, last day: {last_day_str}")

        all_dates_strings = [(first_day_str + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(len(giorni) + days_forecasted)]

        datum["all_dates"] = all_dates_strings

        dates = [datetime.datetime.fromisoformat(d) for d in df[df['denominazione_regione'] == denominazione_regione]['data'].to_numpy()[:total_days_available-days_ago]]

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
        beginning_epidemic = len(rate) - days_analyzed
        for i, r in enumerate(infected):
            if infected[i] > 0:
                beginning_epidemic = max(i, beginning_epidemic)
                break

        try:
            par, cov = curve_fit(f=r_curve,
                             xdata=range(beginning_epidemic, len(rate)),
                             ydata=rate[beginning_epidemic:],
                             sigma=sigma[beginning_epidemic:],
                             maxfev=1000)
        except RuntimeError:
            print("===============Minimization does not converge=================")
            par = np.array([0., np.array(rate).mean()])
            cov = np.array([[0., 0.], [0., 0.]])

        fitted_rate = par[0]
        fitted_offset = par[1]

        if par[0] > 0:
            par_const, cov_const = curve_fit(const, range(beginning_epidemic, len(rate)), rate[beginning_epidemic:], sigma=sigma[beginning_epidemic:])
            par = np.array([0., par_const[0]])
            cov = np.array([[cov_const[0, 0], 0], [0., 0.]])

        future_rate_best, future_rate_p_sigma, future_rate_m_sigma = future_rate_with_error(par=par,
                                                                                            cov=cov,
                                                                                            start_day=len(rate),
                                                                                            last_rate=rate[-1],
                                                                                            days_forecast=days_forecasted)

        future_ricoverati_best, future_ricoverati_p_sigma, future_ricoverati_m_sigma = future_ricoverati_with_error(
            yesterday_ricoverati=infected[-1],
            future_rate_best=future_rate_best,
            future_rate_p_sigma=future_rate_p_sigma,
            future_rate_m_sigma=future_rate_m_sigma,
            days_forecast=days_forecasted)

        if debug:
            print(f"Ricoverati (original): {repr(orig_infected)}")
            print(f"Ricoverati (smooth, if): {repr(infected)}")
            print(f"Future ricoverati for {len(future_ricoverati_best)} days: {future_ricoverati_best}")
            print(f"Rate: {repr(rate)}")
            print(f"Future rate: {future_rate_best}")
            print(f"Error on rate: {future_rate_p_sigma-future_rate_best}")
            print(f"Fit done: par={repr(par)}, cov={repr(cov)}, len(rate)={len(rate)}, len(future rate)={len(future_rate_best)}")

        present = len(infected)  # first forecasted day
        far_future = present + days_forecasted
        peak = np.argmax(future_ricoverati_best)
        peak_date_string = (dates[-1] + datetime.timedelta(days=int(peak))).strftime("%d %b")
        axes = plt.gca()
        axes.set_ylim([0.0, max(future_ricoverati_best) * 1.2])

    #    try:
        datum["stima_" + variable_name] = future_ricoverati_best.tolist()
        maxy = max_y_axis.setdefault(denominazione_regione, min(20000, 5 * max(infected)))
        plt.ylim(0, maxy)
        plt.plot(giorni, infected, 'b-')
        plt.plot(range(present, far_future), future_ricoverati_best, 'r-')
        plt.plot(range(present, far_future), future_ricoverati_p_sigma, 'g-')
        plt.plot(range(present, far_future), future_ricoverati_m_sigma, 'g-')
        plt.fill_between(range(present, far_future), future_ricoverati_m_sigma, future_ricoverati_p_sigma, facecolor="gray", alpha=0.15)
        plt.xticks(range(0, far_future), [giorni[0]] + ['' for i in range(1, len(giorni)-1)] + [giorni[-1]] +
                       ['' for i in range(peak)] + [peak_date_string] +
                       ['' for i in range(far_future-present-peak)], rotation=45)

        plt.title("Previsione " + variable_name.replace("_", " ") + " " + denominazione_regione)
        plt.savefig(dir_name + "/previsione_" + denominazione_regione.replace(" ", "").lower() + last_day_str + ".png")
        plt.clf()
        plt.xticks([])

        #    plt.show()
        datum["rate"] = rate
        datum["stima_rate"] = future_rate_best.tolist()
        plt.ylim(0.75, 2.5)
        plt.plot(giorni, rate, 'b-')
        plt.plot(range(present, far_future), future_rate_best, 'r-')
        plt.plot(range(present, far_future), future_rate_p_sigma, 'g-')
        plt.plot(range(present, far_future), future_rate_m_sigma, 'g-')
        plt.title("Rate " + variable_name.replace("_", " ") + denominazione_regione)
        plt.savefig(dir_name + "/rate_" + denominazione_regione.replace(" ", "").lower() + last_day_str + ".png")
        plt.clf()
        plt.xticks([])
        plt.close()
        with open(dir_name + "/" + denominazione_regione.replace(" ", "").lower() + ".json", "w") as f:
            json.dump(datum, f)
            ## plt.show()
        # except Exception as e:
        #     print("ERRORE " + denominazione_regione)
