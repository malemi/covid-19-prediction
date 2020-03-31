#!/usr/bin/env python3
import csv
import urllib.request

base_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
filename = "time_series_covid19_confirmed_global.csv"
print(f"Reading {filename}")
urllib.request.urlretrieve(base_url + filename, filename)
print("Done")
with open(filename) as f:
    lines = csv.reader(f, delimiter=',', quotechar='"')
    for i, line in enumerate(lines):
        if i == 0:
            keys = line
            confirmed = {}
            continue
        for j, v in enumerate(line):

            if j > 3:
                confirmed.setdefault(" ".join(line[0:2]).strip(), {})[keys[j]] = v

filename = "time_series_covid19_deaths_global.csv"
print(f"Reading {filename}")
urllib.request.urlretrieve(base_url + filename, filename)
print("Done")
with open(filename) as f:
    lines = csv.reader(f, delimiter=',', quotechar='"')
    for i, line in enumerate(lines):
        if i == 0:
            keys = line
            deaths = {}
            continue
        for j, v in enumerate(line):

            if j > 3:
                deaths.setdefault(" ".join(line[0:2]).strip(), {})[keys[j]] = v

filename = "time_series_covid19_recovered_global.csv"
print(f"Reading {filename}")
urllib.request.urlretrieve(base_url + filename, filename)
print("Done")
with open(filename) as f:
    lines = csv.reader(f, delimiter=',', quotechar='"')
    for i, line in enumerate(lines):
        if i == 0:
            keys = line
            recovered = {}
            continue
        for j, v in enumerate(line):

            if j > 3:
                recovered.setdefault(" ".join(line[0:2]).strip(), {})[keys[j]] = v

total = []
for country, kv in confirmed.items():
    try:
        for date, value in kv.items():
            date_clean = "2020-" + "{:02d}".format(int(date.split("/")[0])) + "-" + "{:02d}".format(int(date.split("/")[1]))
            total.append(",".join(['"'+country+'"', date_clean, value]) + "," + deaths[country][date] + "," + recovered[country][date] + "," + str(int(value)-int(deaths[country][date])-int(recovered[country][date])))
    except KeyError:
        print(f"Error on {country}")

with open("time_series_adjusted.csv", "w") as out:
    out.write("country,date,confirmed,deaths,recovered,infectious\n")
    for t in total:
        out.write(t + "\n")
