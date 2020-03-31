# covid-19-prediction

Prediction of the evolution of a pandemic using a simplified Susceptible-Infectious-Recovered model

## Run it

Run:

```bash
./covid-analysis.py  --help
```

Help:
```
usage: covid-analysis.py [-h] [--url URL] [--name-dataset NAME_DATASET]
                         [--regions-field REGIONS_FIELD]
                         [--date-field DATE_FIELD]
                         [--variable-name VARIABLE_NAME]
                         [--start-days-ago START_DAYS_AGO]
                         [--days-analyzed DAYS_ANALYZED]
                         [--days-forecasted DAYS_FORECASTED]
                         [--smoothing SMOOTHING] [--json-name JSON_NAME]
                         [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --url URL             URL or filename. Must be a CSV.
  --name-dataset NAME_DATASET
                        Prefix of files/directories created.
  --regions-field REGIONS_FIELD
                        In the CSV, the field with regions.
  --date-field DATE_FIELD
                        In the CSV, the field with the dates.
  --variable-name VARIABLE_NAME
                        In the CSV, the field with the variable we want to predict.
  --start-days-ago START_DAYS_AGO
                        Run forecasts also in the past, starting N days ago.
  --days-analyzed DAYS_ANALYZED
                        Number of days used for the fit
  --days-forecasted DAYS_FORECASTED
                        How many days in the future.
  --smoothing SMOOTHING
                        How many days used for smoothing (poss >= DAYS_ANALYZED).
  --json-name JSON_NAME
                        Name of the JSON file with all data.
  --debug
```  

## Origin

See: [COVID-19: Evidence that Italy has done it right](https://medium.com/@malemi/covid-19-evidence-that-italy-has-done-it-right-eda758309f58) (English)
or (Stima veloce dell’andamento del numero di ricoverati con sintomi COVID-19 in Italia.)[https://medium.com/@malemi/stima-veloce-dei-ricoverati-con-sintomi-covid-19-in-lombardia-491a0c3f4a7b].
