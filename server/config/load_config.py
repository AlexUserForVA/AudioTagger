import os
import csv

from server.config.config import PROJECT_ROOT

def loadPredictors():
    with open(os.path.join(PROJECT_ROOT, 'server/config/predictors.csv')) as file:
        csvReader = csv.reader(file, delimiter=';')
        header = next(csvReader, None)  # skip header
        return [{header[0] : int(line[0]), header[1] : line[1], header[2] : line[2], header[3] : line[3], header[4] : line[4]} for line in csvReader]

def loadSources():
    with open(os.path.join(PROJECT_ROOT, 'server/config/sources.csv')) as file:
        csvReader = csv.reader(file, delimiter=';')
        header = next(csvReader, None)  # skip header
        return [{header[0] : int(line[0]), header[1] : line[1], header[2] : line[2]} for line in csvReader]