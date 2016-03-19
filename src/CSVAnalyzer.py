import csv
import sys
import glob
import argparse
import json
import numpy as np
from os import listdir


class CSVAnalyser:
    def __init__(self, csvfile = None):
        self.csvfile = csvfile

    def analyze(self):
        csv_reader = csv.reader(self.csvfile, delimiter=',')
        next(csv_reader, None)  # skip the headers
        data = [data for data in csv_reader]
        data_array = np.asarray(data)

        print(data_array)



def main():
    parser = argparse.ArgumentParser(prog='CSVAnalyzer')
    parser.add_argument('csvfile', type=argparse.FileType('r'), help='CSV format <Stud|Rating|Link|Comment>')
    args = parser.parse_args()

    print("csvfile" + str(args.csvfile))
    csv_analyser = CSVAnalyser(args.csvfile)
    csv_analyser.analyze()


if __name__ == "__main__":
    # execute only if run as a script
    main()