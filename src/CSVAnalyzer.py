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
        self.imported_data = None
        np.set_printoptions(threshold=np.nan)

    def analyze(self):
        csv_reader = csv.reader(self.csvfile, delimiter=',')
        next(csv_reader, None)  # skip the headers
        data = [data for data in csv_reader]
        self.imported_data = np.asarray(data)
        self.remove_invalid_entries()


    def remove_invalid_entries(self):
        print(self.imported_data)
        self.imported_data = self.imported_data[np.all(self.imported_data != '', axis=1)]
        print(self.imported_data)



def main():
    parser = argparse.ArgumentParser(prog='CSVAnalyzer')
    parser.add_argument('csvfile', type=argparse.FileType('r'), help='CSV format <Stud|Rating|Link|Comment>')
    args = parser.parse_args()

    csv_analyser = CSVAnalyser(args.csvfile)
    csv_analyser.analyze()


if __name__ == "__main__":
    # execute only if run as a script
    main()