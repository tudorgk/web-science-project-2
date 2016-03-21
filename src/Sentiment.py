import csv
import sys
import glob
import argparse
import json
from io import StringIO
import ast
import pycurl
import numpy as np
import re
from io import BytesIO
from urllib.parse import urlencode

class SentimentV1:
    def __init__(self, data = None):
        self.data = data
        self.results = []
        self.downloaded_results = []
        self.sentimentV1_data = []
        self.index_to_remove = []


    def download_sentiments(self):
        print("Download sentiment data from webservice")
        f1 = open('../tmp/results.txt', 'w+')

        for i in range(self.data[:, ].shape[0]):
            print("Downloading entry" + i)
            escaped = re.escape(self.data[i, 1].encode('utf-8'))

            post_data = {'text': escaped}

            buffer = BytesIO()
            c = pycurl.Curl()
            c.setopt(c.URL, 'http://text-processing.com/api/sentiment/')
            c.setopt(c.POSTFIELDS, urlencode(post_data))
            c.setopt(c.WRITEDATA, buffer)
            c.perform()
            c.close()

            body = buffer.getvalue()
            # Body is a byte string.
            # We have to know the encoding in order to print it to a text file
            # such as standard output.
            response = body.decode('utf-8')
            self.results.append(response)

        f1.write(str(self.results))

    def import_mined_json(self):
        print("Import json data")
        f1 = open('../tmp/results.txt', 'r')
        result_string = f1.read()
        self.downloaded_results = ast.literal_eval(result_string)
        f1.close()

    def construct_arrays(self):
        print("Constructing arrays")
        print(self.data.shape)
        print(len(self.downloaded_results))
        for i in range(self.data[:, ].shape[0]):
            # using try-catch because some downloaded results don't have data associated because of some kind of form
            # error. (Form Validation Errors text: This field is required.)
            try:
                json_row = json.loads(self.downloaded_results[i])
                if json_row['label'] == "pos":
                    raw_value = 1
                elif json_row['label'] == "neg":
                    raw_value = -1
                else:
                    raw_value = 0

                row = [self.data[i, 0],raw_value]
                self.sentimentV1_data.append(row)

            except:
                # save the indices that we need to remove for the second sentiment mining
                self.index_to_remove.append(i)
                pass

        print(self.sentimentV1_data)

    def run_analysis(self):
        print("Run analysis")
        #self.download_sentiments()
        self.import_mined_json()
        self.construct_arrays()



class CSVAnalyser:
    def __init__(self, csvfile = None):
        self.csvfile = csvfile
        self.imported_data = None
        self.sanitized_data = []
        np.set_printoptions(threshold=np.nan)

    def analyze(self):
        csv_reader = csv.reader(self.csvfile, delimiter=',')
        next(csv_reader, None)  # skip the headers
        data = [data for data in csv_reader]
        self.imported_data = np.asarray(data)
        self.remove_invalid_entries()
        return self.sanitized_data

    def remove_invalid_entries(self):
        for i in range(self.imported_data[:,].shape[0]):
            rating = self.imported_data[i,1]
            text = self.imported_data[i,3]
            row = []
            if rating == '1' or rating == '-1' or rating == '0':
                row.append(int(rating))
                row.append(text)
                self.sanitized_data.append(row)

        self.sanitized_data = np.array(self.sanitized_data)

        #f2 = open('../tmp/sanitized.txt', 'w+')
        #f2.write(str(self.sanitized_data))


def main():
    parser = argparse.ArgumentParser(prog='CSVAnalyzer')
    parser.add_argument('csvfile', type=argparse.FileType('r'), help='CSV format <Stud|Rating|Link|Comment>')
    args = parser.parse_args()

    csv_analyser = CSVAnalyser(args.csvfile)
    data = csv_analyser.analyze()

    sentiment_analyzer = SentimentV1(data)
    sentiment_analyzer.run_analysis()

if __name__ == "__main__":
    # execute only if run as a script
    main()