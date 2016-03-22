import csv
import argparse
import json
from uclassify import uclassify
from sklearn.cross_validation import KFold
import ast
import pycurl
import numpy as np
import re
from io import BytesIO
import urllib
from warnings import warn

import string
import random
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

class AccuracyAnalyzer:
    def __init__(self, senftimentV1_data = None, senftimentV2_data = None):
        self.sentimentV1_data = senftimentV1_data

    def run_analysis(self):
        print("Run AccuracyAnalyzer")
        print(self.sentimentV1_data.shape)

class SentimentV1Classifier:
    def __init__(self, senftimentV1_data = None, senftimentV2_data = None):
        self.sentimentV1_data = senftimentV1_data
        self.classifier_name = id_generator(10)
        self.classifier = uclassify()
        self.classifier.setWriteApiKey("6jYmrGb25nVC")
        self.classifier.setReadApiKey("lNin5wW4Mod5")


    def cross_validate_classification(self):

        file_name = 0
        #3-fold validation
        kf = KFold(len(self.sentimentV1_data), n_folds=3)
        for train_index, test_index in kf:
            self.classifier.addClass(["pos", "neg", "neutral"], self.classifier_name)
            X_train, X_test = self.sentimentV1_data[train_index], self.sentimentV1_data[test_index]
            neg_train = []
            neutral_train = []
            pos_train = []
            test = []
            for i in range(X_train[:, ].shape[0]):
                if X_train[i, 0] == '-1':
                    neg_train.append(X_train[i, 1])
                elif X_train[i, 0] == '0':
                    neutral_train.append(X_train[i, 1])
                else:
                    pos_train.append(X_train[i, 1])

            for j in range(X_test[:, ].shape[0]):
                test.append(X_test[j, 1])

            # set train tests
            self.classifier.train(neg_train,
                                  "neg", self.classifier_name)
            self.classifier.train(neutral_train,
                                  "neutral", self.classifier_name)
            self.classifier.train(pos_train,
                                  "pos", self.classifier_name)

            # classify
            output = self.classifier.classify(test, self.classifier_name)

            # write to file
            f1 = open('../tmp/classified_set' + str(file_name) + '.txt', 'w+')
            f1.write(str(output))

            file_name += 1
            self.classifier.removeClass(["pos", "neg", "neutral"], self.classifier_name)



    def run_classifier(self):
        self.classifier.create(self.classifier_name)

        self.cross_validate_classification()

        self.classifier.removeClassifier(self.classifier_name)

warn("Not used!")
class DownloadSentiments:
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
            print("Downloading entry " + str(i))
            escaped = re.escape(self.data[i, 1])

            post_data = {'text': escaped}

            buffer = BytesIO()
            c = pycurl.Curl()
            c.setopt(c.URL, 'http://text-processing.com/api/sentiment/')
            c.setopt(c.POSTFIELDS, urllib.urlencode(post_data))
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

                row = [self.data[i, 1], self.data[i, 0], raw_value]
                self.sentimentV1_data.append(row)

            except:
                # save the indices that we need to remove for the second sentiment mining
                self.index_to_remove.append(i)
                pass

        self.sentimentV1_data = np.array(self.sentimentV1_data)
        print(self.sentimentV1_data)

    def run_analysis(self):
        print("Run analysis")
        #self.download_sentiments()
        self.import_mined_json()
        self.construct_arrays()
        return self.sentimentV1_data



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
        for i in range(self.imported_data[:, ].shape[0]):
            rating = self.imported_data[i, 1]
            text = self.imported_data[i, 3]
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

    #print(data)
    v1Classifier = SentimentV1Classifier(data)
    v1Classifier.run_classifier()


if __name__ == "__main__":
    # execute only if run as a script
    main()