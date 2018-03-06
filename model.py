import pandas as pd
import numpy as np
from discretize import bins
from multiprocessing import Pool
from tabulate import tabulate
import operator


def disc(df):
    columns = list(df.columns.values)
    return bins(np.array(df[columns[0]]), np.array(df[columns[1]]))


def find_bin(value, bins):
    for index, bin_ in enumerate(bins):
        if value > bin_[0] and value <= bin_[1]:
            return index

        
def efbin(arr, bins=20):
    arr = sorted(arr)
    n = float(len(arr))
    n /= bins
    n = int(n)
    counter = 0
    bounds = [(float('-inf'), arr[0])]
    for x in arr:
        if counter >= n:
            bounds.append((bounds[-1][1], x))
            counter = 0
        counter += 1
    bounds.append((bounds[-1][1], float('+inf')))
    return bounds


class NBModel:
    def __init__(self, json_file=None, engine=None, query=None):
        df = pd.read_sql(query, engine)
        columns = list(df.columns.values)
        attributes = columns[:-1]
        class_col = columns[-1]
        classes = np.unique(np.array(df[class_col]))
        p = Pool(8)
        binned = p.map(disc, [df[[attribute, class_col]].sort_values([attribute])
                              for attribute in attributes])
        self.bins = dict(zip(attributes, binned))
        self.counters = {}
        for attribute in attributes:
            n_bins = len(self.bins[attribute])
            self.counters[attribute] = dict(zip(classes, [np.zeros(n_bins) for _ in classes]))
        for row in df.iterrows():
            for attribute in attributes:
                attr = self.counters[attribute]
                cls = attr[row[1][class_col]]
                bin_index = find_bin(row[1][attribute], self.bins[attribute])
                cls[bin_index] += 1.0
        self.attributes = attributes
        self.class_col = class_col
        self.classes = classes
        self.class_p = dict(zip(self.classes, [0.0 for _ in self.classes]))
        for attribute in self.attributes:
            for c in self.classes:
                self.class_p[c] += self.counters[attribute][c].sum()
        summed = sum([self.class_p[c] for c in self.classes])
        for c in self.classes:
            self.class_p[c] /= summed
        self.evaluated = []
        for row in df.iterrows():
            score = self.evaluate(row[1])
            score['c'] = row[1][-1]
            self.evaluated.append(score)
        self.callibration_bins = efbin([x[0] for x in self.evaluated])
        self.callibrated_probs = [0.0 for _ in self.callibration_bins]
        
        
    def __str__(self):
        string = ""
        for attribute in self.attributes:
            string += str(attribute) + ":\n\n"
            table = {}
            table['bin'] = self.bins[attribute]
            for c in self.classes:
                table[str(c)] = list(self.counters[attribute][c])
            string += tabulate(table, headers="keys")
            string += '\n\n\n'
        return string            
    
        
    def store(self, json_path):
        with open(json_path, 'w') as fd:
            fd.write(json.dumps(self.counters))


    def evaluate(self, x):
        result = {}
        for c in self.classes:
            p_c = self.class_p[c]
            p_e_c = 1.0
            for attribute in self.attributes:
                i = find_bin(x[attribute], self.bins[attribute])
                p_e_c *= self.counters[attribute][c][i] / self.counters[attribute][c].sum()
            result[c] = p_c * p_e_c
        summed = sum([result[c] for c in self.classes])
        for c in self.classes:
            result[c] = result[c] / summed
        return result

        
    def classify(self, x):
        y = self.evaluate(x)
        return max(y.iteritems(), key=operator.itemgetter(1))[0]
