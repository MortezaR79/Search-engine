from __future__ import unicode_literals
import json
import time
from copy import copy
from typing import Dict

from hazm import *
from math import log10
import matplotlib.pyplot as plt
import numpy as np

from Term import Term
from functions import terms_to_dict, vector_length

#  replace or delete abnormal tokens
def replace_chars(string: str):
    f1 = open('./filter/replaced chars.txt', 'r', encoding='utf-8')
    replaced = set(f1.read().splitlines())
    f2 = open('./filter/deleted chars.txt', 'r', encoding='utf-8')
    deleted = set(f2.read().splitlines())
    new_str = ""
    for i in string:
        if i in deleted:
            continue
        new_str += " " if i in replaced else i
    return new_str

f = open('IR_data_news_12k.json')
file1 = open('./filter/stop_words.txt', 'r', encoding = 'utf-8')
stopWords = file1.read().splitlines()


data = json.load(f)
normalizer = Normalizer()
lemmatizer = Lemmatizer()

pi = {}

x = []
y = []
x_total = []
y_total = []


tokenCount = 0
step = 500
termCount = dict()
n = len(data) #10000
print(n)
for docId in data:
    if int(docId) >= n:
        break
    normData = normalizer.normalize(data[docId]['content'])
    normData = replace_chars(normData)
    tokenizedData = word_tokenize(normData)
    stemmedData = [(lemmatizer.lemmatize(i), counter) for counter, i in enumerate(tokenizedData)]
    filteredData = [i for i in stemmedData if i[0] not in stopWords]
    # creating inverted index
    for token, position in filteredData:
        if pi.get(token) is None:
            pi[token] = Term(token)
        if pi[token].docs.get(docId) is None:
            pi[token].docs[docId] = []
        pi[token].docs[docId].append(position)
        tokenCount += 1
        # plot
        # if (int(docId)+1)%500 == 0 and int(docId) < 2002:
        #     x.append(log10(tokenCount))
        #     y.append(log10(len(pi)))
        # x_total.append(log10(tokenCount))
        # y_total.append(log10(len(pi)))

print("token:", tokenCount)
print("term:", len(pi))


def calculate_doc_weights(n):
    docs_weight_vectors = {}  # {"docId":[Terms weight in doc]}
    docs_len = {}  # {"docId":vector length}
    # initializing
    for docId in data:
        docs_weight_vectors[docId] = []
    for word in pi:
        term = pi[word]
        term.calculate_weights(n)
        for docId in term.weights:
            docs_weight_vectors[docId].append(term.weights[docId])
    for docId in docs_weight_vectors:
        docs_len[docId] = vector_length(docs_weight_vectors[docId])

    # normalizing
    for word in pi:
        term = pi[word]
        for docId in term.weights:
            term.weights[docId] /= docs_len[docId]


def create_champion_lists(r: int):
    for word in pi:
        term = pi[word]
        term.create_champ_list(r)


calculate_doc_weights(n)
create_champion_lists(20)

dic = terms_to_dict(pi)  # convert object to python dictionary for saving string in file
# print(pi["آسیا"].weights)
# print(pi["آسیا"].docs)
# print(len(pi["آسیا"].docs))
json.dump(dic, open("postings.txt",'w'))

f.close()


#  plot

# x = np.array(x)
# y = np.array(y)
# x_total = np.array(x_total)
# y_total = np.array(y_total)
#
# A = np.vstack([x, np.ones(len(x))]).T
# m, c = np.linalg.lstsq(A, y, rcond=None)[0]
# print(m, c)
# plt.plot(x_total, y_total, 'o', label='Original data', markersize=1)
# plt.plot(x_total, m*x_total + c, 'r', label='Fitted line')
# plt.xlabel("log10 token")
# plt.ylabel("log10 term")
# plt.legend()
# plt.show()

