from __future__ import unicode_literals

import copy
import json
from math import log10
from typing import List
import time
from hazm import *

from Term import Term
from functions import dict_to_terms, vector_length

f = open('IR_data_news_12k.json')

data = json.load(f)

startTime = time.time()

pi = dict_to_terms(json.load(open("postings.txt", 'r', encoding = 'utf-8')))
file1 = open('./filter/stop_words.txt', 'r', encoding = 'utf-8')
stopWords = file1.read().splitlines()
normalizer = Normalizer()
lemmatizer = Lemmatizer()


inputString = ' "تحریم های هسته ای" آمریکا !ایران  '
normString = normalizer.normalize(inputString)
normString = normString.replace("«", "\"").replace("»", "\"")
tokenizedString = word_tokenize(normString)
stemmedString = [lemmatizer.lemmatize(i) for i in tokenizedString]

includedPhrase = []
includedWord = []
excludedWord = []
excludedPhrase = []
firstQuote = False
reachedNot = False
tempArr = []
n = len(data)

for token in stemmedString:
    if firstQuote:
        if token == "\"":
            firstQuote = False
            if reachedNot:
                excludedPhrase.append(tempArr)
            else:

                includedPhrase.append(tempArr)
            tempArr = []
            continue
        tempArr.append(token)
    elif token == "\"":
        firstQuote = True
    elif token == "!":
        reachedNot = True
    elif not reachedNot:
        includedWord.append(token)
    else:
        excludedWord.append(token)


print("includedWord : ", includedWord)
print("includedPhrase : ", includedPhrase)
print("excludedWord : ", excludedWord)
print("excludedPhrase : ", excludedPhrase)


def findWord(word):
    tempArr= []
    if pi.get(word) is None:
        return tempArr
    for docId in pi[word].docs:
        tempArr.append((docId, len(pi[word].docs[docId]), pi[word].docs[docId]))
    return tempArr


# docId empty : does word exist in dictionary ?
# docId : does word exist in given doc ?
def exists(word, docId = -1):
    if pi.get(word) is None:
        return False
    if docId == -1:
        return True
    if pi[word].docs.get(docId) is None:
        return False
    return True


def findPhrase(phrase: List[str]):
    words = []
    result = set()
    startIndex = 0
    for i in phrase:
        if not (i in stopWords):
            words.append(findWord(i))
    # for i in words:
    #     print(i)

    for i, word in enumerate(phrase):
        if word not in stopWords:
            startIndex = i
            break
    if pi.get(phrase[startIndex]) is None:
        return result
    for docId in pi[phrase[startIndex]].docs:
        positions = pi[phrase[startIndex]].docs[docId]
        doesExist = True
        for i in range(startIndex, len(phrase) - 1):
            if phrase[i + 1] in stopWords:
                continue
            if not exists(phrase[i + 1], docId):
                doesExist = False
        if not doesExist:
            continue

        for curPosition in positions:
            for i in range(startIndex, len(phrase) - 1):
                if phrase[i + 1] in stopWords:
                    if i == len(phrase) - 2:
                        result.add(docId)
                        # print("docId: ", docId)
                        # print("position: ", curPosition)
                    continue
                if not(curPosition + i - startIndex + 1 in pi[phrase[i + 1]].docs[docId]):
                    break
                if i == len(phrase) - 2:
                    result.add(docId)
                    # print("docId: ", docId)
                    # print("position: ", curPosition)
    return result


def findPhraseChampion(phrase: List[str]):
    words = []
    result = set()
    startIndex = 0
    allChamp = set()
    for i in phrase:
        if not (i in stopWords):
            words.append(findWord(i))
            if pi.get(i) is not None:
                allChamp = allChamp.union(pi[i].champ_list)

    for i, word in enumerate(phrase):
        if word not in stopWords:
            startIndex = i
            break
    if pi.get(phrase[startIndex]) is None:
        return result
    for docId in allChamp:
        if docId not in pi[phrase[startIndex]].docs:
            continue
        doesExist = True
        positions = pi[phrase[startIndex]].docs[docId]
        for i in range(startIndex, len(phrase) - 1):
            if phrase[i + 1] in stopWords:
                continue
            if not exists(phrase[i + 1], docId):
                doesExist = False
        if not doesExist:
            continue

        for curPosition in positions:
            for i in range(startIndex, len(phrase) - 1):
                if phrase[i + 1] in stopWords:
                    if i == len(phrase) - 2:
                        result.add(docId)
                        # print("docId: ", docId)
                        # print("position: ", curPosition)
                    continue
                if not(curPosition + i - startIndex + 1 in pi[phrase[i + 1]].docs[docId]):
                    break
                if i == len(phrase) - 2:
                    result.add(docId)
                    # print("docId: ", docId)
                    # print("position: ", curPosition)
    return result


def process_query():
    query_words = {}  # query tf
    # calculate query tf
    for i in includedWord:
        if exists(i):
            query_words[i] = 1 if query_words.get(i) is None else query_words[i]+1
    for i in includedPhrase:
        for j in i:
            if exists(j):
                query_words[j] = 1 if query_words.get(j) is None else query_words[j] + 1


    weight = {}  #  {word : query weight }
    ws = []
    for word in query_words:
        weight[word] = Term.weight(n, len(pi[word].docs), query_words[word])
        ws.append(weight[word])
    query_vector_length = vector_length(ws)
    for i in weight:
        weight[i] /= query_vector_length  # normalized weights
    return weight


query_weights = process_query()

print("query weight: ", query_weights)

includedResult = set()
excludedResult = set()
# [[1,2,3], [2 , 5 ,7]]
for i in includedPhrase:
    includedResult = includedResult.union(findPhrase(i))

for i in includedWord:
    if pi.get(i):
        includedResult = includedResult.union(set(pi[i].docs.keys()))
    else:
        includedResult = includedResult.union(set())

# [1, 2, 3, 2, 5 ,7]
for i in excludedWord:
    if pi.get(i) is not None:
        for key in set(pi[i].docs.keys()):
            excludedResult.add(key)

for i in excludedPhrase:
    for key in findPhrase(i):
        excludedResult.add(key)






def merge(included: set, excluded: set):
    to_remove = []
    for i in included:
        if i in excluded:
            to_remove.append(i)
    for i in to_remove:
        included.remove(i)
    return included


def sort_by_tf(result, queryWords):
    arr = dict()
    for docId in result:
        arr[docId] = 0
    #  count terms per doc
    for docId in result:
        for word in queryWords:
            if pi.get(word) is not None and pi[word].docs.get(docId) is not None:
                arr[docId] += len(pi[word].docs[docId])
    return {k: v for k, v in sorted(arr.items(), key=lambda item: item[1])[::-1]}

def sort_by_score(pi, query_weights, docs):
    score = {}  # {docId : score}
    for docId in docs:
        score[docId] = 0
        for word in query_weights:
            term = pi[word]
            if term.weights.get(docId) is None:
                continue
            doc_weight = term.weights[docId]
            query_weight = query_weights[word]
            score[docId] += doc_weight * query_weight  # cosine mul
    return {k: v for k, v in sorted(score.items(), key=lambda item: item[1])[::-1]}


merge(includedResult, excludedResult)



champ_lists = set()
for i in includedPhrase:
    champ_lists = champ_lists.union(findPhraseChampion(i))

for i in includedWord:
    if pi.get(i) is not None:
        champ_lists = champ_lists.union(set(pi[i].champ_list))
champ_lists = champ_lists - excludedResult

for i in includedPhrase:
    includedWord += i

score_sorted_result = sort_by_score(pi, query_weights, includedResult)
score_sorted_result_champ_list = sort_by_score(pi, query_weights, champ_lists)
raw_sorted_result = sort_by_tf(includedResult, includedWord)  #  sort_bt_tf (included docs , included word)
print("tfidf_score_result:", [i for i in score_sorted_result])
print("tfidf_score_result (champ lists):", [i for i in score_sorted_result_champ_list])
print("raw_score_result:", [i for i in raw_sorted_result])
result = score_sorted_result



print("tf idf champList sorted >>>>>>>>>>>>>>>\n>>>>>>>>>>>>>>>>>>>>")
k = 5
c = 0
for i in score_sorted_result_champ_list:
    if c == k:
        break
    print(data[i]["title"])
    print(data[i]["url"])
    print("score: ", score_sorted_result_champ_list[i])
    print("-----------------------------")
    c += 1
print("tf idf safe >>>>>>>>>>>>>>>\n>>>>>>>>>>>>>>>>>>>>")
c = 0
for i in result:
    if c == k:
        break
    print(data[i]["title"])
    print(data[i]["url"])
    print("score: ", result[i])
    print("-----------------------------")
    c += 1
print("tf sorted >>>>>>>>>>>>>>>\n>>>>>>>>>>>>>>>>>>>>")
c = 0
for i in raw_sorted_result:
    if c == k:
        break
    print(data[i]["title"])
    print(data[i]["url"])
    print("score: ", raw_sorted_result[i])
    print("-----------------------------")
    c += 1


# count = dict()
# for i in pi:
#     if count.get(i) is None:
#         count[i] = 0
#     for j in pi[i].docs:
#         count[i] += len(pi[i].docs[j])
# count = [log10(v) for k, v in sorted(count.items(), key=lambda item: item[1])]
# count = count[::-1]



# import matplotlib.pyplot as plt
# import numpy as np
# length = 1000#len(count)
# l = list(range(length+1))
# l.pop(0)
# x = np.array([log10(i) for i in l])
# y = np.array(count[0:length])
# A = np.vstack([x, np.ones(len(x))]).T
# m, c = np.linalg.lstsq(A, y, rcond=None)[0]
#
# length = len(count)
# l = list(range(length+1))
# l.pop(0)
# x = np.array([log10(i) for i in l])
# y = np.array(count[0:length])
#
# plt.plot(x, y, 'o', label='Original data', markersize=1)
# plt.plot(x, m*x + c, 'r', label='Fitted line')
# plt.xlabel("log10 rank")
# plt.ylabel("log10 cf")
# plt.legend()
# plt.show()



print(time.time() - startTime)


