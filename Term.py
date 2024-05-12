from math import log10
from typing import List


class Term:
    def __init__(self, value: str):
        self.value: str = value  # word
        self.docs: dict = {}  # key: docID , value: positional indexes for each doc
        self.weights: dict = {}  # key: docID , value: term weight in docs ( tf-idf )
        self.champ_list: List = []

    def calculate_weights(self, n: int):  # without normalizing
        idf = log10(n / len(self.docs))
        # term at a time
        for docId in self.docs:
            tf = 1 + log10(len(self.docs[docId]))
            self.weights[docId] = (tf * idf)

    def doc_idf(self, n: int):
        return Term.idf(n, len(self.docs))
        # return log10(n / len(self.docs))

    def doc_tf(self, docId):
        Term.tf(len(self.docs[docId]))
        # return 1 + log10(len(self.docs[docId]))

    def create_champ_list(self, r: int):
        self.champ_list = [k for k, v in sorted(self.weights.items(), key=lambda item: item[1])[::-1]][0:r]

    @staticmethod
    def idf(n, df):
        return log10(n / df)

    @staticmethod
    def tf(tf):
        return 1 + log10(tf)

    @staticmethod
    def weight(n, df, tf):
        return Term.idf(n, df) * Term.tf(tf)

