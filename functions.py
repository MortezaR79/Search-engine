import math
from copy import copy
from typing import Dict, List

from Term import Term


def dict_to_terms(dic):
    pi = {}
    for word in dic:
        pi[word] = Term(word)
        pi[word].champ_list = copy(dic[word]["championList"])
        for docId in dic[word]:
            if (docId == "championList"):
                continue
            pi[word].docs[docId] = copy(dic[word][docId]["positons"])
            pi[word].weights[docId] = copy(dic[word][docId]["weights"])

    return pi


def terms_to_dict(pi: Dict[str, Term]):
    new_pi = {}
    for word in pi:
        term = pi[word]

        new_pi[word] = {}
        new_pi[word]["championList"] = copy(term.champ_list)
        for docId in term.docs:
            new_pi[word][docId] = {}
            new_pi[word][docId]["weights"] = term.weights[docId]
            new_pi[word][docId]["positons"] = copy(term.docs[docId])
    return new_pi


def vector_length(values: List[float]):
    return math.sqrt(sum(i**2 for i in values))

