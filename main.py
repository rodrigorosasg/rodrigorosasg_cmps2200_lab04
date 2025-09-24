"""
CMPS 2200 Recitation 04
MapReduce Implementation
"""

from collections import defaultdict
from functools import reduce
import math


def run_map_reduce(map_f, reduce_f, docs):
    """
    Run a MapReduce job.

    Params:
      map_f....function mapping a document (string) -> list of (key, val)
      reduce_f.function reducing a key and list of values -> (key, output)
      docs.....list of documents (strings)

    Returns:
      dict mapping key -> output value
    """
    # 1. Map phase
    mapped = []
    for doc in docs:
        mapped.extend(map_f(doc))

    # 2. Shuffle phase (group by key)
    grouped = defaultdict(list)
    for k, v in mapped:
        grouped[k].append(v)

    # 3. Reduce phase
    reduced = {}
    for k, vs in grouped.items():
        reduced[k] = reduce_f(k, vs)[1]

    return reduced


# ----------------------
# Word count
# ----------------------

def word_count_map(doc):
    """
    Map function for word count.
    Splits a document string into words and emits (word, 1) pairs.
    """
    return [(word, 1) for word in doc.split()]


def word_count_reduce(key, values):
    """
    Reduce function for word count.
    Sums up the counts for a given word.
    """
    return (key, sum(values))


# ----------------------
# Sentiment analysis
# ----------------------

def sentiment_map(doc, pos_terms, neg_terms):
    """
    Map function for sentiment analysis.
    Emits ('positive', 1) or ('negative', 1) for each word in the doc.
    """
    results = []
    for word in doc.split():
        if word in pos_terms:
            results.append(('positive', 1))
        elif word in neg_terms:
            results.append(('negative', 1))
    return results


def sentiment_reduce(key, values):
    """
    Reduce function for sentiment analysis.
    Sums positive or negative counts.
    """
    return (key, sum(values))


# ----------------------
# Example usage
# ----------------------

if __name__ == "__main__":
    docs = [
        "the cat sat on the mat",
        "the dog chased the cat",
        "the cat ate the rat"
    ]

    print("Word count results:")
    wc = run_map_reduce(word_count_map, word_count_reduce, docs)
    print(wc)

    pos_terms = {"good", "happy", "love", "like"}
    neg_terms = {"bad", "sad", "hate", "dislike"}
    docs2 = ["i love my cat", "i hate the dog", "the rat is bad"]

    print("\nSentiment analysis results:")
    sent = run_map_reduce(
        lambda d: sentiment_map(d, pos_terms, neg_terms),
        sentiment_reduce,
        docs2
    )
    print(sent)
