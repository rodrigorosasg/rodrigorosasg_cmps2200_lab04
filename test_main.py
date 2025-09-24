from main import run_map_reduce, word_count_map, word_count_reduce, sentiment_map, sentiment_reduce

def test_word_count_map():
    doc = "the cat sat"
    result = word_count_map(doc)
    assert ("the", 1) in result
    assert ("cat", 1) in result
    assert ("sat", 1) in result
    assert len(result) == 3


def test_word_count_reduce():
    key, total = word_count_reduce("cat", [1, 1, 1])
    assert key == "cat"
    assert total == 3


def test_run_map_reduce_wordcount():
    docs = ["the cat sat", "the dog chased the cat"]
    result = run_map_reduce(word_count_map, word_count_reduce, docs)
    assert result["the"] == 3
    assert result["cat"] == 2
    assert result["dog"] == 1
    assert result["chased"] == 1


def test_sentiment_map_positive_negative():
    pos_terms = {"good", "love"}
    neg_terms = {"bad", "hate"}
    doc = "i love my dog but hate the rat"
    result = sentiment_map(doc, pos_terms, neg_terms)
    assert ("positive", 1) in result
    assert ("negative", 1) in result


def test_sentiment_reduce():
    key, total = sentiment_reduce("positive", [1, 1, 1])
    assert key == "positive"
    assert total == 3


def test_run_map_reduce_sentiment():
    pos_terms = {"good", "happy", "love"}
    neg_terms = {"bad", "sad", "hate"}
    docs = ["i love my cat", "i hate the dog", "the rat is bad"]
    result = run_map_reduce(
        lambda d: sentiment_map(d, pos_terms, neg_terms),
        sentiment_reduce,
        docs
    )
    assert result["positive"] == 1
    assert result["negative"] == 2
