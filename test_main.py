from main import *   


### PART One

def test_word_count_map():
    assert word_count_map('i am sam i am') == \
           [('i', 1), ('am', 1), ('sam', 1), ('i', 1), ('am', 1)]

    
def test_word_count_reduce():
    assert word_count_reduce(['i', [1,1,1]]) == ('i', 3)

def test_word_count():
    assert run_map_reduce(word_count_map, word_count_reduce, ['i am sam i am', 'sam is ham']) == \
           [('am', 2), ('ham', 1), ('i', 2), ('is', 1), ('sam', 2)]


    



    
    
    
### PART TWO ###



def test_sentiment_map():
    assert sentiment_map('it was a terrible waste of time') == [('negative', 1), ('negative', 1)]

    
def test_sentiment():
    docs = [
        'it was not great but not terrible',
        'thou art a boil a plague-sore or embossed carbuncle in my corrupted blood',
        'it was a sockdolager of a good time'
    ]
    result = run_map_reduce(sentiment_map, word_count_reduce, docs)
    assert result == [('negative', 3), ('positive', 3)]

