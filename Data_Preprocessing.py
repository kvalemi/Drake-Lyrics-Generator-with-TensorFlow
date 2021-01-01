import re

## Function we will need later to clean up the lyrics ##
def clean_lyrics(sentence):
    sentence = sentence.lower()

    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"i’m", "i am", sentence)

    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"he’s", "he is", sentence)

    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"she’s", "she is", sentence)

    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"it’s", "it is", sentence)

    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"that’s", "that is", sentence)

    sentence = re.sub(r"what's", "what is", sentence)
    sentence = re.sub(r"what’s", "what is", sentence)

    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"where’s", "where is", sentence)

    sentence = re.sub(r"there's", "there is", sentence)
    sentence = re.sub(r"there’s", "there is", sentence)

    sentence = re.sub(r"who's", "who is", sentence)
    sentence = re.sub(r"who’s", "who is", sentence)

    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"how’s", "how is", sentence)

    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"’ll", " will", sentence)

    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"’ve", " have", sentence)

    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"’re", " are", sentence)

    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"’d", " would", sentence)

    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"won’t", "will not", sentence)

    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"can’t", "cannot", sentence)

    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n’t", " not", sentence)

    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"n’", "ng", sentence)

    sentence = re.sub(r"'bout", "about", sentence)
    sentence = re.sub(r"’bout", "about", sentence)

    sentence = re.sub(r"'til", "until", sentence)
    sentence = re.sub(r"’til", "until", sentence)

    sentence = re.sub(r"c'mon", "come on", sentence)
    sentence = re.sub(r"c’mon", "come on", sentence)
    
    sentence = re.sub("\n", "", sentence)

    sentence = re.sub("[-*/()\"’'#/@;:<>{}`+=~|.!?,]", "", sentence)
    
    return sentence



