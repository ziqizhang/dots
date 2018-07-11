import nltk
from nltk import PorterStemmer, WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
import enchant
import splitter
import csv
import re

import pandas as pd
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

text_processor = TextPreProcessor(
    # terms that will be normalized
# normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
#                'time', 'url', 'date', 'number'],
#     # terms that will be annotated
#     annotate={"hashtag", "allcaps", "elongated", "repeated",
#               'emphasis', 'censored'},

    normalize=[],
    # terms that will be annotated
    annotate={'elongated',
              'emphasis'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=False).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

d = enchant.Dict('en_UK')
dus = enchant.Dict('en_US')
space_pattern = '\s+'
giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
mention_regex = '@[\w\-]+'
emoji_regex = '&#[0-9]{4,6};'

sentiment_analyzer = VS()
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words("english")
other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

# stem_or_lemma: 0 - apply porter's stemming; 1: apply lemmatization; 2: neither
# -set to 0 to reproduce Davidson. However, note that because a different stemmer is used, results could be
# sightly different
# -set to 2 will do 'basic_tokenize' as in Davidson
def tokenize(tweet, stem_or_lemma=0):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and normalizes tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    if stem_or_lemma==0:
        tokens = [stemmer.stem(t) for t in tweet.split()]
    elif stem_or_lemma==1:
        tokens=[lemmatizer.lemmatize(t) for t in tweet.split()]
    else:
        tokens = [t for t in tweet.split()] #this is basic_tokenize in TD's original code
    return tokens


# tweets should have been preprocessed to the clean/right format before passing to this method
def get_pos_tags(tweets):
    """Takes a list of strings (tweets) and
    returns a list of strings of (POS tags).
    """
    tweet_tags = []
    for t in tweets:
        tokens = tokenize(t, 2)
        tags = nltk.pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        #for i in range(0, len(tokens)):
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
    return tweet_tags


def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    parsed_text = re.sub('RT','', parsed_text) #Some RTs have !!!!! in front of them
    parsed_text = re.sub(emoji_regex,'',parsed_text) #remove emojis from the text
    parsed_text = re.sub('…','',parsed_text) #Remove the special ending character is truncated
    #parsed_text = re.sub('#[\w\-]+', '',parsed_text)
    #parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text


def preprocess_clean(text_string, remove_hashtags=True, remove_special_chars=True):
    # Clean a string down to just text
    text_string=preprocess(text_string)

    parsed_text = preprocess(text_string)
    parsed_text = parsed_text.lower()
    parsed_text = re.sub('\'', '', parsed_text)
    parsed_text = re.sub('|', '', parsed_text)
    parsed_text = re.sub(':', '', parsed_text)
    parsed_text = re.sub(',', '', parsed_text)
    parsed_text = re.sub(';', '.', parsed_text)
    parsed_text = re.sub('&amp', '', parsed_text)

    if remove_hashtags:
        parsed_text = re.sub('#[\w\-]+', '',parsed_text)
    if remove_special_chars:
        #parsed_text = re.sub('(\!|\?)+','.',parsed_text) #find one or more of special char in a row, replace with one '.'
        parsed_text = re.sub('(\!|\?)+','',parsed_text)
    return parsed_text

def strip_hashtags(text):
    text = preprocess_clean(text,False,True)
    hashtags = re.findall('#[\w\-]+', text)
    for tag in hashtags:
        cleantag = tag[1:]
        if d.check(cleantag) or dus.check(cleantag):
            text = re.sub(tag,cleantag,text)
            pass
        else:
            hashtagSplit = ""
            for word in splitter.split(cleantag.lower(),'en_US'):
                hashtagSplit = hashtagSplit + word + " "
            text = re.sub(tag,hashtagSplit,text)
    #print(text)
    return text


if __name__ == "__main__":
    sentences = [
        "CANT WAIT for the new season of #TwinPeaks ＼(^o^)／!!! #davidlynch #tvseries :)))",
        "I saw the new #johndoe movie and it suuuuucks!!! WAISTED $10... #badmovies :/",
        "@SentimentSymp:  hoooolly can't wait for the Nov 9 #Sentiment talks! *VERY* good, f**k YAAAAAAY !!! :-D http://sentimentsymposium.com/.",
        "Add anotherJEW fined a bi$$ion for stealing like a lil maggot"
    ]

    # for s in sentences:
    #     res=text_processor.pre_process_doc(s)
    #     res=list(filter(lambda a: a != '<elongated>', res))
    #     print(res)
    # exit(1)

    col_text=7
    input_data_file="/home/zz/Work/chase/data/ml/ml/rm/labeled_data_all_corrected.csv"
    #input_data_file="/home/zz/Work/chase/data/ml/ml/dt/labeled_data_all_2.csv"
    #input_data_file="/home/zz/Work/chase/data/ml/ml/w/labeled_data_all.csv"
    #input_data_file="/home/zz/Work/chase/data/ml/ml/w+ws/labeled_data_all.csv"
    #input_data_file="/home/zz/Work/chase/data/ml/ml/ws-exp/labeled_data_all.csv"
    #input_data_file="/home/zz/Work/chase/data/ml/ml/ws-amt/labeled_data_all.csv"
    #input_data_file="/home/zz/Work/chase/data/ml/ml/ws-gb/labeled_data_all.csv"

    raw_data = pd.read_csv(input_data_file, sep=',', encoding="utf-8")
    header_row=list(raw_data.columns.values)
    with open(input_data_file+"c.csv", 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(header_row)

        for row in raw_data.iterrows():
            tweet = list(row[1])
            tweet_text=text_processor.pre_process_doc(tweet[col_text])
            tweet_text = list(filter(lambda a: a != '<elongated>', tweet_text))
            tweet_text = list(filter(lambda a: a != '<emphasis>', tweet_text))
            tweet_text = list(filter(lambda a: a != 'RT', tweet_text))
            tweet_text = list(filter(lambda a: a != '"', tweet_text))
            tweet_text=" ".join(tweet_text)

            #reset content
            tweet[col_text]=tweet_text

            csvwriter.writerow(tweet)
