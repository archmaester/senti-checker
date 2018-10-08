import re
from nltk.corpus import words, wordnet
from autocorrect import spell


def hashtag_process(hashtag):
    htag = re.compile(r'#[A-Z]{2,}(?![a-z])|[A-Z][a-z]+')
    htag1 = htag.findall(hashtag)
    res = ''
    for wrd in htag1:
        res += wrd+' '
    return res


def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;*')
    #print (word)
    # Remove - & '
    #word = re.sub(r'(-|\')', ' ', word)
    #print (word)

    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    if re.search(r'(.)\1+', word):
        word = re.sub(r'(.)\1+', r'\1\1', word)
        #word = spell(word)
        
        if ((len(wordnet.synsets(word)) == 0) and (word not in words.words()) and word != spell(word)) :
            #word = re.sub(r'(.)\1+', r'\1', word)
            word = spell(word)
    return word


def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)

    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)

    # Love -- <3, :*
    tweet = re.sub(r'(&lt;3|<3|:\*)', ' EMO_POS ', tweet)

    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)

    # Sad -- :-(, : (, :(, ):, )-:, :-|
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:|:-\|)', ' EMO_NEG ', tweet)

    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)

    #html chars
    tweet = re.sub(r'(&quot;)', ' \" ', tweet)
    tweet = re.sub(r'(&lt;)', ' < ', tweet)
    tweet = re.sub(r'(&gt;)', ' > ', tweet)  
    tweet = re.sub(r'(&amp;)', ' & ', tweet) 

    return tweet


def preprocess_tweet(tweet, hashtag):
    processed_tweet = []

    # Replaces #hashtag with hashtag
    if hashtag:
        val = hashtag_process(tweet)
        # print (val)
        tweet = re.sub(r'#(\S+)', val, tweet)
    else:
        tweet = re.sub(r'#(\S+)', '', tweet)

    # Convert to lower case
    tweet = tweet.lower()

    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', '', tweet)

    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', '', tweet)

    # Replace unicode character \u2019 to apostrophe
    tweet = re.sub(u'((\u2018)|(\u2019))', "'", tweet)

    

    '''
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)

    #Replace words like "weren't" with "were not"
    tweet = re.sub(r"[a-zA-Z]*n't\\s*", 'not ', tweet)

    #Replace words like "i'm | im" with "i am"
    tweet = re.sub(r'(i\'m|im|Im|IM)', 'i am', tweet)

    #Replace words like "i've" with "i have"
    tweet = re.sub(r"([a-zA-Z])+\'ve\\s*", r'\1'+' have ', tweet)

    '''
    tweet = re.sub(r"won't", "will not", tweet)
    tweet = re.sub(r"can\'t", "can not", tweet)
    tweet = re.sub(r"isnt", "is not", tweet)
    tweet = re.sub(r"its", "it is", tweet)
    tweet = re.sub(r"thats", "that is", tweet)

    tweet = re.sub(r"n\'t", " not", tweet)
    tweet = re.sub(r"\'re", " are", tweet)
    tweet = re.sub(r"\'s", " is", tweet)
    tweet = re.sub(r"\'d", " would", tweet)
    tweet = re.sub(r"\'ll", " will", tweet)
    tweet = re.sub(r"\'cause", " because", tweet)
    #tweet = re.sub(r"\'t", " not", tweet)
    tweet = re.sub(r"\'ve", " have", tweet)
    #tweet = re.sub(r"\'m", " am", tweet)
    tweet = re.sub(r'(i\'m|\s+im\s+)', ' i am ', tweet)


    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)

    # Strip *, space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    #tweet = tweet.strip('*')

    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)

    tweet = re.sub(r'(-|\')', ' ', tweet)
    #print (tweet)

    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    words = tweet.split()

    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            processed_tweet.append(word)

    return ' '.join(processed_tweet)


def process_tweets(tweets, hashtag=False):
    mod_tweets = []
    for tweet in tweets:
        mod_tweets.append(preprocess_tweet(tweet, hashtag))
    return mod_tweets