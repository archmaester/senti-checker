import pandas as pd
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as spio
from src.classifiers.NNClassifier import NearestNeighborClassifier
from src.classifiers.NaiveBayes import BernoulliNaiveBayesClassifier
from src.classifiers.LogisticClassifier import LogisticClassifier

from src.classifiers.LogisticClassifier import LogisticClassifier
from src.text_processing.TextPreprocessor import TextProcessor
from src.text_processing.PrepareData import process_tweets
from src.text_processing.Tweet2Vec import Tweet2Vec

from src.TestClassifiers import Test


def process_semeval_data(filename):
    project_relative_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    print(project_relative_path)
    # if os.path.exists(os.path.join(project_relative_path, 'output_data/data_distribution_semeval_test_2016.txt')):
    #     os.remove(os.path.join(project_relative_path, 'output_data/data_distribution_semeval_test_2016.txt'))
    # if os.path.exists(os.path.join(project_relative_path, 'dataset/semeval/test/twitter-2016-processed.txt')):
    #     os.remove(os.path.join(project_relative_path, 'dataset/semeval/test/twitter-2016-processed.txt'))
    data_dist_output_file = open(os.path.join(project_relative_path, 'output_data/data_distribution_semeval_test_2013.txt'), 'a')
    input_file = open(filename, mode='r')
    data = input_file.readlines()
    tweets = []
    sentiments = []
    for line in data:
        data_line_list = line.split('\t')
        if len(data_line_list) >= 3:
            tweet = data_line_list[2]
            sentiment = data_line_list[1]
            tweets.append(tweet)
            sentiments.append(sentiment)

    unique_sentiment_labels = set([sentiment for sentiment in sentiments])
    print(unique_sentiment_labels)
    sentiment_labels = []
    # Process sentiments to a file
    for sentiment in sentiments:
        if sentiment == 'negative':
            sentiment_labels.append(-1)
        elif sentiment == 'positive':
            sentiment_labels.append(1)
        else:
            sentiment_labels.append(0)
    output_file_sentiment_label = open(os.path.join(project_relative_path, 'dataset/semeval/test/twitter-2013-sentiment.txt'), 'a')
    for label in sentiment_labels:
        output_file_sentiment_label.write(str(label))
        output_file_sentiment_label.write('\n')

    # Test part
    word_to_ids, _, word_distribution, total_word_count = TextProcessor.build_vocab(tweets)
    raw_vocab_size = len(word_to_ids)
    print("Total word count :: " + str(total_word_count))
    data_dist_output_file.write("Total word count :: " + str(total_word_count) + '\n\n')
    print("Raw vocabulary size :: " + str(raw_vocab_size))
    data_dist_output_file.write("Raw vocabulary size :: " + str(raw_vocab_size) + '\n\n')
    print("Raw word distribution :: " + str(word_distribution))
    data_dist_output_file.write("Raw word distribution :: " + str(word_distribution) + '\n\n')

    # Find raw word frequency distribution
    uniq_freq_list = sorted(set([val for val in word_distribution.values()]))
    word_freq_distribution = {}
    for freq in uniq_freq_list:
        word_count_given_freq = sum(word_freq == freq for word_freq in word_distribution.values())
        word_freq_distribution[freq] = word_count_given_freq

    print("Raw word-frequency distribution :: " + str(word_freq_distribution))
    data_dist_output_file.write("Raw word-frequency distribution :: " + str(word_freq_distribution) + '\n\n')

    # Plot data distribution
    ax = plt.subplot(111)
    ax.plot(np.log(list(word_freq_distribution.keys())), word_freq_distribution.values(), '-o', c='blue')
    plt.xlabel("Frequency")
    plt.ylabel("Word Count")
    plt.savefig(os.path.join(project_relative_path, 'output_plots/freq_distribution_raw_test_2013.pdf'))
    plt.clf()

    stopwords = TextProcessor.get_stopwords(tweets)
    print("Stopwords count before processing :: " + str(len(stopwords)))
    data_dist_output_file.write("Stopwords count before processing :: " + str(len(stopwords)) + '\n\n')

    processed_tweets = process_tweets(tweets, hashtag=True)
    stopwords = TextProcessor.get_stopwords(processed_tweets)
    print("Stopwords count after processing :: " + str(len(stopwords)))
    data_dist_output_file.write("Stopwords count after processing :: " + str(len(stopwords)) + '\n\n')
    processed_tweets = TextProcessor.process_data(processed_tweets, stopwords)

    output_file_processed_tweets = open(os.path.join(project_relative_path, 'dataset/semeval/test/twitter-2013-processed.txt'), 'a')
    for processed_tweet in processed_tweets:
        output_file_processed_tweets.write(processed_tweet)
        output_file_processed_tweets.write('\n')

    # Find processed tweets frequency distribution
    word_to_ids, _, word_distribution, total_word_count = TextProcessor.build_vocab(processed_tweets)
    processed_vocab_size = len(word_to_ids)
    print("Total processed word count :: " + str(total_word_count))
    data_dist_output_file.write("Total processed word count :: " + str(total_word_count) + '\n\n')
    print("Processed vocabulary size :: " + str(processed_vocab_size))
    data_dist_output_file.write("Processed vocabulary size :: " + str(processed_vocab_size))
    print("Processed word distribution :: " + str(word_distribution))
    data_dist_output_file.write("Processed word distribution :: " + str(word_distribution) + '\n\n')

    uniq_freq_list = sorted(set([val for val in word_distribution.values()]))
    processed_word_freq_distribution = {}
    for freq in uniq_freq_list:
        word_count_given_freq = sum(word_freq == freq for word_freq in word_distribution.values())
        processed_word_freq_distribution[freq] = word_count_given_freq

    print("Processed word-frequency distribution :: " + str(processed_word_freq_distribution))
    data_dist_output_file.write("Processed word-frequency distribution :: " + str(processed_word_freq_distribution) + '\n\n')

    # Plot data distribution
    ax = plt.subplot(111)
    ax.plot(np.log(list(processed_word_freq_distribution.keys())), processed_word_freq_distribution.values(), '-o', c='blue')
    plt.xlabel("Frequency")
    plt.ylabel("Word Count")
    plt.savefig(os.path.join(project_relative_path, 'output_plots/freq_distribution_processed_test_2013.pdf'))
    plt.clf()

    text_processor = TextProcessor()
    text_processor.train_tokenizer(processed_tweets)

    tweets_train = text_processor.convert_text_to_sequences(processed_tweets)
    src_vocab_size = len(text_processor.get_word_index_keys())
    print("Vocabulary size after removal of low freq(1,2) words :: " + str(src_vocab_size))
    data_dist_output_file.write("Vocabulary size after removal of low freq(1,2) words :: " + str(src_vocab_size) + '\n\n')
    # End of test


def generate_sentiment_label_distribution(filename):
    project_relative_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    output_file_sentiment_dist = open(os.path.join(project_relative_path, 'output_data/sentiment_distribution.txt'), 'a')
    labels = np.loadtxt(filename)
    pos_label_count = 0
    neg_label_count = 0
    neutral_label_count = 0

    for label in labels:
        if label == 1:
            pos_label_count += 1
        elif label == -1:
            neg_label_count += 1
        else:
            neutral_label_count += 1

    print("Positive sentiment count :: " + str(pos_label_count))
    print("Negative sentiment count :: " + str(neg_label_count))
    print("Neutral sentiment count :: " + str(neutral_label_count))

    output_file_sentiment_dist.write("Dataset :: Test-twitter-2013" + '\n\n')
    output_file_sentiment_dist.write("Positive sentiment count :: " + str(pos_label_count) + '\n\n')
    output_file_sentiment_dist.write("Negative sentiment count :: " + str(neg_label_count) + '\n\n')
    output_file_sentiment_dist.write("Neutral sentiment count :: " + str(neutral_label_count) + '\n\n')

