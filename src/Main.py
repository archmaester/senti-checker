import os
import sys
import numpy as np
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


if __name__ == '__main__':
    IS_SERVER_BUILD = False

    # Create directory and file for output data save
    project_relative_path = os.path.dirname(os.path.dirname(__file__))
    print('Project directory :: ' + str(project_relative_path))
    if not os.path.exists(os.path.join(project_relative_path, "output_data")):
        os.makedirs(os.path.join(project_relative_path, "output_data"))

    if not os.path.exists(os.path.join(project_relative_path, "output_plots")):
        os.makedirs(os.path.join(project_relative_path, "output_plots"))

    if os.path.exists(os.path.join(project_relative_path, 'output_data/accuracy_hindi.txt')):
        os.remove(os.path.join(project_relative_path, 'output_data/accuracy_hindi.txt'))

    if os.path.exists(os.path.join(project_relative_path, 'output_data/f1_score_hindi.txt')):
        os.remove(os.path.join(project_relative_path, 'output_data/f1_score_hindi.txt'))

    accuracy_output_file = open(os.path.join(project_relative_path, 'output_data/accuracy_hindi.txt'), 'a')
    f1_score_output_file = open(os.path.join(project_relative_path, 'output_data/f1_score_hindi.txt'), 'a')

    # Read command line arguments
    arguments = sys.argv[1:]
    if len(arguments) >= 1:
        if arguments[0] == 'server':
            IS_SERVER_BUILD = True

    # train_data = pd.read_csv(os.path.join(project_relative_path, 'dataset/kaggle/train.csv'), dtype=str, encoding='latin-1')
    # test_data = pd.read_csv(os.path.join(project_relative_path, 'dataset/kaggle/test.csv'), dtype=str, encoding='latin-1')

    tweets_train_list = open(os.path.join(project_relative_path, 'dataset/semeval/train/twitter-2016-processed.txt'), 'r').read().splitlines()
    sentiments_train_list = open(os.path.join(project_relative_path, 'dataset/semeval/train/twitter-2016-sentiment.txt'), 'r').read().splitlines()
    tweets_test_list = open(os.path.join(project_relative_path, 'dataset/semeval/test/twitter-2016-processed.txt'), 'r').read().splitlines()
    sentiments_test_list = open(os.path.join(project_relative_path, 'dataset/semeval/test/twitter-2016-sentiment.txt'), 'r').read().splitlines()

    tweets_train_list_2013 = open(os.path.join(project_relative_path, 'dataset/semeval/train/twitter-2013-processed.txt'), 'r').read().splitlines()
    sentiments_train_list_2013 = open(os.path.join(project_relative_path, 'dataset/semeval/train/twitter-2013-sentiment.txt'), 'r').read().splitlines()

    tweets_train_list_2015 = open(os.path.join(project_relative_path, 'dataset/semeval/train/twitter-2015-processed.txt'), 'r').read().splitlines()
    sentiments_train_list_2015 = open(os.path.join(project_relative_path, 'dataset/semeval/train/twitter-2015-sentiment.txt'), 'r').read().splitlines()

    tweets_train_list = tweets_train_list + tweets_train_list_2013 + tweets_train_list_2015
    sentiments_train_list = sentiments_train_list + sentiments_train_list_2013 + sentiments_train_list_2015

    # tweets_train_list = open(os.path.join(project_relative_path, 'dataset/hindi/hindi_train.txt'), 'r').read().splitlines()
    # sentiments_train_list = open(os.path.join(project_relative_path, 'dataset/hindi/hindi_train_sentiment.txt'), 'r').read().splitlines()
    # tweets_test_list = open(os.path.join(project_relative_path, 'dataset/hindi/hindi_train.txt'), 'r').read().splitlines()
    # sentiments_test_list = open(os.path.join(project_relative_path, 'dataset/hindi/hindi_train_sentiment.txt'), 'r').read().splitlines()

    # print(tweets_train_list)
    # print(sentiments_train_list)

    train_tweet_count = len(tweets_train_list)
    train_sentiment_label_count = len(sentiments_train_list)
    assert train_tweet_count == train_sentiment_label_count
    print("Tweet count for training :: " + str(train_tweet_count))

    test_tweet_count = len(tweets_test_list)
    test_sentiment_label_count = len(sentiments_test_list)
    assert test_tweet_count == test_sentiment_label_count
    print("Tweet count for test :: " + str(test_tweet_count))

    processed_train_tweets = []
    for tweet in tweets_train_list:
        processed_train_tweets.append(tweet)

    sentiment_labels_train = []
    for sentiment in sentiments_train_list:
        sentiment_labels_train.append(sentiment)

    processed_test_tweets = []
    for tweet in tweets_test_list:
        processed_test_tweets.append(tweet)

    sentiment_labels_test = []
    for sentiment in sentiments_test_list:
        sentiment_labels_test.append(sentiment)

    assert len(sentiment_labels_train) == len(processed_train_tweets)
    assert len(sentiment_labels_test) == len(processed_test_tweets)

    # train_tweet_count = train_data.shape[0]
    # test_tweet_count = test_data.shape[0]
    # # print(train_data.shape)
    # # print(train_data['Sentiment'])
    # sentiments_train = list(np.array(train_data['Sentiment']))
    # # sentiments_test = list(np.array(test_data['Sentiment']))
    # # print(sentiments.shape)
    # tweets_train = list(np.array(train_data['SentimentText']))
    # tweets_test = list(np.array(test_data['SentimentText']))

    # Test part
    # word_to_is, _ = TextProcessor.build_vocab(tweets_train)
    # print(len(word_to_is.keys()))
    # stopwords = TextProcessor.get_stopwords(tweets_train)
    # print(len(stopwords))
    # tweets = TextProcessor.process_data(tweets_train, stopwords)
    # print(tweets.shape)
    # # print(tweets.shape)
    # print(list(tweets)[1])
    #
    # text_processor = TextProcessor()
    # text_processor.train_tokenizer(tweets)
    #
    # tweets_train = text_processor.convert_text_to_sequences(tweets)
    # print(tweets_train[1])
    # src_vocab_size = len(text_processor.get_word_index_keys())
    # print(src_vocab_size)

    # End of test

    # Semeval data
    if IS_SERVER_BUILD:
        train_size = train_tweet_count
        test_size = test_tweet_count
    else:
        train_size = 200
        test_size = 100

    print('Train size :: ' + str(train_size))
    print('Test size :: ' + str(test_size))

    processed_train_tweets = processed_train_tweets[0:train_size]
    processed_test_tweets = processed_test_tweets[0:test_size]

    sentiment_labels_train = sentiment_labels_train[0:train_size]
    sentiment_labels_test = sentiment_labels_test[0:test_size]

    # # Hindi data
    # if IS_SERVER_BUILD:
    #     train_size = int(0.8 * train_tweet_count)
    #     test_size = int(0.2 * train_tweet_count)
    # else:
    #     train_size = 200
    #     test_size = 100
    #
    # print('Train size :: ' + str(train_size))
    # print('Test size :: ' + str(test_size))
    #
    # processed_train_tweets = processed_train_tweets[0:train_size]
    # processed_test_tweets = processed_test_tweets[train_size:train_size+test_size]
    #
    # sentiment_labels_train = sentiment_labels_train[0:train_size]
    # sentiment_labels_test = sentiment_labels_test[train_size:train_size+test_size]

    # processed_train_tweets = process_tweets(tweets_train[0:train_size])
    # processed_test_tweets = process_tweets(tweets_train[train_size:train_size + test_size])
    #
    # processed_train_tweets = text_processor.process_data(tweets_train[0:train_size], stopwords)
    # processed_test_tweets = text_processor.process_data(tweets_train[train_size:train_size + test_size], stopwords)
    #
    # # processed_train_tweets = text_processor.process_data(processed_train_tweets, stopwords)
    # # processed_test_tweets = text_processor.process_data(processed_test_tweets, stopwords)

    # processed_train_tweets = np.load(os.path.join(project_relative_path, 'output_data/train_embeddings.mat'))
    # processed_test_tweets = np.load(os.path.join(project_relative_path, 'output_data/test_embeddings.mat'))
    #
    # print(processed_train_tweets.shape)
    # print(processed_test_tweets.shape)

    # tweet2vec = Tweet2Vec()
    # tweet2vec.train(processed_train_tweets)
    # tweet_train_embeddings = tweet2vec.embed_tweets(processed_train_tweets)
    # tweet_test_embeddings = tweet2vec.embed_tweets(processed_test_tweets)
    # np.matrix(tweet_train_embeddings).dump(os.path.join(project_relative_path, 'output_data/train_embeddings.mat'))
    # np.matrix(tweet_test_embeddings).dump(os.path.join(project_relative_path, 'output_data/test_embeddings.mat'))

    test_class = Test(processed_train_tweets, sentiment_labels_train, processed_test_tweets, sentiment_labels_test)

    accuracy_naive_bayes, f1_score_naive_bayes = test_class.test_bernoulli_naive_bayes_classifier()
    print('Accuracy with Naive Bayes with processed tweets:: ' + str(accuracy_naive_bayes))
    print('F1-score with Naive Bayes with processed tweets:: ' + str(f1_score_naive_bayes))
    accuracy_output_file.write("Naive Bayes Accuracy :: >> " + str(accuracy_naive_bayes) + "\n")
    f1_score_output_file.write("Naive Bayes F1-score :: >> " + str(f1_score_naive_bayes) + "\n")

    max_ent_accuracy, max_ent_f1_score = test_class.test_max_ent_classifier()
    print('Accuracy with MaxEnt/Logistic classifier with processed tweets:: ' + str(max_ent_accuracy))
    print('F1-score with MaxEnt/Logistic classifier with processed tweets:: ' + str(max_ent_f1_score))
    accuracy_output_file.write("MaxEnt/Logistic Accuracy :: >> " + str(max_ent_accuracy) + "\n")
    f1_score_output_file.write("MaxEnt/Logistic F1-score :: >> " + str(max_ent_f1_score) + "\n")

    linear_svc_accuracy, linear_svc_f1_score = test_class.test_linear_svc()
    print('Accuracy with Linear support vector classifier with processed tweets:: ' + str(linear_svc_accuracy))
    print('F1-score with Linear support vector classifier with processed tweets:: ' + str(linear_svc_f1_score))
    accuracy_output_file.write("Linear support vector Accuracy :: >> " + str(linear_svc_accuracy) + "\n")
    f1_score_output_file.write("Linear support vector F1-score :: >> " + str(linear_svc_f1_score) + "\n")

    mlp_accuracy, mlp_f1_score = test_class.test_mlp_classifier()
    print('Accuracy with Multilayer Perceptron classifier with processed tweets:: ' + str(mlp_accuracy))
    print('F1-score with Multilayer Perceptron classifier with processed tweets:: ' + str(mlp_f1_score))
    accuracy_output_file.write("Multilayer Perceptron Accuracy :: >> " + str(mlp_accuracy) + "\n")
    f1_score_output_file.write("Multilayer Perceptron F1-score :: >> " + str(mlp_f1_score) + "\n")

    random_forest_accuracy, random_forest_f1_score = test_class.test_random_forest()
    print('Accuracy with Random Forest classifier with processed tweets:: ' + str(random_forest_accuracy))
    print('F1-score with Random Forest classifier with processed tweets:: ' + str(random_forest_f1_score))
    accuracy_output_file.write("Random Forest Accuracy :: >> " + str(random_forest_accuracy) + "\n")
    f1_score_output_file.write("Random Forest F1-score :: >> " + str(random_forest_f1_score) + "\n")

    ada_boost_accuracy, ada_boost_f1_score = test_class.test_ada_boost()
    print('Accuracy with Ada-Boost classifier with processed tweets:: ' + str(ada_boost_accuracy))
    print('F1-score with Ada-Boost classifier with processed tweets:: ' + str(ada_boost_f1_score))
    accuracy_output_file.write("Ada-Boost Accuracy :: >> " + str(ada_boost_accuracy) + "\n")
    f1_score_output_file.write("Ada-Boost F1-score :: >> " + str(ada_boost_f1_score) + "\n")

    # # hmm_accuracy = test_class.test_hmm_classifier()
    # # print('Accuracy with Hidden Markov Model classifier with processed tweets:: ' + str(hmm_accuracy))
    # # accuracy_output_file.write("HMM Accuracy :: >> " + str(hmm_accuracy) + "\n")
    #

    # lstm_accuracy, lstm_f1_score = Test.test_lstm_classifier()
    # print('Accuracy with LSTM classifier with processed tweets:: ' + str(lstm_accuracy))
    # print('F1-score with LSTM classifier with processed tweets:: ' + str(lstm_f1_score))
    # accuracy_output_file.write("LSTM Accuracy :: " + str(lstm_accuracy) + "\n")
    # f1_score_output_file.write("LSTM f1-score :: " + str(lstm_f1_score) + "\n")

    # lstm_accuracy, lstm_f1_score = Test.test_lstm_for_hindi()
    # print('Accuracy with LSTM classifier with processed tweets:: ' + str(lstm_accuracy))
    # print('F1-score with LSTM classifier with processed tweets:: ' + str(lstm_f1_score))
    # accuracy_output_file.write("LSTM Accuracy :: " + str(lstm_accuracy) + "\n")
    # f1_score_output_file.write("LSTM f1-score :: " + str(lstm_f1_score) + "\n")
    #
    # accuracy_output_file.write("Naive Bayes Accuracy :: %s  >> \n"
    #                            "NN Accuracy :: %s  >> \n"
    #                            "MaxEnt Accuracy :: %s  >> \n"
    #                            "Linear SVM Accuracy :: %s  >> \n"
    #                            "MLP Accuracy :: %s  >> \n"
    #                            "LSTM Accuracy :: %s  >> \n" %
    #                            (str(accuracy_naive_bayes),
    #                             str(0),
    #                             str(max_ent_accuracy),
    #                             str(linear_svc_accuracy),
    #                             str(mlp_accuracy),
    #                             str(0)))
    #
    # print('\n\n')

    accuracy_output_file.close()
    f1_score_output_file.close()

    # project_relative_path = os.path.dirname(os.path.dirname(__file__))
    # print('Project directory :: ' + str(project_relative_path))
    # # SemEvalProcessor.process_semeval_data(os.path.join(project_relative_path, 'dataset/semeval/test/twitter-2013.txt'))
    # SemEvalProcessor.generate_sentiment_label_distribution(os.path.join(project_relative_path, 'dataset/semeval/test/twitter-2013-sentiment.txt'))

    # accuracy, f1_score = Test.ensemble_classifier_performance()
    # print("Accuracy of ensemble :: " + str(accuracy))
    # print("F1-score of ensemble :: " + str(f1_score))