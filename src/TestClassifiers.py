from src.classifiers.NNClassifier import NearestNeighborClassifier
from src.classifiers.NaiveBayes import BernoulliNaiveBayesClassifier
from src.classifiers.LogisticClassifier import LogisticClassifier
from src.classifiers.SupportVectorClassifier import LinearSupportVectorClassifier
from src.classifiers.RandomForest import RandomForest
from src.classifiers.AdaBoostClassifier import AdaBoost
from src.deep_models.MLPClassifier import MLPClassifier
from src.deep_models.LSTMClassifier import LSTMClassifier
from src.classifiers.generative_models.HMMClassifier import HMMClassifier

from src.text_processing.TextPreprocessor import TextProcessor
from src.text_processing.Tweet2Vec import Tweet2Vec
from sklearn.decomposition import IncrementalPCA
from keras.preprocessing import sequence
import os
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


class Test:
    def __init__(self, train_x, train_y, test_x, test_y):
        self._original_train_x = train_x
        self._original_test_x = test_x
        self._train_y = train_y
        self._test_y = test_y
        self._text_processor = TextProcessor()
        self._text_processor.train_tokenizer(train_x)
        self._train_x = self._text_processor.convert_text_to_matrix(train_x, mode='tfidf')
        self._test_x = self._text_processor.convert_text_to_matrix(test_x, mode='tfidf')

        # ipca = IncrementalPCA(n_components=100)
        # self._train_x = ipca.fit_transform(np.array(self._train_x))
        # self._test_x = ipca.fit_transform(np.array(self._test_x))

        # self._train_x = train_x
        # self._test_x = test_x
        self._create_saved_model_dir()

        project_relative_path = os.path.dirname(os.path.dirname(__file__))
        output_file_sentiment_label = open(
            os.path.join(project_relative_path, 'saved_model_data/actual_labels.txt'), 'a')
        for label in self._test_y:
            output_file_sentiment_label.write(str(label))
            output_file_sentiment_label.write('\n')

    def test_bernoulli_naive_bayes_classifier(self):
        bernoulli_naive_bayes_classifier = BernoulliNaiveBayesClassifier(self._train_x, self._train_y)
        bernoulli_naive_bayes_classifier.train()
        accuracy_naive_bayes = bernoulli_naive_bayes_classifier.accuracy(self._test_x, self._test_y)
        f1_score_naive_bayes = bernoulli_naive_bayes_classifier.get_average_f1_score(self._test_x, self._test_y)
        return accuracy_naive_bayes, f1_score_naive_bayes

    def test_nn_classifier(self):
        nn_classifier = NearestNeighborClassifier(self._train_x, self._train_y)
        nn_classifier.train()
        nn_accuracy = nn_classifier.test_accuracy(self._test_x, self._test_y)
        return nn_accuracy

    def test_max_ent_classifier(self):
        logistic_classifier = LogisticClassifier()
        logistic_classifier.train(self._train_x, self._train_y)
        logistic_accuracy = logistic_classifier.accuracy_test(self._test_x, self._test_y)
        logistic_f1_score = logistic_classifier.get_average_f1_score(self._test_x, self._test_y)
        return logistic_accuracy, logistic_f1_score

    def test_linear_svc(self):
        linear_svc = LinearSupportVectorClassifier()
        linear_svc.train(self._train_x, self._train_y)
        linear_svc_accuracy = linear_svc.accuracy_test(self._test_x, self._test_y)
        linear_svc_f1_score = linear_svc.get_average_f1_score(self._test_x, self._test_y)
        return linear_svc_accuracy, linear_svc_f1_score

    def test_random_forest(self):
        random_forest = RandomForest()
        random_forest.train(self._train_x, self._train_y)
        random_forest_accuracy = random_forest.accuracy_test(self._test_x, self._test_y)
        random_forest_f1_score = random_forest.get_average_f1_score(self._test_x, self._test_y)
        return random_forest_accuracy, random_forest_f1_score

    def test_ada_boost(self):
        ada_boost = AdaBoost()
        ada_boost.train(self._train_x, self._train_y)
        ada_boost_accuracy = ada_boost.accuracy_test(self._test_x, self._test_y)
        ada_boost_f1_score = ada_boost.get_average_f1_score(self._test_x, self._test_y)
        return ada_boost_accuracy, ada_boost_f1_score

    def test_mlp_classifier(self):
        mlp_classifier = MLPClassifier(input_shape=self._train_x.shape[1])
        mlp_classifier.train(self._train_x, self._train_y)
        mlp_accuracy = mlp_classifier.accuracy_test(self._test_x, self._test_y)
        mlp_f1_score = mlp_classifier.get_average_f1_score(self._test_x, self._test_y)
        return mlp_accuracy, mlp_f1_score

    @staticmethod
    def test_lstm_classifier():
        project_relative_path = os.path.dirname(os.path.dirname(__file__))

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

        print(tweets_train_list)
        print(sentiments_train_list)

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

        text_processor = TextProcessor()
        text_processor.train_tokenizer(tweets_train_list)
        tweets_train_list = text_processor.convert_text_to_sequences(tweets_train_list)
        tweets_test_list = text_processor.convert_text_to_sequences(tweets_test_list)

        src_vocab_size = len(text_processor.get_word_index_keys())
        print(src_vocab_size)

        # for i in range(len(x_train)):
        #     x_train[i][:] = [x - 1 for x in x_train[i]]
        # for i in range(len(x_target)):
        #     x_target[i][:] = [x - 1 for x in x_target[i]]

        tweets_train_seq = sequence.pad_sequences(tweets_train_list, maxlen=50, padding='post', truncating='post')
        tweets_test_seq = sequence.pad_sequences(tweets_test_list, maxlen=50, padding='post', truncating='post')

        print(len(tweets_train_seq))
        print(len(tweets_train_seq))

        lstm_classifier = LSTMClassifier(max_features=src_vocab_size + 1)
        lstm_classifier.train(tweets_train_seq, sentiments_train_list)
        lstm_accuracy = lstm_classifier.accuracy_test(tweets_test_seq, sentiments_test_list)
        lstm_f1_score = lstm_classifier.get_average_f1_score(tweets_test_seq, sentiments_test_list)
        return lstm_accuracy, lstm_f1_score

    def test_hmm_classifier(self):
        hmm_classifier = HMMClassifier()
        hmm_classifier.train(self._train_x)
        hmm_accuracy = hmm_classifier.accuracy(self._test_x, self._test_y)
        return hmm_accuracy

    def test_doc_2_vec_max_ent_classifier(self):
        # tweet2vec = Tweet2Vec()
        # tweet2vec.train(self._original_train_x)
        # tweet_embeddings_train = tweet2vec.embed_tweets(self._original_train_x)
        # tweet_embeddings_test = tweet2vec.embed_tweets(self._original_test_x)

        logistic_classifier = LogisticClassifier()
        logistic_classifier.train(self._original_train_x, self._train_y)
        logistic_accuracy = logistic_classifier.accuracy_test(self._original_test_x, self._test_y)
        return logistic_accuracy

    @staticmethod
    def test_lstm_for_hindi():
        IS_SERVER_BUILD = True
        project_relative_path = os.path.dirname(os.path.dirname(__file__))

        tweets_train_list = open(os.path.join(project_relative_path, 'dataset/hindi/hindi_train.txt'),
                                 'r').read().splitlines()
        sentiments_train_list = open(os.path.join(project_relative_path, 'dataset/hindi/hindi_train_sentiment.txt'),
                                     'r').read().splitlines()
        tweets_test_list = open(os.path.join(project_relative_path, 'dataset/hindi/hindi_train.txt'),
                                'r').read().splitlines()
        sentiments_test_list = open(os.path.join(project_relative_path, 'dataset/hindi/hindi_train_sentiment.txt'),
                                    'r').read().splitlines()

        print(tweets_train_list)
        print(sentiments_train_list)

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

        # Hindi data
        if IS_SERVER_BUILD:
            train_size = int(0.8 * train_tweet_count)
            test_size = int(0.2 * train_tweet_count)
        else:
            train_size = 200
            test_size = 100

        print('Train size :: ' + str(train_size))
        print('Test size :: ' + str(test_size))

        processed_train_tweets = processed_train_tweets[0:train_size]
        processed_test_tweets = processed_test_tweets[train_size:train_size + test_size]

        sentiment_labels_train = sentiment_labels_train[0:train_size]
        sentiment_labels_test = sentiment_labels_test[train_size:train_size + test_size]

        text_processor = TextProcessor()
        text_processor.train_tokenizer(processed_train_tweets)
        tweets_train_list = text_processor.convert_text_to_sequences(processed_train_tweets)
        tweets_test_list = text_processor.convert_text_to_sequences(processed_test_tweets)

        src_vocab_size = len(text_processor.get_word_index_keys())
        print(src_vocab_size)

        # for i in range(len(x_train)):
        #     x_train[i][:] = [x - 1 for x in x_train[i]]
        # for i in range(len(x_target)):
        #     x_target[i][:] = [x - 1 for x in x_target[i]]

        tweets_train_seq = sequence.pad_sequences(tweets_train_list, maxlen=50, padding='post', truncating='post')
        tweets_test_seq = sequence.pad_sequences(tweets_test_list, maxlen=50, padding='post', truncating='post')

        print(len(tweets_train_seq))
        print(len(tweets_train_seq))

        lstm_classifier = LSTMClassifier(max_features=src_vocab_size + 1)
        lstm_classifier.train(tweets_train_seq, sentiment_labels_train)
        lstm_accuracy = lstm_classifier.accuracy_test(tweets_test_seq, sentiment_labels_test)
        lstm_f1_score = lstm_classifier.get_average_f1_score(tweets_test_seq, sentiment_labels_test)
        return lstm_accuracy, lstm_f1_score

    @staticmethod
    def _create_saved_model_dir():
        relative_path_project_dir = os.path.dirname(os.path.dirname(__file__))
        print(relative_path_project_dir)
        if not os.path.exists(os.path.join(relative_path_project_dir, "saved_model_data")):
            os.makedirs(os.path.join(relative_path_project_dir, "saved_model_data"))

    @staticmethod
    def get_saved_model_dir():
        relative_path_project_dir = os.path.dirname(os.path.dirname(__file__))
        return os.path.join(relative_path_project_dir, 'saved_model_data')

    @staticmethod
    def ensemble_classifier_performance():
        project_relative_path = os.path.dirname(os.path.dirname(__file__))
        original_sentiments_list = open(os.path.join(project_relative_path, 'saved_model_data/hindi/actual_labels.txt'), 'r').read().splitlines()
        naive_bayes_sentiments_list = open(os.path.join(project_relative_path, 'saved_model_data/hindi/naive_bayes_labels.txt'), 'r').read().splitlines()
        max_ent_sentiments_list = open(os.path.join(project_relative_path, 'saved_model_data/hindi/max_ent_labels.txt'), 'r').read().splitlines()
        mlp__sentiments_list = open(os.path.join(project_relative_path, 'saved_model_data/hindi/mlp_labels.txt'), 'r').read().splitlines()
        ada_boost_sentiments_list = open(os.path.join(project_relative_path, 'saved_model_data/hindi/ada_boost_labels.txt'), 'r').read().splitlines()
        lstm_sentiments_list = open(os.path.join(project_relative_path, 'saved_model_data/hindi/lstm_labels.txt'), 'r').read().splitlines()
        # random_forest_sentiments_list = open(os.path.join(project_relative_path, 'saved_model_data/semeval/random_forest_labels.txt'), 'r').read().splitlines()

        assert len(original_sentiments_list) == len(naive_bayes_sentiments_list)
        assert len(max_ent_sentiments_list) == len(mlp__sentiments_list)
        assert len(ada_boost_sentiments_list) == len(lstm_sentiments_list)

        predicted_sentiment_list = []
        for i in range(len(original_sentiments_list)):
            sentiments = []
            sentiments.append(naive_bayes_sentiments_list[i])
            # sentiments.append(max_ent_sentiments_list[i])
            sentiments.append(mlp__sentiments_list[i])
            sentiments.append(ada_boost_sentiments_list[i])
            # sentiments.append(lstm_sentiments_list[i])
            # sentiments.append(random_forest_sentiments_list[i])
            print(sentiments)
            most_frequent = max(set(sentiments), key=sentiments.count)
            print('Most freq:: ' + str(most_frequent))
            predicted_sentiment_list.append(most_frequent)

        accuracy = accuracy_score(original_sentiments_list, predicted_sentiment_list)
        f1_val = f1_score(original_sentiments_list, predicted_sentiment_list, average='weighted', labels=[1, 0, -1])
        return accuracy, f1_val
