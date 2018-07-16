from cleaning.read_n_write import ReadingNWrite
from processing.processing import Processing
from processing.processing_from_raw_tweet import RawProcessing
#from final_try import CaptionGeneration
from processing_images.caption_generation_csv import CaptionGenerationCSV
from ml.classifier import Classification
from ml.gensim_model import GensimModel
# import configargparse # for future use
import sys, os
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer


class Main:
    def __init__(self):

        self.model_paths = ['./resources/models/lsa1', './resources/models/lsa2', './resources/models/lsa3',
                            './resources/models/lsa4', './resources/models/lsa5', './resources/models/lsa6',
                            './resources/models/vectorizer',
                            './resources/models/finalized_model_text.svm',
                            './resources/models/finalized_model_image.svm',
                            './resources/models/finalized_model_all.svm']

        # loading all the classes
        self.read_write = ReadingNWrite()
        self.raw_tweet_processing = RawProcessing()
        self.processing_tweet = Processing(self.read_write)
        self.caption_generation = CaptionGenerationCSV()

        self.gensim_model = GensimModel()
        self.classification = Classification()

        # all models loaded here
        self.model_text, self.model_image, self.model_all = None, None, None

        self.lsa1, self.lsa2, self.lsa3, = self.do_LSA(3), self.do_LSA(75), self.do_LSA(30)
        self.lsa4, self.lsa5 = self.do_LSA(30), self.do_LSA(25)

        self.lsa6 = self.do_LSA(25)
        self.vectorizer = TfidfVectorizer()

        # all parameters
        self.status_tt = 'idle'
        self.feature_reduction = True  # want to reduce the feature or not

    def do_LSA(self, no_comp):
        svd = TruncatedSVD(no_comp)
        lsa = make_pipeline(svd, Normalizer(copy=False))
        return lsa

    def create_LSA_hash_features(self, hash_tag_dict):
        keys = []
        hash_text = []
        for i_d in hash_tag_dict:
            keys.append(i_d)
            hash_text.append(hash_tag_dict[i_d])
        tfidf_vec, lsa_vec = None, None
        if self.status_tt == 'training':
            tfidf_vec = self.vectorizer.fit_transform(hash_text)
            lsa_vec = self.lsa6.fit_transform(tfidf_vec)
        elif self.status_tt == 'testing':
            tfidf_vec = self.vectorizer.transform(hash_text)
            lsa_vec = self.lsa6.transform(tfidf_vec)
        else:
            print('The status is different ' + self.status_tt)
            sys.exit(1)
        hash_tag_vec_dict = {}
        for i_d, f in zip(keys, lsa_vec):
            hash_tag_vec_dict[i_d] = f

        return hash_tag_vec_dict

    def create_LSA_features(self, features_dict, lsa_model):
        features = []
        keys = []
        for i_d in features_dict:
            keys.append(i_d)
            features.append(features_dict[i_d])
        lsa_vec = None
        if self.status_tt == 'training':
            lsa_vec = lsa_model.fit_transform(features)
        elif self.status_tt == 'testing':
            lsa_vec = lsa_model.transform(features)
        else:
            print('The status is different ' + self.status_tt)
            sys.exit(1)
        new_features = {}
        for i_d, f in zip(keys, lsa_vec):
            new_features[i_d] = f
        return new_features

    def get_text_features(self, xml_file_dict):
        text_features_dict = {}
        hash_tag_dict = {}
        normal_10_features_dict = {}
        word_vec_dict = {}
        topic_vec_dict = {}
        print('Generating text features')
        for i_d in xml_file_dict:
            single_user_tweets = xml_file_dict[i_d]
            # generating text features
            tweets = self.read_write.read_tweets(single_user_tweets)
            new_tweets, text_features, hash_tags = self.raw_tweet_processing.raw_processing(tweets)
            hash_tag_dict[i_d] = hash_tags

            temp, final_tweets, final_tweet_lemmas = self.processing_tweet.process(new_tweets)
            text_features.extend(temp)
            normal_10_features_dict[i_d] = text_features

            word_vec_dict[i_d] = self.gensim_model.get_vecs(final_tweets, final_tweet_lemmas, 200)

            # topic vectors: may be this will be good with less number of vecs
            text_topics = self.gensim_model.find_topics(final_tweet_lemmas)
            topic_vec_dict[i_d] = self.gensim_model.get_vecs(text_topics, text_topics, 100)

        #if self.feature_reduction: # if required
        #    normal_10_features_dict = self.create_LSA_features(normal_10_features_dict, self.lsa1)
        #    word_vec_dict = self.create_LSA_features(word_vec_dict, self.lsa2)
        #    topic_vec_dict = self.create_LSA_features(topic_vec_dict, self.lsa3)

        hash_tag_dict = self.create_LSA_hash_features(hash_tag_dict)

        for i_d in xml_file_dict:
            text_features_dict[i_d] = normal_10_features_dict[i_d].tolist() + hash_tag_dict[i_d].tolist() + \
                                      word_vec_dict[i_d].tolist() + topic_vec_dict[i_d].tolist()
        return text_features_dict

    def get_image_features(self, image_directory_dict):
        image_cap_dict = {}
        image_topic_dict = {}
        image_feature_dict = {}
        print('Generating image features')
        for i_d in image_directory_dict:
            image_captions = self.caption_generation.get_caption(i_d)
            if image_captions:
                image_captions = self.processing_tweet.processing_lemma(image_captions)
                image_cap_dict[i_d] = self.gensim_model.get_vecs(image_captions, image_captions, 100)
                image_topics = self.gensim_model.find_topics(image_captions)
                image_topic_dict[i_d] = self.gensim_model.get_vecs(image_topics, image_topics, 100)
            else:
                image_cap_dict[i_d] = [0]*100
                image_topic_dict[i_d] = [0]*100

        #if self.feature_reduction:
        #    image_cap_dict = self.create_LSA_features(image_cap_dict, self.lsa4)
        #    image_topic_dict = self.create_LSA_features(image_topic_dict, self.lsa5)

        for i_d in image_directory_dict:
            image_feature_dict[i_d] = image_cap_dict[i_d].tolist() + image_topic_dict[i_d].tolist()
        return image_feature_dict

    def for_testing(self, test_input_address):
        self.status_tt = 'testing'
        if test_input_address == '' or not os.path.exists(os.path.abspath(test_input_address)):
            print('Test address found.')
            sys.exit(1)
        labels, xml_file_dict, image_directory_dict = self.read_write.files_in_folder(test_input_address)
        print('number of files to process = ' + str(len(xml_file_dict)))
        all_features = {}
        text_features_dict = self.get_text_features(xml_file_dict)
        image_features_dict = self.get_image_features(image_directory_dict)
        for i_d in xml_file_dict:
            all_features[i_d] = text_features_dict[i_d] + image_features_dict[i_d]

        text_labels = self.classification.test(text_features_dict, self.model_text)
        image_labels = self.classification.test(image_features_dict, self.model_image)
        all_labels = self.classification.test(all_features, self.model_all)
        return text_labels, image_labels, all_labels

    def for_training(self, train_input_address):
        self.status_tt = 'training'
        if train_input_address == '' or not os.path.exists(os.path.abspath(train_input_address)):
            print('Training address not found.')
            sys.exit(1)
        # label is dict
        labels, xml_file_dict, image_directory_dict = self.read_write.files_in_folder(train_input_address)
        # for i_d in labels: # for now i am doing with xml_file_dict later i need to do this
        print('number of files to process = ' + str(len(xml_file_dict)))

        all_features = {}
        text_features_dict = self.get_text_features(xml_file_dict)
        image_features_dict = self.get_image_features(image_directory_dict)
        for i_d in xml_file_dict:
            all_features[i_d] = text_features_dict[i_d] + image_features_dict[i_d]


        '''self.classification.ten_fold_cross_validation(all_features, labels)
        self.classification.ten_fold_cross_validation(text_features_dict, labels)
        self.classification.ten_fold_cross_validation(image_features_dict, labels)'''

        print('Training text features')
        self.model_text = self.classification.train(text_features_dict, labels)

        print('Training image features')
        self.model_image = self.classification.train(image_features_dict, labels)

        print('Training all features')
        self.model_all = self.classification.train(all_features, labels)

        self.save_all_models_after_trainging()

        # This is for just writing the rest of emojis which are not in happy or sad emoji emotion lexicon
        # self.processing_tweet.write_unk_emoji()

    def save_all_models_after_trainging(self):
        # saving tf-idf and lsa models
        self.classification.save_model(self.lsa1, self.model_paths[0])
        self.classification.save_model(self.lsa2, self.model_paths[1])
        self.classification.save_model(self.lsa3, self.model_paths[2])
        self.classification.save_model(self.lsa4, self.model_paths[3])
        self.classification.save_model(self.lsa5, self.model_paths[4])
        self.classification.save_model(self.lsa6, self.model_paths[5])
        self.classification.save_model(self.vectorizer, self.model_paths[6])

        # saving all the classificatin models.
        self.classification.save_model(self.model_text, self.model_paths[7])
        self.classification.save_model(self.model_image, self.model_paths[8])
        self.classification.save_model(self.model_all, self.model_paths[9])

    def load_all_models(self):
        # loading tf-idf and lsa models
        self.lsa1 = self.classification.load_model(self.model_paths[0])
        self.lsa2 = self.classification.load_model(self.model_paths[1])
        self.lsa3 = self.classification.load_model(self.model_paths[2])
        self.lsa4 = self.classification.load_model(self.model_paths[3])
        self.lsa5 = self.classification.load_model(self.model_paths[4])
        self.lsa6 = self.classification.load_model(self.model_paths[5])
        self.vectorizer = self.classification.load_model(self.model_paths[6])

        # laoding all the classification models
        self.model_text = self.classification.load_model(self.model_paths[7])
        self.model_image = self.classification.load_model(self.model_paths[8])
        self.model_all = self.classification.load_model(self.model_paths[9])

    def already_trained(self):
        for path in self.model_paths:
            if os.path.exists(os.path.abspath(path)):
                continue
            else:
                return False
        return True

    def process_all(self, training_input_add, test_input_add, test_output_add):
        if self.already_trained():
            print('Found trained models. No need to train again.')
            self.load_all_models()
            print('Loaded all the models.')
        else:
            print('Training required. Going for training.')
            self.for_training(training_input_add)
            print('Training complete. Goining for test')
        text_labels, image_labels, combined_labels = self.for_testing(test_input_add)
        self.read_write.format_n_write_output(test_output_add, text_labels, image_labels, combined_labels)


def main():
    language = 'en'
    # TODO: make a config parser to control the variables
    training_input_add = sys.argv[1] + '/' + language + '/'
    test_input_add = sys.argv[2] + '/' + language + '/'
    test_output_add = sys.argv[3] + '/' + language + '/'

    print('Make sure that the addresses are correct')
    print('Training address = ' + training_input_add)
    print('Test address = ' + test_input_add)
    print('Test output address = ' + test_output_add)
    x = Main()
    x.process_all(training_input_add, test_input_add, test_output_add)


if __name__ == '__main__':
    main()
