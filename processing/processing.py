import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

import emoji


class Processing:

    def __init__(self, read_n_write):
        self.happy_emoticons = read_n_write.read_any_list('./resources/happy_emoticons.txt')
        self.sad_emoticons = read_n_write.read_any_list('./resources/sad_emoticons.txt')
        self.slang = read_n_write.read_any_list('./resources/slang.txt')
        self.wnl = nltk.WordNetLemmatizer()
        self.tokenizer = TweetTokenizer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(['url', "i'm", '@name', "@name's", "that's", "doesn't", 'u', 'would', 'else',
                                'anyone', "can't", "what's", "i've", 'could', "they're"])

        self.happy_emoticons_count = 0
        self.sad_emoticons_count = 0
        self.emoji_count = 0
        self.slang_count = 0
        self.stopwords_count = 0
        self.emoji_list = []
        self.emoji_list.extend(self.happy_emoticons)
        self.emoji_list.extend(self.sad_emoticons)

        self.punc_list = set(string.punctuation)

        self.ngrams = {}
        self.ngrams_pos = {}

    def write_unk_emoji(self):
        print('Printing not emotion annotated emoticon list')
        na_emoticon = list(set(self.emoji_list) - set(self.happy_emoticons).union(set(self.sad_emoticons)))
        print(na_emoticon)
        for emoticon in na_emoticon:
            print(emoticon + '\t' + self.emoji_dict[emoticon])

    def process(self, tweets):
        self.happy_emoticons_count = 0
        self.sad_emoticons_count = 0
        self.emoji_count = 0
        self.slang_count = 0
        self.stopwords_count = 0
        list_tweets, list_tweet_lemmas = self.processing_pos(tweets)
        final_tweet_lemmas = []
        final_tweets = []
        for list_token, list_lemmas in zip(list_tweets, list_tweet_lemmas):
            single_tweet_lemma = []
            single_tweet = []
            for word, lemma in zip(list_token, list_lemmas):
                if self.all_count(word, lemma):
                    word = word.lower()
                    single_tweet.append(word)
                    single_tweet_lemma.append(lemma)
            final_tweet_lemmas.append(single_tweet_lemma)
            final_tweets.append(single_tweet)
        final_features = [self.happy_emoticons_count, self.sad_emoticons_count, self.emoji_count, self.slang_count, self.stopwords_count]
        return final_features, final_tweets, final_tweet_lemmas

    def all_count(self, word, lemma):
        if word.strip() == '':
            return False
        emoji_flag = False
        try:
            if word in emoji.UNICODE_EMOJI or word in self.emoji_list:
                emoji_flag = True
                new_emoji = emoji.demojize(word)
                self.emoji_count += 1
                if not new_emoji in self.emoji_list:
                    self.emoji_list.append(new_emoji)
                if new_emoji in self.happy_emoticons:
                    self.happy_emoticons_count += 1
                elif new_emoji in self.sad_emoticons:
                    self.sad_emoticons_count += 1
        except:
            pass
        if word in self.slang or lemma in self.slang:
            self.slang_count += 1
        if word in self.stop_words or lemma in self.stop_words:
            self.stopwords_count += 1
            return False
        if word in self.punc_list:
            return False
        if emoji_flag:
            return False
        return True

    def processing_pos(self, tweets):
        list_lemmas = []
        list_tokens = []
        for tweet in tweets:
            words = []
            lemmas = []
            for word in self.tokenizer.tokenize(tweet):
                lemma = self.wnl.lemmatize(word.lower())
                if word in self.stop_words or word in self.punc_list or lemma in self.stop_words:
                    continue
                else:
                    lemmas.append(lemma)
                    words.append(word)
            list_lemmas.append(lemmas)
            list_tokens.append(words)
        return list_tokens, list_lemmas

    def processing_lemma(self, list_of_sent):
        list_output = []
        for sent in list_of_sent:
            words = []
            for word in self.tokenizer.tokenize(sent):
                word = self.wnl.lemmatize(word.lower())
                if word in self.stop_words or word in self.punc_list:
                    continue
                else:
                    words.append(word)
            list_output.append(words)
        return list_output