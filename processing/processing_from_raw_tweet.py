import re
import string


class RawProcessing:

    def __init__(self):
        self.punc_list = set(string.punctuation)

    def raw_processing(self, tweets):
        new_tweets = []
        hash_count = 0
        direct_tweet_count = 0
        repeat_punc_count = 0
        punc_count = 0
        url_count = 0
        hash_tags, hash_count = self.get_hash_tag_info(tweets)
        for tweet in tweets:
            # hash_count += self.count_hashtag(tweet)
            temp_count, tweet = self.count_directTweet(tweet)
            direct_tweet_count += temp_count
            temp_rep_punc, temp_punc = self.count_repeat_punc(tweet)
            repeat_punc_count += temp_rep_punc
            punc_count += temp_punc
            temp_count, tweet = self.count_url(tweet)
            url_count += temp_count
            new_tweets.append(self.remove_repeat_punc(tweet))
        final_features = [hash_count, direct_tweet_count, repeat_punc_count, punc_count, url_count]
        return new_tweets, final_features, hash_tags

    def get_hash_tag_info(self, tweets):
        count = 0
        hash_tags = []
        for tweet in tweets:
            regex_pattern = "#[\w]*"
            single_tweet_hashtags = re.findall(regex_pattern, tweet)
            count += len(single_tweet_hashtags)
            hash_tags.extend(single_tweet_hashtags)
        return ' '.join(hash_tags), count

    def count_directTweet(self, tweet):
        regex_pattern = "@[\w]+"
        dt = re.findall(regex_pattern, tweet)
        for name in dt:
            tweet = tweet.replace(name, '@name')
        return len(dt), tweet

    def count_url(self, tweet):
        urls = re.findall(r'(https?://[^\s]+)', tweet)
        new_urls = re.findall(r'(http?://[^\s]+)', tweet)
        for url in urls + new_urls:
            tweet = tweet.replace(url, 'URL')
        return len(urls) + len(new_urls), tweet

    def count_repeat_punc(self, tweet):
        num_puc = 0
        last = None
        visited = None
        output = 0
        for c in tweet:
            if c in self.punc_list:
                num_puc += 1
            if c == last:
                if c in self.punc_list:
                    if visited != c:
                        output += 1
                        visited = c
            else:
                last = c
        return output, num_puc

    def remove_repeat_punc(self, tweet):
        last = None
        last_pos = None
        output = []
        for pos, c in enumerate(tweet):
            if not c == last or not pos-1 == last_pos:
                if c in self.punc_list:
                    last = c
                    last_pos = pos
                output.append(c)
        tweet = ''.join(output)
        tweet = tweet.replace('...', '.').replace('..', '.').replace('…', '.').replace('!!', '!')
        return tweet