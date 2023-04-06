import snscrape.modules.twitter as sntwitter
from joblib import load
from flask import Flask, render_template, request
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from collections import Counter

model = load('sentiment_analysisV2.pkl')
vectorizer = load('vectorizer.pkl')

stop_words = StopWordRemoverFactory().get_stop_words()

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    query = request.form['text']
    max_tweets = 10

    scraped_tweets = []

    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(scraped_tweets) == max_tweets:
            break
        else:
            scraped_tweets.append(tweet.content)
    # for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query + 'lang:id').get_items()):
    #     if i >= max_tweets:
    #         break
    #     scraped_tweets.append(tweet.content)

    X = vectorizer.transform(scraped_tweets)

    results = []

    for i, tweet in enumerate(scraped_tweets):
        sentiment = model.predict(X[i])
        results.append((tweet, sentiment[0]))

        label_dict = {-1: 'NEGATIF', 0: 'NETRAL', 1: 'POSITIF'}
        labeled_results = [label_dict[result[1]] for result in results]
        sentiment_count = Counter(labeled_results)

        most_common_sentiment = sentiment_count.most_common(1)[0][0]
    return render_template('result.html', results=results, most_common_sentiment=most_common_sentiment, query=query)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
