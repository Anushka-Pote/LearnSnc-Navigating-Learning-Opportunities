# app.py

from flask import Flask, render_template, jsonify
from newsapi import NewsApiClient

app = Flask(__name__, template_folder='templates')

# Replace with your NewsAPI key
newsapi_api_key = "263f24e3d72e4880ab9ce9559725bef3"

@app.route('/')
def index():
    return render_template('news.html')

@app.route('/get_it_market_news')
def get_it_market_news():
    # Initialize NewsAPI client
    newsapi = NewsApiClient(api_key=newsapi_api_key)

    # Fetch technology headlines (IT market news)
    try:
        it_market_news_data = newsapi.get_top_headlines(category='technology', language='en', country='us', page_size=50)
    except Exception as e:
        return jsonify({'error': str(e)})

    return jsonify(it_market_news_data)

if __name__ == '__main__':
    app.run(debug=True)
