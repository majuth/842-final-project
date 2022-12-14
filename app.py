# importing Flask and other modules
from flask import Flask, request, render_template
import os
import json
from plotly import plotly
from collections import defaultdict
import math
from porterStemming import PorterStemmer
from bm25 import retrieve_docs


# read in dataset
with open('movies_metadata.json', 'r') as f:
  corpus = json.load(f)

# Flask constructor
app = Flask(__name__)

# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["GET", "POST"])
def home():
    return render_template('form.html')
           
@app.route('/bm25', methods=['GET', 'POST'])
def bm25():
    query_term = request.form.get("user_query")
    # stem query
    query_term_stemmed = []
    for word in query_term.split():
        p=PorterStemmer()
        query_term_list = []
        query_term_list = (p.stem(word, 0, len(word)-1))
        query_term_stemmed.append("".join(query_term_list))
    query_term_stemmed = ' '.join(query_term_stemmed)
    rankings = retrieve_docs(query_term_stemmed, 100000)
    num_results = len(rankings)
    titles = []
    overviews = []
    bm25_rankings = []
    for i in range (0, num_results):
        if rankings[i][1] > 0:
            bm25_rankings.append(rankings[i][1])
            docid = rankings[i][0]
            titles.append(corpus[docid]['title'])
            overviews.append(corpus[docid]['overview'][:300] + "...")
    lengthRange = range(0, num_results)

    fig = plotly.graph_objs.Figure(data=[plotly.graph_objs.Scatter(
    x=[1, 2, 3, 4], y=[10, 11, 12, 13],
    mode='markers',
    marker_size=[40, 60, 80, 100])
    ])

    fig.write_html("chart.html")

    return render_template('output.html', n=num_results, r=bm25_rankings, t=titles, o=overviews, l=lengthRange)

if __name__=='__main__':
    app.run(host=os.getenv('IP', '0.0.0.0'), 
        port=int(os.getenv('PORT', 7777)))