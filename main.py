# importing Flask and other modules
from flask import Flask, request, render_template, redirect
import os
import json
from collections import defaultdict
import math
from porterStemming import PorterStemmer
from bm25 import retrieve_docs
from naiveBayes import retrieve_docsNB
# from kmeans import predict_cluster, KMeans

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

    return render_template('output.html', n=num_results, r=bm25_rankings, t=titles, o=overviews, l=lengthRange)

@app.route('/naivebayes', methods=['GET', 'POST'])
def naivebayes():
    query_term = request.form.get("user_query")
    # stem query
    query_term_stemmed = []
    for word in query_term.split():
        p=PorterStemmer()
        query_term_list = []
        query_term_list = (p.stem(word, 0, len(word)-1))
        query_term_stemmed.append("".join(query_term_list))
    query_term_stemmed = ' '.join(query_term_stemmed)
    rankings = retrieve_docsNB(1,query_term_stemmed, 100000)
    num_results = len(rankings)
    titles = []
    overviews = []
    nb_rankings = []
    for i in range (0, num_results):
        if rankings[i][1] > 0:
            nb_rankings.append(rankings[i][1])
            docid = rankings[i][0]
            titles.append(corpus[docid]['title'])
            overviews.append(corpus[docid]['overview'][:300] + "...")
    lengthRange = range(0, num_results)

    return render_template('outputNB.html', n=num_results, r=nb_rankings, t=titles, o=overviews, l=lengthRange)

# @app.route('/kmeans', methods=['GET', 'POST'])
# def kmeans():
#     query_term = request.form.get("user_query")
#     output = predict_cluster(query_term)
#     cluster = output[0]
#     result = output[1]
#     titles = result[result['cluster'] == cluster]['title']
#     titles_lst = titles.to_list()
#     num_results = len(titles)
#     overviews = []
#     for title in titles:
#         # id = result[result['title'] == title]['docID']
#         content = result[result['title'] == title]['contents']
#         overviews.append(content)
#     print("There are %d results" %len(titles))
#     lengthRange = range(0, num_results)
#     return render_template('outputKMeans.html', n=num_results,  t=titles_lst, o=overviews, l=lengthRange)

if __name__=='__main__':
    app.run(host=os.getenv('IP', '0.0.0.0'), 
        port=int(os.getenv('PORT', 7777)))