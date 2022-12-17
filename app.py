# importing Flask and other modules
from flask import Flask, request, render_template
import os
import json
from porterStemming import PorterStemmer
from bm25 import retrieve_docs, retrieve_term_freq
from naiveBayes import retrieve_docsNB
import plotly
import plotly.express as px
import pandas as pd

# read in dataset
with open('movies_metadata.json', 'r') as f:
  corpus = json.load(f)

# Flask constructor
main = Flask(__name__)

# A decorator used to tell the application
# which URL is associated function
@main.route('/', methods =["GET", "POST"])
def home():
    return render_template('form.html')
           
@main.route('/bm25', methods=['GET', 'POST'])
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
    user_ratings = []
    genres = []
    tf = []
    for i in range (0, num_results):
        if rankings[i][1] > 0:
            bm25_rankings.append(rankings[i][1])
            docid = rankings[i][0]
            titles.append(corpus[docid]['title'])
            overviews.append(corpus[docid]['overview'][:300] + "...")
            user_ratings.append(float(corpus[docid]['user_rating']))
            genre_dict = eval((corpus[docid]['genres']))
            genres.append(genre_dict[0]['name'])
            tf.append(retrieve_term_freq(query_term_stemmed, docid))
    length_range = range(0, num_results)

    zipped = list(zip(bm25_rankings, user_ratings, titles, genres, tf))

    df = pd.DataFrame(zipped, columns=['BM25_Ranking', 'User_Rating', 'Movie_Title', 'Genre', 'Term Frequency'])

    fig = px.scatter(df, x="BM25_Ranking", y="User_Rating", color="Genre", size="Term Frequency", hover_name="Movie_Title", log_x=True, size_max=30)
    fig.update_layout(
        yaxis=dict(
            title_text = "User Rating",
            tickvals=[4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10],
        ),
        xaxis=dict(
            title_text = "BM25 Ranking"
        ),
    )

    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('output.html', q=query_term, n=num_results, r=bm25_rankings, t=titles, o=overviews, l=length_range, graph=graph_json)

@main.route('/naivebayes', methods=['GET', 'POST'])
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
    user_ratings = []
    genres = []
    tf = []
    for i in range (0, num_results):
        if rankings[i][1] > 0:
            nb_rankings.append(rankings[i][1])
            docid = rankings[i][0]
            titles.append(corpus[docid]['title'])
            overviews.append(corpus[docid]['overview'][:300] + "...")
            user_ratings.append(float(corpus[docid]['user_rating']))
            genre_dict = eval((corpus[docid]['genres']))
            genres.append(genre_dict[0]['name'])
            tf.append(retrieve_term_freq(query_term_stemmed, docid))
    length_range = range(0, num_results)

    zipped = list(zip(nb_rankings, user_ratings, titles, genres, tf))

    df = pd.DataFrame(zipped, columns=['NB_Ranking', 'User_Rating', 'Movie_Title', 'Genre', 'Term Frequency'])

    fig = px.scatter(df, x="NB_Ranking", y="User_Rating", color="Genre", size="Term Frequency", hover_name="Movie_Title", log_x=True, size_max=30)
    fig.update_layout(
        yaxis=dict(
            title_text = "User Rating",
            tickvals=[4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10],
        ),
        xaxis=dict(
            title_text = "Naive Bayes Ranking"
        ),
    )

    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('outputNB.html',q=query_term, n=num_results, r=nb_rankings, t=titles, o=overviews, l=length_range, graph=graph_json)

@main.route('/compare', methods=['GET', 'POST'])
def compare():
    query_term = request.form.get("user_query")
    # stem query
    query_term_stemmed = []
    for word in query_term.split():
        p=PorterStemmer()
        query_term_list = []
        query_term_list = (p.stem(word, 0, len(word)-1))
        query_term_stemmed.append("".join(query_term_list))
    query_term_stemmed = ' '.join(query_term_stemmed)
    bm25_rankings = retrieve_docs(query_term_stemmed, 100000)
    nb_rankings = retrieve_docsNB(1,query_term_stemmed, 100000)
    num_results = len(bm25_rankings)
    titles = []
    overviews = []
    rankings_bm25 = []
    rankings_nb = []
    user_ratings = []
    genres = []
    tf = []
    for i in range (0, num_results):
        if bm25_rankings[i][1] > 0:
            rankings_bm25.append(bm25_rankings[i][1])
            rankings_nb.append(nb_rankings[i][1])
            docid = bm25_rankings[i][0]
            titles.append(corpus[docid]['title'])
            overviews.append(corpus[docid]['overview'][:300] + "...")
            user_ratings.append(float(corpus[docid]['user_rating']))
            genre_dict = eval((corpus[docid]['genres']))
            genres.append(genre_dict[0]['name'])
            tf.append(retrieve_term_freq(query_term_stemmed, docid))
    length_range = range(0, num_results)

    zipped = list(zip(rankings_bm25, rankings_nb, user_ratings, titles, genres, tf))
    df = pd.DataFrame(zipped, columns=['BM25_Ranking', 'NB_Ranking', 'User_Rating', 'Movie_Title', 'Genre', 'Term Frequency'])
    bm25_fig = px.scatter(df, x="BM25_Ranking", y="User_Rating", color="Genre", size="Term Frequency", hover_name="Movie_Title", log_x=True, size_max=30)
    bm25_fig.update_layout(
        yaxis=dict(
            title_text = "User Rating",
            tickvals=[4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10],
        ),
        xaxis=dict(
            title_text = "BM25 Ranking"
        ),
    )
    bm25_graph_json = json.dumps(bm25_fig, cls=plotly.utils.PlotlyJSONEncoder)

    nb_fig = px.scatter(df, x="NB_Ranking", y="User_Rating", color="Genre", size="Term Frequency", hover_name="Movie_Title", log_x=True, size_max=30)
    nb_fig.update_layout(
        yaxis=dict(
            title_text = "User Rating",
            tickvals=[4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10],
        ),
        xaxis=dict(
            title_text = "NB Ranking"
        ),
    )
    nb_graph_json = json.dumps(nb_fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('compare.html', q=query_term, n=num_results, bm_r=rankings_bm25, nb_r=rankings_nb, t=titles, o=overviews, l=length_range, bm_graph=bm25_graph_json, nb_graph=nb_graph_json)


if __name__=='__main__':
    main.run(host=os.getenv('IP', '0.0.0.0'), 
        port=int(os.getenv('PORT', 7777)), debug=True)
    