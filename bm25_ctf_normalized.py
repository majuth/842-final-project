import json
from collections import defaultdict
import math
from porterStemming import PorterStemmer

stopwords=[]
with open('stopwords.txt', 'r') as f:
    for term in f:
        term = term.split('\n')
        stopwords.append(term[0])

# read in dataset
with open('movies_metadata.json', 'r') as f:
  corpus = json.load(f)

doc_length = []
doc_contents = {}
doc_contents_lst = []

for key in corpus.keys():
    docID = corpus[key]['id']
    contents = corpus[key]['title'] + " " + corpus[key]['overview']
    doc_length.append(len(contents))
    contents_list=contents.split()
    filtered_terms = []
    acceptable_characters = set('-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    # filter out punctuation
    for term in contents_list:
      answer = ''.join(filter(acceptable_characters.__contains__, term))
      filtered_terms.append(answer.lower())
    stemmed_terms=[]
    # apply stemming
    for term in filtered_terms:
      p=PorterStemmer()
      stemmed_term = []
      stemmed_term = (p.stem(term, 0, len(term)-1))
      stemmed_terms.append("".join(stemmed_term))
      filtered_terms=stemmed_terms
    # remove stopwords
    stopwords_removed = [term for term in filtered_terms if term not in stopwords]
    filtered_terms = stopwords_removed
    # remove blank list items
    filtered_terms = (list(filter(None, filtered_terms)))
    doc_contents_lst.extend(filtered_terms)
    doc_contents[docID] = filtered_terms

NUM_OF_DOCS = len(corpus)
LEN_DOCS = sum(([doc for doc in doc_length]))
AVG_LEN_DOCS = LEN_DOCS/NUM_OF_DOCS
# DONE: get total number of terms in document -> use filtered version? or nah?
NUM_OF_TERMS = LEN_DOCS #1582624
NUM_TERMS_FILTERED = len(doc_contents_lst) #188229

inverted_index = defaultdict(set)
for docid, terms in doc_contents.items():
    for term in terms:
        inverted_index[term].add(docid)

def tf_idf_score_with_ictf(k1, b, term, docid):
    # get document frequency by counting length of inverted index
    doc_freq = len(inverted_index[term.lower()])

    # get term frequency by counting number of times term appears in given doc
    term_freq = doc_contents[str(docid)].count(term)

    # get collection term frequency by counting number of times term appears in collection
    collec_freq = doc_contents_lst.count(term)

    # calculate td-idf score with ictf
    ictf_comp = math.log((NUM_TERMS_FILTERED - collec_freq + 0.5)/(collec_freq+0.5))
    idf_comp = math.log((NUM_OF_DOCS - doc_freq + 0.5)/(doc_freq+0.5))
    ictf_idf_comp = ictf_comp*idf_comp
    tf_comp = ((k1 + 1)*term_freq)/(k1*((1-b) + b*(len(list(filter(None, doc_contents[str(docid)])))/AVG_LEN_DOCS))+term_freq)
    return ictf_idf_comp * tf_comp

# creates a matrix for all terms
def create_tf_idf(k1, b):
    tf_idf = defaultdict(dict)
    for term in set(inverted_index.keys()):
        for docid in inverted_index[term]:
            tf_idf[term][docid] = tf_idf_score_with_ictf(k1, b, term, docid)
    return tf_idf

# Create tf_idf matrix and write to json file -- uncomment if no tf_idf_with_icf.json
# tf_idf = create_tf_idf(1.5, 0.5) # k1 = 1.5, b = 0.5 (default values)
# with open("tf_idf_with_ictf.json","w") as f:
#     json.dump(tf_idf,f)

# read in tf_idf matrix from json file
with open('tf_idf_with_ictf.json', 'r') as f:
  tf_idf = json.load(f)

def get_query_tf_comp(k3, term, query_tf):
    return ((k3+1)*query_tf[term])/(k3 + query_tf[term])

def retrieve_docs_baseline(query, result_count):
    q_terms = [term.lower() for term in query.split() if term not in stopwords]
    query_tf = {}
    for term in q_terms:
        query_tf[term] = query_tf.get(term, 0) + 1
    
    scores = {}
    for word in query_tf.keys():
        for document in inverted_index[word]:
            scores[document] = scores.get(document, 0) + (tf_idf[word][document] * get_query_tf_comp(0,word,query_tf)) #k3 = 0 (default)
    return sorted(scores.items(), key=lambda x : x[1], reverse=True)[:result_count]

def retrieve_term_freq(query, docid):
    total_tf=0
    for term in query.split():
        total_tf+=(doc_contents[str(docid)].count(term))
    return (total_tf)

# queryTerm = ""
# while queryTerm != "ZZEND":
#     queryTerm = input("Enter the term you are searching for: ")
    
#     # stem query
#     queryTerm_stemmed = []
#     for word in queryTerm.split():
#         p=PorterStemmer()
#         queryTerm_list = []
#         queryTerm_list = (p.stem(word, 0, len(word)-1))
#         queryTerm_stemmed.append("".join(queryTerm_list))

#     queryTerm_stemmed = ' '.join(queryTerm_stemmed)

#     rankings = retrieve_docs_baseline(queryTerm_stemmed, 100000)
#     print("There are %d results" %len(rankings))

#     all_scores = []
#     for i in range (0, len(rankings)):
#         if rankings[i][1] > 0:
#             all_scores.append(rankings[i][1])
#     sum_scores = sum(([score for score in all_scores]))

#     for i in range (0, len(rankings)):
#         if rankings[i][1] > 0:
#             score = rankings[i][1]
#             print("BM25-ctf Ranking: %f" %score)
#             normalized_score = score/sum_scores
#             print("BM25-ctf-normalized Ranking: %f" %normalized_score)
#             docID = rankings[i][0]
            # print("Doc ID: %s" %docID)
            # print("Title:")
            # print(corpus[docID]['title'])
            # print("Overview:")
            # print(corpus[docID]['overview'][:300] + "...")
            # print("Docment Frequency:")
            # print(retrieve_term_freq(queryTerm_stemmed, docID))
            # print("\n")