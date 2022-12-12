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
    doc_contents[docID] = filtered_terms

NUM_OF_DOCS = len(corpus)
LEN_DOCS = sum(([doc for doc in doc_length]))

ALPHA = 1
P_DOC = 1 / NUM_OF_DOCS

def pWordGivenDoc(alpha, term, docid):
    df = doc_contents[str(docid)].count(term)
    len_doc = len(doc_contents[str(docid)])

    len_unique_terms = len(set(doc_contents[str(docid)]))

    return (df + alpha) / (len_doc + (alpha * len_unique_terms))

def createProbailityMatrix(alpha, docContents):
    probability_matrix = defaultdict(dict)
    for docID in docContents.keys():
        for term in set(docContents[docID]):
            probability_matrix[term][docID] = pWordGivenDoc(alpha, term, docID)
    return probability_matrix

matrix = createProbailityMatrix(ALPHA, doc_contents)

def retrieve_docsNB(alpha, query, result_count):
    possibleDocuments = set()
    for word in query.split():
        if word in matrix:
            for docID in list(matrix[word].keys()):
                possibleDocuments.add(docID)
    
    scores = {}
    for doc in possibleDocuments:
        score = P_DOC
        for word in query.split():
            if doc in list(matrix[word].keys()):
                score = score * matrix[word][doc]
            else:
                score = score * (alpha / LEN_DOCS)
        scores[str(doc)] = score
    #print("Valid scores: " + str(scores))
    return sorted(scores.items(), key=lambda x : x[1], reverse=True)[:result_count]

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

#     rankings = retrieve_docs(ALPHA, queryTerm_stemmed, 10)
#     #print("Outputable rankings: " + str(rankings))
#     print("There are %d results" %len(rankings))
#     for i in range (0, len(rankings)):
#         if rankings[i][1] > 0:
#             print("Naive Bayes Ranking: %f" %rankings[i][1])
#             docID = rankings[i][0]
#             print("Doc ID: %s" %docID)
#             print("Title:")
#             print(corpus[docID]['title'])
#             print("Overview:")
#             print(corpus[docID]['overview'][:300] + "...")
#             print("\n")