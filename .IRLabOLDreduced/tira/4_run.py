from nltk.stem import WordNetLemmatizer
import nltk
import xml.etree.ElementTree as ET
import argparse
from elasticsearch import Elasticsearch
import pandas as pd
from time import sleep

#PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-dir")
parser.add_argument("-o", "--output-dir")
args = parser.parse_args()
args = vars(args)

#HANDLE INPUT
input_dir = args['input_dir']
output_dir = args['output_dir']

sleep(240)

es = Elasticsearch(hosts="http://irlab_elastic_1:9200")

while not es.ping():
    sleep(10)
    print("WAITING...")
    es = Elasticsearch(hosts="http://irlab_elastic_1:9200", timeout=300)
print("done waiting")

###############
# make dictionary with all topics
tree = ET.parse('topics.xml')
# tree = ET.parse('topics_selected.xml')
topics = tree.findall('topic')

# Dictionary with all 50 topics
topicsDic = {}

for topic in topics:
    title = topic.find('title').text
    number = topic.find('number').text
    topicsDic[number] = title

num_results = 10

# fourth run: sentiment+OCR+ImageClustering+QueryPreprocessing
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')

# Method to preprocess the query


def query_preprocessing(query):
    stopwords = {'should', 'get', 'is', 'with', 'be', 'used', 'in', 'a', 'it', 'who',
                 'have', 'to', 'for', 'the', 'can', 'at', 'or', 'an',  'does', 'do', 'are', 'our'}
    query = query.replace("?", "")

    rslt_query = ''
    for word in query.split():
      if word.lower() not in stopwords:
        word = lemmatizer.lemmatize(word)
        rslt_query += word + ' '
    #print(query + ': ' + rslt_query)

    return rslt_query

# Method to search the index


def search_refined_OCR_Clustering_prototype(query, num_results):
    body_positive = {
        "from": 0,
        "size": num_results,
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"document_text": {"query": query}}},
                            {"match": {"ocr_text": {"query": query, "boost": 5}}}
                        ],

                        "filter": [
                            {"range": {"sentiment": {"gt": 0}}}
                        ]
                    }
                },
                #"boost": "5",
                "functions": [
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "0"}}, "weight": 5.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "1"}}, "weight": 1.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "2"}}, "weight": 3.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "3"}}, "weight": 3.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "4"}}, "weight": 4.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "5"}}, "weight": 1.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "6"}}, "weight": 5.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "7"}}, "weight": 3.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "8"}}, "weight": 1.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "9"}}, "weight": 1.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "10"}}, "weight": 2.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "11"}}, "weight": 5.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "12"}}, "weight": 1.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "13"}}, "weight": 2.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "999"}}, "weight": 0.0}
                ],
                "max_boost": 5.0,
                "score_mode": "max",
                "boost_mode": "multiply",
                "min_score": 0.0
            }
        }
    }
    res_positive = es.search(index="final_boromir_index", body=body_positive)

    body_negative = {
        "from": 0,
        "size": num_results,
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"document_text": {"query": query}}},
                            {"match": {"ocr_text": {"query": query, "boost": 5}}}
                        ],

                        "filter": [
                            {"range": {"sentiment": {"lt": 0}}}
                        ]
                    }
                },
                #"boost": "5",
                "functions": [
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "0"}}, "weight": 5.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "1"}}, "weight": 1.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "2"}}, "weight": 3.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "3"}}, "weight": 3.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "4"}}, "weight": 4.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "5"}}, "weight": 1.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "6"}}, "weight": 5.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "7"}}, "weight": 3.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "8"}}, "weight": 1.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "9"}}, "weight": 1.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "10"}}, "weight": 2.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "11"}}, "weight": 5.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "12"}}, "weight": 1.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "13"}}, "weight": 2.0},
                    {"filter": {
                        "match": {"cluster_when_14_clusters_in_10dim": "999"}}, "weight": 0.0}
                ],
                "max_boost": 5.0,
                "score_mode": "max",
                "boost_mode": "multiply",
                "min_score": 0.0
            }
        }
    }
    res_negative = es.search(index="final_boromir_index", body=body_negative)

    # get ID's from retrieved documents
    resultPos = []
    results_positive = res_positive.get('hits').get('hits')

    for doc in results_positive:
        id = doc.get('_id')
        score = doc.get('_score')
        resultPos.append([id, score])

    resultNeg = []
    results_negative = res_negative.get('hits').get('hits')

    for doc in results_negative:
        id = doc.get('_id')
        score = doc.get('_score')
        resultNeg.append([id, score])

    return resultPos, resultNeg


resultList = []
for topic in topicsDic:
    query = query_preprocessing(topicsDic.get(topic))
    result_ids_pro, result_ids_con = search_refined_OCR_Clustering_prototype(
        query, num_results)
    i = 1
    for each in result_ids_pro:
        resultList.append([topic, "PRO", each[0], i, each[1], "Boromir4"])
        i += 1
    i = 1
    for each in result_ids_con:
        resultList.append([topic, "CON", each[0], i, each[1], "Boromir4"])
        i += 1

resultdf = pd.DataFrame(resultList, columns=[
                        'topicID', 'stance', 'pageID', 'rank', 'score', 'Method'])
with open(f'{output_dir}/run.txt', 'a+') as f:
    resultdf[['topicID', 'stance', 'pageID', 'rank', 'score', 'Method']].to_csv(
        f, sep=' ', header=False, index=False)

print(resultdf)
