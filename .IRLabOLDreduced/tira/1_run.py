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
import xml.etree.ElementTree as ET
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

## first run: sentiment+OCR ###

def search_refined_OCR_prototype(query, num_results):
    body_positive = {
        "from":0,
        "size":num_results,
        "query": {
            "bool":{
                "should":[
                    {"match": { "document_text":query}},
                    {"match": {"ocr_text":{"query":query, "boost":5}}}  # ocr_text sollte einen natürlichen boost besitzen, da die texte viel kürzer sind. Vielleicht muss dieser auch abgeschwächt werden?
                
                ],
                "filter":[
                    {"range": {"sentiment": {"gt": 0}}}
                ]
            }   
        }
    }
    res_positive = es.search(index="final_boromir_index", body=body_positive)

    body_negative = {
        "from":0,
        "size":num_results,
        "query": {
            "bool":{
                "should":[
                    {"match": { "document_text":query}},
                    {"match": {"ocr_text":{"query":query, "boost":5}}}  # ocr_text sollte einen natürlichen boost besitzen, da die texte viel kürzer sind. Vielleicht muss dieser auch abgeschwächt werden?
                

                ],
                "filter":[
                    {"range": {"sentiment": {"lt": 0}}}
                ]            
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
    query = topicsDic.get(topic)
    result_ids_pro, result_ids_con = search_refined_OCR_prototype(query, num_results)
    i = 1
    for each in result_ids_pro:
        resultList.append([topic, "PRO", each[0], i, each[1], "Boromir1"])
        i += 1
    i = 1
    for each in result_ids_con:
        resultList.append([topic, "CON", each[0], i, each[1], "Boromir1"])
        i += 1

resultdf = pd.DataFrame(resultList, columns = ['topicID','stance','pageID','rank','score','Method'])
with open(f'{output_dir}/run.txt', 'a+') as f:
    resultdf[['topicID','stance','pageID','rank','score','Method']].to_csv(f, sep=' ', header=False, index=False)

print(resultdf)