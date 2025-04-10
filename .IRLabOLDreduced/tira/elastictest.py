from elasticsearch import Elasticsearch

# es = Elasticsearch(hosts="irlab_elastic_1")
es = Elasticsearch(hosts="http://irlab_elastic_1:9200", timeout = 300)

print(es.ping())

indices=es.indices.get_alias().keys()
sorted(indices)

# es.indices.delete(index='test-index', ignore=[400, 404])
