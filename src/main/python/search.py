import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import math
from nltk.stem import *   #From nltk.org


from readers import read_queries, read_documents

inverted_index = {}
doc_count = 0


def remove_not_indexed_toknes(tokens):
    return [token for token in tokens if token in inverted_index]   # For every token that's in the inverted index
                                                                    # add it to a list that gets returned

#This function merges the postings together
#   if the posting are the same go to next posting on BOTH lists
#   else if the second posting is greater than the first go to the next posting on the first porting list
#   else go to the next index posting in the second posting.
#   This will return the merged list
def merge_two_postings(first, second):
    #https: // stackoverflow.com / questions / 1319338 / combining - two - lists - and -removing - duplicates - without - removing - duplicates - in -orig
    setlist = set(second) - set(first)
    return first + list(setlist) #This changes it into an OR instead of an AND


# indexed_token is an array
# inverted_index is a dictionary[key[postion in the array/list]]
def merge_postings(indexed_tokens):
    first_list = inverted_index[indexed_tokens[0]].keys()  #postings for the first token
    second_list = []
    for each in range(1, len(indexed_tokens)):  # number of tokens we have
        second_list = inverted_index[indexed_tokens[each]].keys()        # Pointing to Documents ID ["key", values(list)] think map
        first_list = merge_two_postings(first_list, second_list)       #See merge_two_posting
    return first_list


def search_query(query):
    tokens = tokenize(str(query['query']))
    indexed_tokens = remove_not_indexed_toknes(tokens)   # remove tokens we do not have
    if len(indexed_tokens) == 0:
        return []    # returns an empty array
    elif len(indexed_tokens) == 1:   # If there is only one
        return inverted_index[indexed_tokens[0]].keys()   # returning the posting list because there is only one element
    else:
        return rank_results(indexed_tokens)
    #
    # def synonyms(text):
    #     synonyms = []
    #     from PyDictionary.test_pydictionary import dictionary
    #     list_of_syn = dictionary.synonym("good")
    #     for syn in wordnet.synsets(list_of_syn):
    #         for text in syn.lemmas():
    #             synonyms.append(text.name())
    #     return set(synonyms)

def tokenize(text):
    stopWords = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english", ignore_stopwords = True)  # from nltk.org
    tokensplit = re.split("-|\s|,|\.|\(|\)", text)
    #specialchar = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', text)
    stemlist = []
    for term in tokensplit:
        if term not in stopWords:
            stemlist.append(stemmer.stem(term))
    #synonyms = wordnet.synsets('heat')[0]
    #synonyms.definition()
    #specialchar = re.split(r'[`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]', text)
    #https: // stackoverflow.com / questions / 21023901 / how - to - split - at - all - special - characters - with-re - split
    #return [term for term in specialchar if not term in stopWords]  # Stop words split by hyphens or make an array list
    #https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
    #return [term for term in re.split("-|\s|,|\.|\ing", text) if not term in stopWords]  # Stop words split by hyphens or make an array list
    return stemlist

#log termfrequency * (log (n) / idf)

#term frequency = {Key = token, [list of document_ids] }
#tf[(token, doc_id)] = term_frequency


# Have this function to do term frequency (per documents) and IDF (All documents)
# Term Frequency - Count terms in the document


def add_token_to_index(token, doc_id):
    # check if token in inverted ind
        if token in inverted_index:
            current_postings = inverted_index[token]
            if doc_id in current_postings:
                inverted_index[token][doc_id] += 1
            else:
                inverted_index[token][doc_id] = 1
        else:
            inverted_index[token] = {doc_id : 1}

def calculate_idf(token):
    idf = float(math.log10(doc_count/len(inverted_index[token])))
    return idf

def term_freq(token, doc_id):
    return inverted_index[token][doc_id]  #returns the number of times the term occurs in the document with that key token


#This function ranks the doc id by score from greatest to smallest
def rank_results(query):
    notranked = merge_postings(query)  # applies mergepostings and this will be a list returning document ids
    ranked = {}     # This creates a new dictionary
    for doc_id in notranked:  #For every document in the notranked list
        score = probability_score(query, doc_id)
        ranked.update({doc_id : score})    #Updates the ranked dictionary
    #https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    sorted_ranked = sorted(ranked, key=ranked.get, reverse=True)      #Sorts the score
    return sorted_ranked


def calculate_dcg(query, documents):
    count = 1
    sum = 0.0
    for document in documents:
        id = str(query['query number']) + "_" + str(document)
        if id in query_document_relevance:
            relevance = query_document_relevance[id]
            sum = sum + relevance / math.log10(count + 1)
        count = count + 1
    return sum

    # def length_normalize(): #This will take vector
    #     for token in inverted_index:
    #         sum_of_vectors = (inverted_index)
    #     length_of_vector = math.sqrt()
    #     return 0
#Normalizing the length of a vector
#Divide each of the components by its length



#The query is represented as a weighted tf-idf vector
#Each document as a weighted tf-idf vector
#Compute the Cos Sim Score for the document vector and query vector
#Rank documents with respect to query by score
#Return the top K to the user (Rank the document by the smallest angle)
#Sort the documents
        # Calculate the query term and fetch postings list for that query
        # For each document in the postings list each term has a frequency in that document (scale by log)
        # Dot product of the query terms and summing to the scores array
    #Length of each document for each document
    #Divide the score array of each document = Scores[d]/Length[d] (Normalization of the different document
    #Return top K components of Scores (high values for scores)
    #dp = dot_product(first, second)
def probability_score(query, doc_id):
    #float scores[N] = 0 # score array for all documents for each query terms
    #float Length[N] = 0 # array for the different documents
    score = 0.0
    for token in query:  # For each term in the query
        if doc_id in inverted_index[token]:
            # score += calculate_idf(token) * term_freq(token, doc_id) #Cosine of Query
            score += calculate_idf(token)  #Probability
    return score




#TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)
#IDF(t) = log_e(Total number of documents / Number of documents with term t in it)
def add_to_index(document): #(Array of tokens)
    for token in tokenize(document['title']):
        add_token_to_index(token, document['id'])
    for token in tokenize(document['body']):
        add_token_to_index(token, document['id'])



def create_index():
    global doc_count
    for document in read_documents():
        doc_count += 1
        add_to_index(document)
    print ("Created index with size {}".format(len(inverted_index)))

create_index()
if __name__ == '__main__':
    all_queries = [query for query in read_queries() if query['query number'] != 0]
    for query in all_queries:
        documents = search_query(query)
        print ("Query:{} and Results:{}".format(query, documents))
