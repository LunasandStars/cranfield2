import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import math


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
    setlist = set(first + second)
    return list(setlist) #This changes it into an OR instead of an AND
    """
    first_index = 0
    second_index = 0
    merged_list = []
    
    while first_index < len(first) and second_index < len(second): # while both lists are within range
        if first[first_index] == second[second_index]: # if they contain the same element ex: "2" = "2"
            merged_list.append(first[first_index]) #add it to the list mergelist
            first_index = first_index + 1  #increment one to first index
            second_index = second_index + 1  #increment one to second index
        elif first[first_index] < second[second_index]: #if the value first is less than second
            first_index = first_index + 1 #increment one to first index
        else:                               # if second is less then
            second_index = second_index + 1  #increment one to second index
    return merged_list       # this is a boolean search and remove duplicates
    """

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
        return merge_postings(indexed_tokens)


def tokenize(text):
    stopWords = set(stopwords.words('english'))
    #return [w for w in re.split("-|\s|,|\.", text) if not in stopwords]
    return [term for term in re.split("-|\s|,|\.", text) if not term in stopWords]  # Stop words split by hyphens or make an array list

    # return text.split(" ")

#log termfrequency * (log (n) / idf)

#term frequency = {Key = token, [list of document_ids] }
#tf[(token, doc_id)] = term_frequency


# Have this function to do term frequency (per documents) and IDF (All documents)
# Term Frequency - Count terms in the document
'''
def add_token_to_index(token, doc_id):
    # add count for number of document ids in current postings
    # make sure when the document id is not in the invert index at that key to increment one
    #inverted_index[token][doc_id] = {}
    count = 0
    if token in inverted_index:
    #inverted_index[token][doc_id] = {}
        current_postings = inverted_index[token]    # updating list of postings (List of document ids)
        #inverted_index[token][doc_id] += 1
        if doc_id not in current_postings:    # If it is already in the list do nothing (helps with dupes)
            current_postings.append(doc_id)
            inverted_index[token] = current_postings
    else:
        inverted_index[token] = [doc_id]    # creating the list of postings, [] means list
        #inverted_index[token][doc_id] = 1
'''

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
    if token in inverted_index:
        if doc_id in inverted_index[token]:
            return inverted_index[token][doc_id]

    #What if the token or doc id isn't there

def sum_of_tf_idf(token, doc_id):
    return sum(term_freq(token, doc_id) * calculate_idf(token))


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

def length_normalize(): #This will take vector
    for token in inverted_index
        sum_of_vectors = (inverted_index)
    length_of_vector = math.sqrt()
#Normalizing the length of a vector
#Divide each of the components by its length


#The query is represented as a weighted tf-idf vector
#Each document as a weighted tf-idf vector
#Compute the Cos Sim Score for the document vector and query vector
#Rank documents with respect to query by score
#Return the top K to the user (Rank the document by the smallest angle)
def cosine_similarity(query):
    #float scores[N] = 0 # score array for all documents for each query terms
    #float Length[N] = 0 # array for the different documents
    for term in query:  # For each term in the query
        current_postings = inverted_index[term]
        for doc_id in inverted_index:
            score += term_freq(token, doc_id) * calculate_idf(token)
    norm = score / len(inverted_index)
    #Sort the documents
        # Calculate the query term and fetch postings list for that query
        # For each document in the postings list each term has a frequency in that document (scale by log)
        # Dot product of the query terms and summing to the scores array
    #Length of each document for each document
    #Divide the score array of each document = Scores[d]/Length[d] (Normalization of the different document
    #Return top K components of Scores (high values for scores)

    #dp = dot_product(first, second)
    return 0 




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
