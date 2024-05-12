# Search engine
In this project, we have created a search engine for retrieving text documents based on IR_data_news file in such a way that the user enters his query including NOT(!) operand, multi word("") and of course single word search and the system returns related documents.

## Document preprocessing
Before creating a positional index, it is necessary to preprocess the texts. The necessary steps in this section are as follows.
- Token Extraction
- Text normalization
- Removing  stop words
- Stemming


## Creating positional index
Build the positional index using the preprocessed documents in the previous step. In addition to the location of the words in the documents, in the created positional index, it should be known for each word from the dictionary how many times that word is repeated in all the documents. It should also be clear how many times a specific word is repeated in each document.

## Tf-idf
In the previous step, after extracting the tokens, the information was stored in the form of a dictionary and a positional index. In this section, the aim is to represent the documents in the vector space. Using the tf-idf weighting method, a numerical vector will be calculated for each document, and finally each document will be represented as a vector containing the weights of all the words in that document. The weight of each word t in a document d is calculated using the following equation:

![equation1](https://github.com/MortezaR79/Search-engine/assets/88440848/19511586-5106-47d1-97c6-d0e763ced3fd)


## Retreiving query in vector space
Having the user's question, we extract the vector of the query (we calculate the weight of the words in the query). Then, using the similarity criterion, we try to find the documents that have the most similarity (the least distance) to the input question. Then we display the results in the order of similarity. Different distance criteria can be considered for this task, the simplest of which is the cosine similarity between the vectors, which is defined as follows:

![equation2](https://github.com/MortezaR79/Search-engine/assets/88440848/0325dc78-76e3-4c5d-ac2c-6b9e2918e2a8)

### Index elimination
To increase the speed, you can use the index elimination technique to not calculate the cosine similarity with the documents that will get zero points. At the end of the work, to display a page of query results, it is enough to select K documents that are most similar to the query.

## Increasing the speed of query processing using champion lists

In order to increase the speed of processing and response, you can use champion lists to keep a list of the most relevant documents related to each term in a separate list before a query is raised and during the document processing stage. To implement this section, after constructing the positional inverse index, we create champion list and compare only the query vector with the document vector obtained by searching champion list and We display the related document.
