import os
import re
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math
from collections import defaultdict
import time
import json


directory = r"C:\Users\PCS\Desktop\k16-3721\ShortStories"
directorydict= r"C:\Users\PCS\Desktop\k16-3721\dictionary.txt"
directoryidf= r"C:\Users\PCS\Desktop\k16-3721\idf.txt"
directorytf= r"C:\Users\PCS\Desktop\k16-3721\tf.txt"
directoryvsm= r"C:\Users\PCS\Desktop\k16-3721\vsm.txt"      
path = r"C:/Users/PCS/Desktop/k16-3721/dictionary.txt"
pathidf = r"C:/Users/PCS/Desktop/k16-3721/idf.txt"
pathtf = r"C:/Users/PCS/Desktop/k16-3721/tf.txt"
pathvsm = r"C:/Users/PCS/Desktop/k16-3721/vsm.txt"


#storing stopwords file in dictionary 
#with open('list.txt','r') as readl:
 #     stoplist = readl.read() 

#file = open('list.txt', 'r') 
#print (file.read())
stoplist = dict()

stoplist = "a is the of all and to can be as once for at am are has have had up his her in on no we do"

stoplist=nltk.word_tokenize(stoplist)




x=1
dictionary = {}

for filename in (os.listdir(directory)):
    #print(filename)
    f = os.fsdecode(filename)
    if f.endswith(".txt"):
        story = open(r'C:\Users\PCS\Desktop\k16-3721\ShortStories\\' + filename)
        stripslash = '.txt'
        filename = ''.join(char for char in filename if char not in stripslash)
        c=int(filename)
        x=c
        dictionary[c]= story.read()
        #print(c , dictionary[c])
        dictionary[c]= dictionary[c].lower()
            
        #removing stop words
        
    for k in range(len(stoplist)):
        r = stoplist[k]
    # print(r)
    dictionary[x] = re.sub( r"\s+" + stoplist[k] + "\s"," ",dictionary[x] )
   
        #removing all unnecessary punctutations , keeping the word clean
   
    dictionary[x] = re.sub(r"'","",dictionary[x] )
    dictionary[x] = re.sub(r"_"," ",dictionary[x] )
    dictionary[x] = re.sub(r";","",dictionary[x] )
    dictionary[x] = re.sub(r"\?","",dictionary[x] )
    dictionary[x] = re.sub(r","," ",dictionary[x] )
    
    dictionary[x] = re.sub(r"\W"," ",dictionary[x] )
    dictionary[x] = re.sub(r"\d"," ",dictionary[x] )
    dictionary[x] = re.sub(r"\s+"," ",dictionary[x])
    
    
with open(path, 'w') as outfile:
    json.dump(dictionary, outfile)

        


    
#######x-x-x-x-x-x-x-################
alld = 50 

##to remove stop words

positionalindex ={}
for i in range(1,51):        
    positionalindex[i] = nltk.word_tokenize(dictionary[i])

        

index = defaultdict(list)
a=1
s=0
for keys,values in sorted(positionalindex.items()):
    for value in values:
        if value not in stoplist:
            if value not in index:
                index[keys].append(value)               


positionalindex = index
x=50
        
        

##CREATING A TERM FREQUENCY MATRIX
termfrequency = defaultdict(list)
for i in range(1,51):
    for terms in positionalindex[i]:
        #print(terms)
        if terms not in termfrequency.keys():
            termfrequency[terms] = 1
            
        else:
            termfrequency[terms] += 1


###x-x-x-x#######
uniqueterm = {} 
r=1
for term in termfrequency:
    uniqueterm[r]=term
    r +=1


##DETERMINING idf MATRIX VALUE

idf={}
alld=50
docfre = {}

if os.stat(directoryidf).st_size == 0:

    for term in uniqueterm.values(): 
        doc=0
        for i in range(1,51):
            if term in positionalindex[i]: 
               # print("term" , term )
                doc +=1
        docfre[term]=doc
        if doc > 0 :
           # print("term" , term , ": ",alld,"**", docfre[term], "alld/ docfre[term]", alld/ docfre[term] )
           idf[term] = math.log10(alld/ docfre[term])
        
    with open(pathidf, 'w') as outfile:
        json.dump(idf, outfile)
else:
    with open(pathidf, 'r') as f:
        idf = json.load(f)  
        
        
        
       # print(term , ":",idf[term], "*" , alld , "/" , doc ,"  -",np.log((x-1)/(doc)))


##DETERMINING TF MATRIX
## tf = no of occurrence of term in document 

      
tf = {}
if os.stat(directorytf).st_size == 0:
    x = 0

    for term in idf: 
        i=1
        tfval=[]
        for i in range(1,51):
            # print("i : " ,  i)
            tff = 0
            # print("document" , document)
            for word in positionalindex[i]:
                if term == word:
                    tff +=1
            #print(tff, "CCCC@@@@@@@: " ,len(nltk.word_tokenize(document))) 
             
            #tf_val = tff/len(positionalindex[i])
            tfval.append(tff)
        
            tf[term]=tfval   
 
    with open(pathtf, 'w') as outfile:
        json.dump(tf, outfile)
else:
    with open(pathtf, 'r') as f:
        tf = json.load(f)        
      
      
    
    
    
###TF*IDF
tfidf={}

if os.stat(directoryvsm).st_size == 0:

    for tfterm in tf.keys():
        #print("tfterm: ",tfterm)
        tf_idf=[]
        #print("idfterm:    ",idfterm)
        for term in tf[tfterm]:
            #print("term :   ",term   ,"::", tfterm)
            tidf= term * idf[tfterm]
            tf_idf.append(tidf)
           
        tfidf[tfterm]=tf_idf    
        

      
    vsm = []
    g=1    
    for i in range(0,50):   
        vsm_m = []
        for term in idf:
            vsm_m.append(tfidf[term][i]) 
        vsm.append(vsm_m)
        g +=1
    
    
    with open(pathvsm, 'w') as outfile:
        json.dump(vsm, outfile)
else:
    with open(pathvsm, 'r') as f:
        vsm = json.load(f)    
        

     
        
#####################X_X_X_X_X_X_X_X_X_##########################      

while (1):
      
    an = input("Do you want to enter query or exit: \n Press any key- Enter Query \n 0- Exit      \n")        
    
    if an != '0':

        query = input("Enter Query:    ")
        start = time. time()
        pquery=query
        pquery=nltk.word_tokenize(pquery) 

        qtf = {}
        for term in idf: 
            i=1
            tfval=[]
    
            tff = 0
            # print("document" , document)
            for word in pquery:
                if term == word:
                    tff +=1
                #print(tff, ": " ,len(nltk.word_tokenize(document)))                 
            qtf[term]=tff   
        
        ###TF*IDF
        qtfidf={}
        for term in qtf.keys():
            #print("term: ",term,qtf[term])
            qtfidf[term]=qtf[term] * idf[term]
              
        qvsm = []
        g=1
        vsm_m = []
    
        for term in qtfidf: 
            vsm_m.append(qtfidf[term]) 
        qvsm.append(vsm_m)
   
    
        ##--------DETERMINING COSINE
        finalanswer = {}


        for i in range(0,50):
            #print(vsm[i])
            dotproduct = np.dot((vsm[i]), qvsm[0])
            normterm = np.linalg.norm(vsm[i])
            normqvsm = np.linalg.norm(qvsm)
            cosine = dotproduct / (normterm * normqvsm)
            finalanswer[i]=cosine    
    

        alpha = 0.005 
        print("Query: ",query)
        for i in finalanswer:
            if finalanswer[i] > alpha:
                print("Document: " , i+1,"->", finalanswer[i])
        
        end = time. time()
        print("The total process by query is ",end - start ," seconds.")        

    else:
        print("\nYou typed 0 to exit")
        break


    
    
        
             
          

