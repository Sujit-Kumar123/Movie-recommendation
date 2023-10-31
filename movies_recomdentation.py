import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import pickle

"""
                                                          DATA CLEANING
"""
movies=pd.read_csv("tmdb_5000_movies.csv")
# Complete information of movies
# print(movies.info())
cradits=pd.read_csv("tmdb_5000_credits.csv")
# Complete information of cradits
# print(cradits.info())
# Joining of movies with cradits on the basis of title
movies=movies.merge(cradits,on='title')
"""Shape of joined data set
    print(movies.shape)
    Complete information of joined data set
    print(movies.info())
    Counting of languages of movies
    print(movies['original_language'].value_counts())
"""
"""Choosing of fields from the mearged data set i.e
    movie_id,title,overview,genres,keywords,cast,crew.
"""
movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]
# print(movies.head(1))
# To check null valuse are present in the in the data set
#print(movies.isnull().sum())
# Removing of null values from data set
movies.dropna(inplace=True)
# print(movies.isnull().sum())
# Remove duplicate data
# print(movies.duplicated().sum())
'''Examine of genres column value 
obj=movies.iloc[0].genres
print(obj)'''
# Converting the genres into a list with name values only
# Converter function
def converte(x):
    li=[]
    for i in ast.literal_eval(x):
        li.append(i['name'])
    return(li)
# Converting column movies[genres] into movies[genres] list
movies['genres']=movies['genres'].apply(converte)
#print(movies['genres'])
#print(movies.head(1))
#print(movies.iloc[0].keywords)
# Converting column movies[Keywords] into movies[Keywords] list
movies['keywords']=movies['keywords'].apply(converte)
#print(movies.iloc[0].keywords)
'''Extracting  first three cast name from movies[cast] the function is defined as'''
def convert_cast(x):
    li=[]
    counter=0
    for i in ast.literal_eval(x):
        if counter!=3:
            li.append(i['name'])
            counter +=1
        else:
            break
    return li
movies['cast']=movies['cast'].apply(convert_cast)
#print(movies.iloc[1].cast)
""" Fetching of director name from the crew column the function is given by"""
def fetching_directr(x):
    li=[]
    for i in ast.literal_eval(x):
        if i['job']=="Director":
            li.append(i['name'])
            break
    return li
movies['crew']=movies['crew'].apply(fetching_directr)
# print(movies.iloc[1].crew)
#print(movies.crew)
""" Overview converted string into list"""
movies['overview']=movies['overview'].apply(lambda x:x.split())
#print(movies.iloc[1].overview)
#print(movies.overview)
""" Removing all spaces from genres,keywords,cost,crew"""
movies['genres']=movies['genres'].apply(lambda  x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda  x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda  x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda  x:[i.replace(" ","") for i in x])
'''print(movies.genres)
print(movies.keywords)
print(movies.cast)
print(movies.crew)'''
""" Creating new column on the name [tags] which contain [overview,genres,keywords,cast,crew] """
movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
#print(movies)
""" Creating new dataframe which contain movies[movie_id,title,tags]"""
new_df=movies[['movie_id','title','tags']].copy()
#print(new_df)
""" Converting new_df[tags} in to string"""
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
#print(new_df.tags)
""" Converting new_df[tags} string in to lower case"""
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
#print(new_df.tags)
"""
                                                    MACHINE LEARNING CODES
"""
# Vectorization of tags using sklearn CountVectorization
cv=CountVectorizer(max_features=5000,stop_words='english')
# To get 5000 words which are taken to consider
vectros=cv.fit_transform(new_df['tags']).toarray()
#print(cv.get_feature_names())
#print(vectros)
#print(vectros[0])
""" Removal of similar words by using nltk(Natural language tool kit) steamings
 Ex- actions-action,actioned-action
 """
ps=PorterStemmer()
def stem(text):
    li=[]
    for i in text.split():
        li.append(ps.stem(i))
    return " ".join(li)
new_df['tags']=new_df['tags'].apply(stem)
#print(new_df['tags'][0])
"""
Now we need calculate the distance between vectors by using cosine distance
Using cosine distance from sklearn we find the cosine similarity
"""
similarity=cosine_similarity(vectros)
#print(similarity[0])
#print(sorted(similarity[0]))
#print(sorted(similarity[0],reverse=True))
""" To find index of the movies"""
#print(new_df[new_df['title']=='Avatar'])
#print(new_df[new_df['title']=='Avatar'].index[0])
# To find top FIVE similar movies
#print(list(enumerate(similarity[0])))
#print(sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1]))
def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distance=similarity[movie_index]
    movi_list=sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movi_list:
        #print(i[0])
        print(new_df.iloc[i[0]].title)
recommend("Batman Begins")
""" Dumping of new_df """
'''pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))'''