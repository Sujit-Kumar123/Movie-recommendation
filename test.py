import ast
import sklearn
import  nltk
print(sklearn.__version__)
print(nltk.__version__)
'''x=[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]
def converte(x):
    li=[]
    for i in ast.literal_eval(x):
        li.append(i['name'])
    print(li)
converte(x)'''