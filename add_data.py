from pymongo import MongoClient
from face import get_embeddings
import numpy as np
import cv2
import pandas as pd

client = MongoClient('localhost', 27017)
db = client['local']
coll = db['data']


# Function to convert
def listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 = str1+" "+str(ele)
        # return string
    return str1

embedding = get_embeddings(['WIN_20220422_08_10_05_Pro.jpg'])
new_data1 = {"name": "Nguyen Linh Dan", "embedding": str(embedding.tolist())}

embedding = get_embeddings(['WIN_20220425_17_30_06_Pro.jpg'])
new_data2 = {"name": "Tran Van Loi", "embedding": str(embedding.tolist())}
#print(embedding)
#print(embedding.size)
#print(np.fromstring(str(embedding.tolist())[2:-2], sep=', ').size)

coll.insert_one(new_data1)
coll.insert_one(new_data2)

