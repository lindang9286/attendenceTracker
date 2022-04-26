import cv2
from pymongo import MongoClient
from face import is_match
import pandas as pd
import numpy as np
from face import get_embeddings
vid = cv2.VideoCapture(0)

client = MongoClient('localhost', 27017)
db = client['local']
coll = db['data']

data = list(coll.find({}))
df = pd.DataFrame(data).drop(columns=['_id'])
emb = df['embedding'].to_numpy()
np_emb = np.array([])
for ele in emb:
    temp = ""
    temp = np.fromstring(ele[2:-2], sep=', ')
    np_emb = np.append(np_emb,temp)
while (True):

    ret, frame = vid.read()
    frame_vec = get_embeddings(frame)
    for ele in range(len(np_emb)):
        is_match(frame_vec, np_emb[ele])
        print(ele)
    cv2.imshow('frame', frame)
    

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
print(np_emb)
#print(type(np.fromstring(emb[0][2:-2], sep=', ')))