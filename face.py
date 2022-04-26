from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from deepface import DeepFace
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import cv2
import os
import time

# CHANGE THIS BELOW PATHS TO YOUR OWN DRIVE PATH CONTAINED YOUR FOLDER DATASET
path = r'C:\Users\Admin\PycharmProjects\NhanDienKhuonMat\keras-vggface\image'
os.chdir(path)
# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    s = time.time()
    results = DeepFace.detectFace(pixels, detector_backend="opencv", enforce_detection=False)

    ''''# extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size'''
    image = Image.fromarray(results)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

def extract_face(pixels, required_size=(224, 224)):
    # create the detector, using default weights
    results = DeepFace.detectFace(pixels, detector_backend="opencv", enforce_detection=False)

    ''''# extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size'''
    image = cv2.resize(results,(224,224))
    face_array = asarray(image)
    return face_array

# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
    # extract faces
    faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
    yhat = model.predict(samples)
    return yhat






def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
filenames = ['ajb.jpg']
s = time.time()
embedding = extract_face('WIN_20220422_08_10_05_Pro.jpg')
e = time.time()

print(e-s)