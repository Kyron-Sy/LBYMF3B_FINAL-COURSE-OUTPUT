from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

from keras.preprocessing.image import img_to_array, load_img
import numpy as np

# -----------------------------------------------------------------------------
# Dimensions of images
# -----------------------------------------------------------------------------
img_width, img_height = 200, 200
input_shape = (img_width, img_height, 3)

# -----------------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------------
test_model = load_model('model.keras')

# -----------------------------------------------------------------------------
# Image data path
# -----------------------------------------------------------------------------
basedir = "data/test/Mix/"

# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------
i=0

def predict(basedir, i, model):
    
    path = basedir+'1 ('+str(i)+').jpg'
    
    img = load_img(path,False,target_size=(img_width,img_height))
        
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
        
    #preds = model.predict_classes(x) #old syntax
    preds = np.argmax(model.predict(x), axis=-1)
    #preds = np.argmax(model.predict(x), axis=-1) #for multi-class classification
    
    #probs = model.predict_proba(x)  #old 
    probs=model.predict(x)
    probs=(probs > 0.5).astype(int)
        
    if preds==0: print('Crazing')
    if preds==1: print('Inclusion')
    if preds==2: print('Patches')
    if preds==3: print('Pitted')
    if preds==4: print('Rolled')
    if preds==5: print('Scratches')

    print ('')
    #print ('Probability: ', probs*100)
    print ('Probability: ', probs)
    
    return preds, probs
 
# ***********************************************    
# MAIN
for i in range (1,101): # Images to be tested

    print('Test Sample: ', i)

    (preds,probs)=predict(basedir, i, test_model) # prediction 

    print(' ')

    
