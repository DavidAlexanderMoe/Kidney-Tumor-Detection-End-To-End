import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image   # to preprocess the image given in input by the user
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts/training","model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))     # with target size of VGG-16
        test_image = image.img_to_array(test_image)                    # img to numpy array for keras
        test_image = np.expand_dims(test_image, axis = 0)              # add an extra dimension to the array
        # adds a dimension at the beginning (axis=0), effectively converting the shape of the array from 
        # (height, width, channels) to (1, height, width, channels), making it suitable for VGG-16
        result = np.argmax(model.predict(test_image), axis=1)          # argmax for softmax
        print(result)

        if result[0] == 1:
            prediction = 'Tumor'
            return [{ "image" : prediction}]
        else:
            prediction = 'Normal'
            return [{ "image" : prediction}]