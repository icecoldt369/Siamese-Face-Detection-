#custom L1 Distance layer module
#required to load the custom model {whenever use custom object (layer, optimiser, loss function)}

#import depesndencies
import tensorflow as tf
import keras.api._v2.keras as keras
from keras.layers import Layer

#custom L1 distance layer
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__() 
    #similarity calculation
    def call(self, input_embedding, validation_embedding): #pass through specific keyword arguments, it will handle them innately
        return tf.math.abs(input_embedding - validation_embedding) 