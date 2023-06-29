#import kivy dependencies first
from kivy.app import App  #base app class
from kivy.uix.boxlayout import BoxLayout #box layer imported

#import kivy asset classes UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

#import other kivy stuff
from kivy.clock import Clock #get read time feed from our webcam / kivy app
from kivy.graphics.texture import Texture #convert our image from webcam into a texture
from kivy.logger import Logger

#import other dependencies
import cv2 #access webcam
import tensorflow as tf 
from layers import L1Dist #l1 layer distance from layers file
import os #file path
import numpy as np

#build app and layout

class CamApp(App):

    #build is a inherent function when using KIVY
    def build(self): 
        #main layout components
        #three verification levels
        self.web_cam = Image(size_hint=(1,.8)) #our main image from Image class + have keyword item 1, 80% of veritical height
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1,.1)) #size of our button 
        self.verification_label = Label(text="Verification Uninitated", size_hint=(1,.1)) #

        #add items to layout to add the objects created above to actual kivy layout
        layout = BoxLayout(orientation='vertical') #or horizontal
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button) # widget follows order strictly - sequential order
        layout.add_widget(self.verification_label)

        #load keras model 
        #load from the siamese model file in thedolfer
        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist': L1Dist})

        #setup video capture device which connect to webcam similar to open CV
        self.capture = cv2.VideoCapture(0)

        #run update below on this interval
        #specified the frequency function runs 33 times every second
        Clock.schedule_interval(self.update, 1.0/33.0) 

        return layout
    
    #run continuously to get webcam feed #runs 1 over 33 seconds
    def update(self, *args):
        
        #read frame from opencv
        ret, frame = self.capture.read()
        frame = frame[120:120+250,200:200+250, :]

        #flip horizontal and convert image to texture
        #convert opencv buffer by flipping horizontally 
        buf = cv2.flip(frame, 0).tostring() 
        #convert opencv buff array to texture
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
        #pass string value of the image to blitbuffer for rendering
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        #set the texture of webcam  equal to texture
        self.web_cam.texture = img_texture

    #load image from file and convert to 100x100 px
    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path) #reads the files
        img = tf.io.decode_jpeg(byte_img) #opens the jpg images
        img = tf.image.resize(img, (100,100)) #resize the image by 100*100
        img = img/255.0
        return img

    #verification function to verify person 
    def verify(self, *args):
        #specify threshold
        #the limit is before our prediction is considered positive
        detection_threshold = 0.5
        #what proposition of predictions need to be positive for a match
        verification_threshold = 0.5

        #capture input image from our webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250,200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
            
            # Make Predictions 
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        # Detection Threshold: Metric above which a prediciton is considered positive 
        detection = np.sum(np.array(results) > detection_threshold)
        
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification > verification_threshold
        
        #set verification text
        self.verification_label.text = 'verified' if verified == True else 'unverified'


        #log out details 
        Logger.info(results)
        Logger.info(detection)
        Logger.info(verification)
        Logger.info(verified)

        #want to find the array of results that are passing 50% and sum it
        """
        Logger.info(np.sum(np.array(results)> 0.5)) 
        Logger.info(np.sum(np.array(results)> 0.2)) 
        Logger.info(np.sum(np.array(results)> 0.4)) 
        Logger.info(np.sum(np.array(results)> 0.8)) 
        """
        
        return results, verified

    #bring over verification function
if __name__ == '__main__':
    CamApp().run()