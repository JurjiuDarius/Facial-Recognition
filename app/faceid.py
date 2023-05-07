from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np


class CamApp(App):

    def build(self):
        try:
            os.chdir('Facial Recognition/app/')
        except Exception:
            print("Invalid Path!")
        self.webcam = Image(size_hint=(1, .8))
        self.button = Button(
            text='Verify', on_press=self.verify, size_hint=(1, .1))
        self.verification_label = Label(
            text="Verification not started", size_hint=(1, .1))

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.webcam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/33.0)

        self.model = tf.keras.models.load_model(
            'siamese_model.h5', custom_objects={'L1Dist': L1Dist})

        self.DETECTION_THRESHOLD = 0.5
        self.VERIFICATION_THRESHOLD = 0.5
        self.VERIFICATION_PATH = os.path.join(
            'application_data', 'verification_images')
        self.INPUT_PATH = os.path.join(
            'application_data', 'input_image', 'input_image.jpg')

        return layout

    def update(self, *args):
        _, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.webcam.texture = img_texture

    def verify(self, *args):

        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        # Saving the last verification attempt
        cv2.imwrite(self.INPUT_PATH, frame)
        results = []
        input_image = self.preprocess(self.INPUT_PATH)
        for image in os.listdir(self.VERIFICATION_PATH):
            validation_img = self.preprocess(
                os.path.join(self.VERIFICATION_PATH, image))

            result = self.model.predict(
                list(np.expand_dims([input_image, validation_img], axis=1)))
            results.append(result)
        detection = np.sum(np.array(results) > self.DETECTION_THRESHOLD)

        verification = detection / len(os.listdir(self.VERIFICATION_PATH))
        verified = verification > self.VERIFICATION_THRESHOLD

        self.verification_label.text = 'Verified' if verified else 'Unverififed'

        return results, verified

    def preprocess(self, file_path):
        byte_img = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(byte_img)
        img = tf.image.resize(img, (100, 100))
        img = img/255.0
        return img


if __name__ == '__main__':
    CamApp().run()
