import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import os
from pymongo import MongoClient
from datetime import datetime
import uuid

import imgto64


model = tf.keras.models.load_model('model.hdf5')

def predict(image_data,model):
    
    size = (128,128)    
    img = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = img.convert('RGB')
    img = np.asarray(img)
    img = (img.astype(np.float32) / 255.0)
    img_reshape = img[np.newaxis,...]

    prediction = model.predict_classes(img_reshape)
    prob = model.predict_proba(img_reshape)

    if prediction[0,0] == 0:
        st.warning("Covid-19")
        st.write("Probability : {:.2f}".format(((1-prob[0,0]))*100),"%")
        st.image(image_data, use_column_width=True)
        return "Covid","{:.2f} %".format((1-prob[0,0])*100)
    else:
        st.info("Normal")
        st.write("Probability : {:.2f}".format((prob[0,0])*100),"%")
        st.image(image_data, use_column_width=True)
        return "Normal","{:.2f} %".format((prob[0,0])*100)

def main():
    st.sidebar.write("""
        # Covid-19 Detection using Deep Learning
        """
        )
    
    st.sidebar.info("Upload the PA view Chest X-ray, and our app will will do the magic 🧙‍♂️")
  
    st.subheader("1) Upload your PA View X-ray 📤")

    st.set_option('deprecation.showfileUploaderEncoding', False)

    file = st.file_uploader("Upload Image here ['jpg', 'jpeg']", type=["jpg", "jpeg"])

    if file is not None:      
        if st.button("Predict"):
            if file is None:
                st.text("You haven't uploaded an image file")
            else:
                image = Image.open(file)
                base64string = imgto64.b64(file)
                status, percent = predict(image,model)
            


    st.subheader("2) Don't have an X-ray? Worry not! Try our app with test images :pick:")
    test_img = st.radio('Select Test Image!', ('Image 1', 'Image 2'))


    if test_img == 'Image 1':
        image = Image.open('img/test.jpeg')
        if st.button("Test"):
            predict(image,model)
        else:
            st.image(image, caption="Test Image 1" ,use_column_width=True)
        
    elif test_img == 'Image 2':
        image = Image.open('img/test1.jpg')
        if st.button("Test"):
            predict(image,model)
        else:
            st.image(image, caption="Test Image 2" ,use_column_width=True)


main()
