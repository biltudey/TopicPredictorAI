# from tensorflow.keras.models import load_model
import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import random
import warnings
warnings.filterwarnings('ignore')


target = ['BACKGROUND','CONCLUSIONS', 'METHODS', 'OBJECTIVE','RESULTS']
def preprocess(text):
    df = pd.DataFrame(text,columns=['text'])
    df['line_number'] = df.index
    df['total_lines'] = len(text)
    return df

def split_chars(text):
  return " ".join(list(text))

def convartTotensor(df):
    label = np.array(random.choices(target,k=len(df)))
    line_number_one_hot = tf.one_hot(df['line_number'].to_numpy(),depth=15)
    total_lines_one_hot =  tf.one_hot(df['total_lines'].to_numpy(),depth=20)
    test_sentence = df['text'].tolist()
    test_sentence = [i.strip() for i in test_sentence]
    test_char =[split_chars(sentence) for sentence in test_sentence]

    one_hot_encoder = OneHotEncoder(sparse=False)
    label_one_hot = one_hot_encoder.fit_transform(label.reshape(-1, 1))
    test_dataset = tf.data.Dataset.from_tensor_slices((line_number_one_hot,
                                           total_lines_one_hot,
                                           test_sentence,
                                           test_char))

    random_labels = tf.data.Dataset.from_tensor_slices(label_one_hot)

    test_dataset = tf.data.Dataset.zip((test_dataset,random_labels))
    test_dataset=test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

    return test_dataset

    
def prediction(test_dataset):
    model = tf.keras.models.load_model("model")
    prediction = model.predict(test_dataset)
    predict_label = tf.argmax(prediction,axis=1)
    pred_label = predict_label.numpy().tolist()
    return pred_label

def printText(pred_label,test_sentence):
    BACKGROUND = " "
    CONCLUSIONS = " "
    METHODS = " "
    OBJECTIVE = " "
    RESULTS = " "
    l = len(test_sentence)
    for i in range(l):
        if pred_label[i] == 0:
            BACKGROUND +="\n"+ test_sentence[i]

        elif pred_label[i] == 1:
            CONCLUSIONS +="\n"+  test_sentence[i]

        elif pred_label[i] == 2:
            METHODS +="\n"+  test_sentence[i]

        elif pred_label[i] == 3:
            OBJECTIVE +="\n"+  test_sentence[i]

        elif pred_label[i] == 4:
            RESULTS +="\n"+  test_sentence[i]
    if BACKGROUND != " ": 
        st.write(f"**Background** :\n {BACKGROUND}\n")
    if OBJECTIVE !=" ":
        st.write(f"**OBJECTIVE** :\n {OBJECTIVE}\n")
    if METHODS !=" ":
        st.write(f"**METHODS** :\n {METHODS}\n")
    if RESULTS !=" ":
        st.write(f"**RESULTS** :\n {RESULTS}\n")
    if CONCLUSIONS != " ":
        st.write(f"**CONCLUSIONS** :\n {CONCLUSIONS}\n")




def main():
    st.title("Skimming the text")
    st.subheader("Enter The Text")
    raw_text = st.text_area(" ",)
        
    if st.button("Skim"):
        text = raw_text.split(". ")

        data = preprocess(text)
        st.success('Text is preprocesed.', icon="✅")
        test_dataset = convartTotensor(data)
        st.success('Dataset is created.', icon="✅")
        # st.write(test_dataset)
        st.success('Model is working the data', icon="⏳")
        predict = prediction(test_dataset)
        

        printText(predict,text)

        # st.write(predict)

if __name__ == '__main__':
    main()