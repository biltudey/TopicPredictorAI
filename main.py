# from tensorflow.keras.models import load_model
import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import random
tf.keras.utils.disable_interactive_logging()
st.set_page_config(layout="wide")


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

def model(raw_text):
    
    text = raw_text.split(". ")
    data = preprocess(text)
    st.success('Text is preprocesed.', icon="✅")
    test_dataset = convartTotensor(data)
    st.success('Dataset is created.', icon="✅")
    # st.write(test_dataset)
    st.success('Model is working the data', icon="⏳")
    predict = prediction(test_dataset)
            

    printText(predict,text)

def aboutMe():
    st.subheader("About me")

    me = "### <b>Hi I am Biltu Dey. Currently I am a student. I am doing this project for educational Purpose.</b>"

    st.markdown(me,unsafe_allow_html=True)

    follow = """
    If you liked the project you can follow me on social media.

    [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/BiltuDey/)
    [![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/mr_biltu)
    

    ### check [![github](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://twitter.com/mr_biltu) account for outher project.

    ## Support
    Any kind of help or facing any problem feel free to contact me.

    For support, email biltudey222@gmail.com.
    """
    st.markdown(follow,unsafe_allow_html=True)



def main():
    """
        This is main Function
    
    """
    st.title("TopicPredictorAI")
    try:
        font_css = """
        <style>
        button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 24px;
        }
        </style>
        """

        st.write(font_css, unsafe_allow_html=True)
        button_style = """
            <style>
            .stButton > button {
                color: white;
                background: black;
                width: 100px;
                height: 50px;
                font: 50px;
            }
            </style>


            """
        st.markdown(button_style, unsafe_allow_html=True)

        tab1, tab2,  tab3 = st.tabs(["AI","About Me" ,"How to use"])

        with tab1:
            st.subheader("Enter The Text")
            raw_text = st.text_area(" ",value='', height=None, max_chars=None, key=None)
            
            if st.button("Skim"):
                if raw_text.strip() == "":
                    st.error("Please enter something")
                else:
                    model(raw_text)
        with tab2:
            aboutMe()
        with tab3:
            text = """
                        ### **Enter your abstract text data.**
                        #### Something like this.
                """
            st.markdown(text,unsafe_allow_html=True)
            st.image('screenshot\Screenshot.png')
            text = """
                        ### **Then click the skim button.**
                        #### Wait while text is processed by the model.
                        #### After few second your data will look like this.
                """
            st.markdown(text,unsafe_allow_html=True)

            st.image('screenshot\Screenshot1.png')


        

        
    except:
        st.error("There is an error. Please check your text data.")

        # st.write(predict)

if __name__ == '__main__':
    main()