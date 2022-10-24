

import pandas as pd

import streamlit as st
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
import sqlite3
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.cluster import KMeans
import re
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from wordcloud import WordCloud










st.set_page_config(

    page_title="EREPUTATION MODULE",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",

)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)





@st.cache(allow_output_mutation=True)
def tokenization(reviews_list):
    reviews_list = [word_tokenize(i) for i in reviews_list]
    return reviews_list

@st.cache(allow_output_mutation=True)
def remove_symbols(review_list):
    reviews = [replace_with_space.sub(" ", i.lower()) for i in review_list]
    return reviews
@st.cache(allow_output_mutation=True)
def remove_punctuation(review_list):
    reviews = [remove.sub("", i.lower() ) for i in review_list]
    return reviews
@st.cache(allow_output_mutation=True)
def remove_stopword(review_list):
    reviews_list_stopword = []
    for i in review_list:
        reviews_list_stopword.append([word for word in i if word not in stopwords.words('english')])
    return reviews_list_stopword

@st.cache(allow_output_mutation=True)
def reviews_wordcloud(reviews_list_stopword):
    reviews_wordcloud = ""
    for i in reviews_list_stopword:
        for ii in i:
            ispace = ii + " "
            reviews_wordcloud += ispace
    return reviews_wordcloud



@st.cache(allow_output_mutation=True)
def load_data():
    conn = sqlite3.connect('Database-reviews-portobello.db')
    c = conn.cursor()
    c.execute('''  
    SELECT * FROM reviews
              ''')
    df = pd.DataFrame(c.fetchall(), columns=['Nom', 'Dates', 'Title_Review', 'Score_reviews', 'text', 'label', 'score',
                                              'isTranslated', 'textInEnglish', 'sentiment', 'autorating', 'Topic', 'Name'])
    return df

@st.cache(allow_output_mutation=True)
def load_bertopic_model(df):
    cluster_model = KMeans(n_clusters=6)
    # docs = df[df["Score_reviews"]<4]['textInEnglish'].tolist()
    docs = df['textInEnglish']
    embedding_model = "all-MiniLM-L6-v2"
    hdbscan_model = HDBSCAN(min_cluster_size=30, metric='euclidean',
                            cluster_selection_method='eom', prediction_data=True, min_samples=5)

    umap_model = UMAP(n_neighbors=30, n_components=5,
                      min_dist=0.0, metric='cosine')

    vectorizer_model = CountVectorizer(stop_words="english")
    topic_model = BERTopic(vectorizer_model=vectorizer_model, nr_topics=5, hdbscan_model=cluster_model,
                           calculate_probabilities=True, embedding_model=embedding_model)  # diversity=0.2
    topics, probs = topic_model.fit_transform(docs)
    return topic_model,topics

@st.cache(allow_output_mutation=True)
def liste_textInEnglish_def(df):
    liste_textInEnglish=[]
    for i in df['textInEnglish']:
        liste_textInEnglish.append(i)
    return liste_textInEnglish

@st.cache(allow_output_mutation=True)
def to_float(x):
    return float(x)



@st.cache
def get_all_data():
    root = "Datasets/"
    with open(root + "imdb_labelled.txt", "r") as text_file:
        data = text_file.read().split('\n')

    with open(root + "amazon_cells_labelled.txt", "r") as text_file:
        data += text_file.read().split('\n')

    with open(root + "yelp_labelled.txt", "r") as text_file:
        data += text_file.read().split('\n')

    return data


@st.cache
def preprocessing_data(data):
    processing_data = []
    for single_data in data:
        if len(single_data.split("\t")) == 2 and single_data.split("\t")[1] != "":
            processing_data.append(single_data.split("\t"))

    return processing_data





@st.cache
def split_data(data):
    total = len(data)
    training_ratio = 0.75
    training_data = []
    evaluation_data = []

    for indice in range(0, total):
        if indice < total * training_ratio:
            training_data.append(data[indice])
        else:
            evaluation_data.append(data[indice])

    return training_data, evaluation_data

@st.cache
def altair_histogram():
    brushed = alt.selection_interval(encodings=["x"], name="brushed")

    return (
        alt.Chart(hist_data)
            .mark_bar()
            .encode(alt.X("x:Q", bin=True), y="count()")
            .add_selection(brushed)
    )
@st.cache
def preprocessing_step():
    data = get_all_data()
    processing_data = preprocessing_data(data)
    return split_data(processing_data)

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

@st.cache
def reviews_wordcloud_f_f(df):
    reviews = liste_textInEnglish_def(df)
    # The regex code for the symbols and punctuation to be removed. We will simply remove the punctuation but # will replace the set of symbols with a space so we don'tjoin two separate words together!

    reviews = remove_symbols(reviews)
    reviews = remove_punctuation(reviews)
    reviews_token = tokenization(reviews)
    reviews_list_stopword = remove_stopword(reviews_token)
    reviews_wordcloud_f = reviews_wordcloud(reviews_list_stopword)
    return reviews_wordcloud_f


def training_step(data, vectorizer):
    training_text = [data[0] for data in data]
    training_result = [data[1] for data in data]
    training_text = vectorizer.fit_transform(training_text)

    return BernoulliNB().fit(training_text, training_result)

def analyse_text(classifier, vectorizer, text):
    return text, classifier.predict(vectorizer.transform([text]))


def print_result(result):
    text, analysis_result = result
    print_text = "Positive" if analysis_result[0] == '1' else "Negative"
    return text, print_text


def moyenne_glissante(valeurs, intervalle):
    indice_debut = (intervalle - 1) // 2
    liste_moyennes = [sum(valeurs[i - indice_debut:i + indice_debut + 1]) / intervalle for i in
                      range(indice_debut, len(valeurs) - indice_debut)]
    return liste_moyennes

def display_mean_glissante():
    valeurs = df["stars"]
    intervalle = 5

    print(valeurs)
    print(moyenne_glissante(valeurs, intervalle))
    return None

def to_float_(x):
    return float(x)

compteur = 0
replace_with_space = re.compile("(<br\s*/><br\s*/>) 1(\-)1(V)1(\n*V)I(\r*\n)1(#&)")
remove = re.compile("[.;:!\'?,\"()\[\]]")

selection_mode = st.sidebar.radio("Sélectionnez votre mode",('DashBoard','More exploration','Sentiment Analysis Module',"Download Scraped Dataframe in CSV"))

if selection_mode == "DashBoard":
    col1,col2 = st.columns(2)
    with col1:

        st.image("./Dashboard-pnj-save/dashboard.png")
        st.write("Col1")
        labels = 'Coffee', 'Brisket', 'Lunch', 'Sandwiches',"Good day"
        sizes = [33, 23, 16, 10, 16]
        explode = (0, 0.1, 0, 0,0)  # only "explode" the 2nd slice

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.pyplot(fig1)
    with col2:

        st.image("./Dashboard-pnj-save/dashboard1.png")
        st.write("")
        df = load_data()
        st.subheader("Latest reviews :")
        #st.dataframe(df)
        for i in range(2):

            st.write("Name of reviewer: ",df["Nom"].iloc[i])
            st.write(df["Dates"].iloc[i])
            st.write("Title",df["Title_Review"].iloc[i])
            st.write(int(float(df["Score_reviews"].iloc[i]))*':star:',df["Score_reviews"].iloc[i])

            st.write("Reviews :",df["text"].iloc[i])
            st.write("_____________")
        agree = st.checkbox('View More Reviews',key=compteur)
        compteur +=1

        if agree:
            st.write('Great!')
            for i in range(2,10):
                st.write("Name of reviewer: ", df["Nom"].iloc[i])
                st.write(df["Dates"].iloc[i])
                st.write("Title", df["Title_Review"].iloc[i])
                st.write(int(df["Score_reviews"].iloc[i]) * ':star:', df["Score_reviews"].iloc[i])

                st.write("Reviews :", df["text"].iloc[i])
                st.write("_____________")








if selection_mode == "More exploration":
    df = load_data()
    st.dataframe(df)
    df['Score_reviews'] = df['Score_reviews'].apply(to_float_)
    score_means = df['Score_reviews'].mean()
    st.write("Means=",score_means)
    hist_data = df['label'].value_counts().to_frame()
    st.subheader("Repartition Negative and Positive Review")
    st.bar_chart(hist_data)

    genre = st.radio(
        "See positive or negative review",
        ('Positive', 'Negative score < 3', 'Negative Sentiment'))


    if genre == 'Positive':
        st.write(f'You selected {genre}.')
        df = load_data()
        df = df[df["label"] == 'POSITIVE']
        for i in range(2):
            st.write("Name of reviewer: ", df["Nom"].iloc[i])
            st.write(df["Dates"].iloc[i])
            st.write("Title", df["Title_Review"].iloc[i])
            st.write(int(df["Score_reviews"].iloc[i]) * ':star:', df["Score_reviews"].iloc[i])

            st.write("Reviews :", df["text"].iloc[i])
            st.write("_____________")
        agree = st.checkbox('View More Reviews',key=compteur)
        compteur += 1

        if agree:
            st.write('Great!')
            for i in range(2, 10):
                st.write("Name of reviewer: ", df["Nom"].iloc[i])
                st.write(df["Dates"].iloc[i])
                st.write("Title", df["Title_Review"].iloc[i])
                st.write(int(df["Score_reviews"].iloc[i]) * ':star:', df["Score_reviews"].iloc[i])

                st.write("Reviews :", df["text"].iloc[i])
                st.write("_____________")


    if genre == 'Negative score < 3':
        st.write(f'You selected {genre}.')
        df = load_data()
        df['Score_reviews'] = df['Score_reviews'].apply(to_float_)
        df = df[df["Score_reviews"] <= 3]

        for i in range(2):
            st.write("Name of reviewer: ", df["Nom"].iloc[i])
            st.write(df["Dates"].iloc[i])
            st.write("Title", df["Title_Review"].iloc[i])
            st.write(int(df["Score_reviews"].iloc[i]) * ':star:', df["Score_reviews"].iloc[i])

            st.write("Reviews :", df["text"].iloc[i])
            st.write("_____________")
        agree = st.checkbox('View More Reviews',key=compteur)
        compteur += 1

        if agree:
            st.write('Great!')
            for i in range(2, 10):
                st.write("Name of reviewer: ", df["Nom"].iloc[i])
                st.write(df["Dates"].iloc[i])
                st.write("Title", df["Title_Review"].iloc[i])
                st.write(int(df["Score_reviews"].iloc[i]) * ':star:', df["Score_reviews"].iloc[i])

                st.write("Reviews :", df["text"].iloc[i])
                st.write("_____________")

    if genre == 'Negative Sentiment':
        st.write(f'You selected {genre}.')
        df = load_data()
        df = df[df["label"] == 'NEGATIVE']

        for i in range(2):
            st.write("Name of reviewer: ", df["Nom"].iloc[i])
            st.write(df["Dates"].iloc[i])
            st.write("Title", df["Title_Review"].iloc[i])
            st.write(int(df["Score_reviews"].iloc[i]) * ':star:', df["Score_reviews"].iloc[i])

            st.write("Reviews :", df["text"].iloc[i])
            st.write("_____________")
        agree = st.checkbox('View More Reviews',key=compteur)
        compteur += 1

        if agree:
            st.write('Great!')
            for i in range(2, 10):
                st.write("Name of reviewer: ", df["Nom"].iloc[i])
                st.write(df["Dates"].iloc[i])
                st.write("Title", df["Title_Review"].iloc[i])
                st.write(int(df["Score_reviews"].iloc[i]) * ':star:', df["Score_reviews"].iloc[i])

                st.write("Reviews :", df["text"].iloc[i])
                st.write("_____________")
    st.write("____________________________________________")
    st.subheader("Topic Labelling PART")
    agree_bertopic = st.checkbox('Click Here for the Topic Labelling Part',key=compteur)
    compteur+=1

    if agree_bertopic:
        df=load_data()
        df['Score_reviews'] = df['Score_reviews'].apply(to_float_)
        topic_model,topics = load_bertopic_model(df)

        fig = topic_model.visualize_topics()
        st.plotly_chart(fig, use_container_width=True)
        fig2 = topic_model.visualize_barchart()

        st.plotly_chart(fig2, use_container_width=True)

        df['Score_reviews'] = df['Score_reviews'].apply(to_float)

        reviews_wordcloud_f = reviews_wordcloud_f_f(df)

        option = st.selectbox(
            'Inspect Topics',
            ('Coffee', 'Brisket', 'Lunch', 'Sandwiches',"Good day"))


        st.write('You selected:', option)
        if option == 'Coffee':
            col1,col2 =st.columns(2)
            df = load_data()
            df = df[df["Topic"] == 0]
            with col1:
                for i in range(2):
                    st.write("Name of reviewer: ", df["Nom"].iloc[i])
                    st.write(df["Dates"].iloc[i])
                    st.write("Title", df["Title_Review"].iloc[i])
                    st.write(int(df["Score_reviews"].iloc[i]) * ':star:', df["Score_reviews"].iloc[i])

                    st.write("Reviews :", df["text"].iloc[i])
                    st.write("_____________")
                agree = st.checkbox('View More Reviews',key=compteur)
                compteur += 1

                if agree:
                    st.write('Great!')
                    for i in range(2, 10):
                        st.write("Name of reviewer: ", df["Nom"].iloc[i])
                        st.write(df["Dates"].iloc[i])
                        st.write("Title", df["Title_Review"].iloc[i])
                        st.write(int(df["Score_reviews"].iloc[i]) * ':star:', df["Score_reviews"].iloc[i])

                        st.write("Reviews :", df["text"].iloc[i])
                        st.write("_____________")



            with col2:
                st.subheader(f"WordCloud Reviews TOPIC- {option} ")
                wordcloud = WordCloud(background_color='white', max_words=50).generate(reviews_wordcloud_f_f(df))
                plt.imshow(wordcloud)
                plt.axis("off")
                st.pyplot(fig=plt)



        if option == 'Brisket':
            df = load_data()
            df = df[df["Topic"] == 1]
            col1, col2 = st.columns(2)
            with col1:
                for i in range(2):
                    st.write("Name of reviewer: ", df["Nom"].iloc[i])
                    st.write(df["Dates"].iloc[i])
                    st.write("Title", df["Title_Review"].iloc[i])
                    st.write(int(df["Score_reviews"].iloc[i]) * ':star:', df["Score_reviews"].iloc[i])

                    st.write("Reviews :", df["text"].iloc[i])
                    st.write("_____________")
                agree = st.checkbox('View More Reviews',key=compteur)
                compteur += 1

                if agree:
                    st.write('Great!')
                    for i in range(2, 10):
                        st.write("Name of reviewer: ", df["Nom"].iloc[i])
                        st.write(df["Dates"].iloc[i])
                        st.write("Title", df["Title_Review"].iloc[i])
                        st.write(int(df["Score_reviews"].iloc[i]) * ':star:', df["Score_reviews"].iloc[i])

                        st.write("Reviews :", df["text"].iloc[i])
                        st.write("_____________")

            with col2:
                st.subheader(f"WordCloud Reviews TOPIC- {option} ")
                wordcloud = WordCloud(background_color='white', max_words=50).generate(reviews_wordcloud_f_f(df))
                plt.imshow(wordcloud)
                plt.axis("off")
                st.pyplot(fig=plt)

        if option == 'Lunch':
            df = load_data()
            df = df[df["Topic"] == 2]
            col1, col2 = st.columns(2)
            with col1:
                for i in range(2):
                    st.write("Name of reviewer: ", df["Nom"].iloc[i])
                    st.write(df["Dates"].iloc[i])
                    st.write("Title", df["Title_Review"].iloc[i])
                    st.write(int(df["Score_reviews"].iloc[i]) * ':star:', df["Score_reviews"].iloc[i])

                    st.write("Reviews :", df["text"].iloc[i])
                    st.write("_____________")
                agree = st.checkbox('View More Reviews',key=compteur)
                compteur += 1

                if agree:
                    st.write('Great!')
                    for i in range(2, 10):
                        st.write("Name of reviewer: ", df["Nom"].iloc[i])
                        st.write(df["Dates"].iloc[i])
                        st.write("Title", df["Title_Review"].iloc[i])
                        st.write(int(df["Score_reviews"].iloc[i]) * ':star:', df["Score_reviews"].iloc[i])

                        st.write("Reviews :", df["text"].iloc[i])
                        st.write("_____________")
            with col2:
                st.subheader(f"WordCloud Reviews TOPIC- {option} ")
                wordcloud = WordCloud(background_color='white', max_words=50).generate(reviews_wordcloud_f_f(df))
                plt.imshow(wordcloud)
                plt.axis("off")
                st.pyplot(fig=plt)

        if option == 'Sandwiches':
            df = load_data()
            df = df[df["Topic"] == 3]
            col1, col2 = st.columns(2)
            with col1:
                for i in range(2):
                    st.write("Name of reviewer: ", df["Nom"].iloc[i])
                    st.write(df["Dates"].iloc[i])
                    st.write("Title", df["Title_Review"].iloc[i])
                    st.write(int(df["Score_reviews"].iloc[i]) * ':star:', df["Score_reviews"].iloc[i])

                    st.write("Reviews :", df["text"].iloc[i])
                    st.write("_____________")
                agree = st.checkbox('View More Reviews',key=compteur)
                compteur += 1

                if agree:
                    st.write('Great!')
                    for i in range(2, 10):
                        st.write("Name of reviewer: ", df["Nom"].iloc[i])
                        st.write(df["Dates"].iloc[i])
                        st.write("Title", df["Title_Review"].iloc[i])
                        st.write(int(df["Score_reviews"].iloc[i]) * ':star:', df["Score_reviews"].iloc[i])

                        st.write("Reviews :", df["text"].iloc[i])
                        st.write("_____________")
            with col2:
                st.subheader(f"WordCloud Reviews TOPIC- {option} ")
                wordcloud = WordCloud(background_color='white', max_words=50).generate(reviews_wordcloud_f_f(df))
                plt.imshow(wordcloud)
                plt.axis("off")
                st.pyplot(fig=plt)

        if option == 'Good day':
            df = load_data()
            df = df[df["Topic"] == 4]
            col1, col2 = st.columns(2)
            with col1:
                for i in range(2):
                    st.write("Name of reviewer: ", df["Nom"].iloc[i])
                    st.write(df["Dates"].iloc[i])
                    st.write("Title", df["Title_Review"].iloc[i])
                    st.write(int(df["Score_reviews"].iloc[i]) * ':star:', df["Score_reviews"].iloc[i])

                    st.write("Reviews :", df["text"].iloc[i])
                    st.write("_____________")
                agree = st.checkbox('View More Reviews',key=compteur)
                compteur += 1

                if agree:
                    st.write('Great!')
                    for i in range(2, 10):
                        st.write("Name of reviewer: ", df["Nom"].iloc[i])
                        st.write(df["Dates"].iloc[i])
                        st.write("Title", df["Title_Review"].iloc[i])
                        st.write(int(df["Score_reviews"].iloc[i]) * ':star:', df["Score_reviews"].iloc[i])

                        st.write("Reviews :", df["text"].iloc[i])
                        st.write("_____________")

            with col2:
                st.subheader(f"WordCloud Reviews TOPIC- {option} ")
                wordcloud = WordCloud(background_color='white', max_words=50).generate(reviews_wordcloud_f_f(df))
                plt.imshow(wordcloud)
                plt.axis("off")
                st.pyplot(fig=plt)

        col1,col2 = st.columns(2)
        with col2:
            #df=load_data()
            df= df[df["Score_reviews"]<3]
            st.subheader("WordCloud Reviews score < 3")
            wordcloud = WordCloud(background_color='white', max_words=50).generate(reviews_wordcloud_f_f(df))
            plt.imshow(wordcloud)
            plt.axis("off")
            st.pyplot(fig=plt)











if selection_mode=="Sentiment Analysis Module":
    st.title("Sentiment Analyzer Based On Text Analysis ")
    st.write('\n\n')
    all_data = get_all_data()
    if st.checkbox('Show PreProcessed Dataset'):
        st.write(preprocessing_data(all_data))


    if st.checkbox('Show Dataset'):
        st.write(all_data)

    training_data, evaluation_data = preprocessing_step()
    vectorizer = CountVectorizer(binary='true')
    classifier = training_step(training_data, vectorizer)




    review = st.text_input("Enter The Review", "Write Here...")
    if st.button('Predict Sentiment'):
        result = print_result(analyse_text(classifier, vectorizer, review))
        st.success(result[1])
    else:
        st.write("Press the above button..")


if selection_mode == "Download Scraped Dataframe in CSV":
    df= load_data()
    csv = convert_df(df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='df.csv',
        mime='text/csv',
    )