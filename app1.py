import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit.components.v1 as components
import numpy as np


df1 = pd.read_csv("modeldataset.csv")
df1 = df1.loc[:, ~df1.columns.str.contains('^Unnamed')]
X = df1.drop('price', axis='columns')
y = df1.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
linear_classifier = LinearRegression()
linear_classifier.fit(X_train, y_train)
linear_classifier.score(X_test, y_test)


def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return linear_classifier.predict([x])[0]



def Main_page():
    header = st.container()
    dataset = st.container()
    features = st.container()
    modelTraining = st.container()

    def plot_no_of_bedrooms(df, location1):
        loc_data = df[(df.location == location1)]
        loc_data['bhk'] = loc_data['bhk'].astype(str)
        fig = px.histogram(loc_data, x="bhk")
        st.plotly_chart(fig, use_container_width=True)

    def plot_averageprice_bhk(df, location):
        loc_data = df[(df.location == location)]
        loc_data['bhk'] = loc_data['bhk'].astype(str)
        loc_data1 = loc_data.groupby(loc_data['bhk'])["price_per_sqft"].agg(["mean"])
        fig = px.line(loc_data1)
        st.plotly_chart(fig, use_container_width=True)




    with header:
        st.title("Real Estate Analyser")
        data = pd.read_csv("visualization.csv")

        location1 = st.text_input('Enter the location:')
        plot_no_of_bedrooms(data, location1)
        plot_averageprice_bhk(data, location1)

        st.header('Predict the price ')
        sqft_value = st.number_input('Enter the value of square feet',min_value=1,value=1,step=1)
        bath_value= st.number_input('Enter the number of bathrooms',min_value=1,value=1,step=1)
        bhk_value = st.number_input('Enter the number of Bedrooms',min_value=1,value=1,step=1)
        st.header('Compare prices with other locality')
        alternate_location = []
        alternate_location1=st.text_input('Enter the location 1:')
        alternate_location2=st.text_input('Enter the location 2:')
        alternate_location.append(alternate_location1)
        alternate_location.append(alternate_location2)
        place1 = predict_price(location1, sqft_value, bath_value, bhk_value)
        alplace = []
        alplace.append(predict_price(alternate_location[0], sqft_value, bath_value, bhk_value))
        alplace.append(predict_price(alternate_location[1], sqft_value, bath_value, bhk_value))

        alplace.append(place1)
        alternate_location.append(location1)
        df_predictprice = pd.DataFrame(list(zip(alternate_location, alplace)),
                                       columns=['location', 'price'])
        fig = px.bar(df_predictprice, x="location", y="price")
        st.plotly_chart(fig, use_container_width=True)


def model():
    st.markdown("# Page 2 ❄️")


def data():
    st.header("ABOUT DATA")
    st.sidebar.markdown("About Data")
    st.subheader("DATASET")
    banglore_data=pd.read_csv('Bengaluru_House_Data.csv',na_values=['='])
    st.dataframe(banglore_data)
    st.subheader("INFORMATION ABOUT DATASET")
    html_temp = """<script type='text/javascript' src='https://prod-apnortheast-a.online.tableau.com/javascripts/api/viz_v1.js'></script><div class='tableauPlaceholder' style='width: 1000px; height: 827px;'><object class='tableauViz' width='1000' height='827' style='display:none;'><param name='host_url' value='https%3A%2F%2Fprod-apnortheast-a.online.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='&#47;t&#47;realestateprediction' /><param name='name' value='RealEstatePrediction&#47;Dashboard1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='showAppBanner' value='false' /></object></div>"""
    components.html(html_temp)


page_names_to_funcs = {
    "Main Page": Main_page,
    "Visualization": data,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()