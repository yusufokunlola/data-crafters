# import libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
@st.cache_data
def load_data():
    url = 'dataset/cleaned_WHO_Infrastructure.csv'  # Replace with your dataset URL or file path
    df = pd.read_csv(url)
    return df

# The Main function for the Streamlit app
def main():
    st.markdown("<h1 style='text-align: center; color: blue;'>Health Facilities Density Prediction App</h1>", unsafe_allow_html=True)
    st.write("This app uses a Linear Regression model to predict the density of health facilities such as Health posts, Health centers, Rural/District hospitals, Provincial hospitals, Specialized Hospitals and Hospitals based on other features.")

    # Load the dataset
    df = load_data()

    # Sidebar with user inputs
    st.sidebar.header('User Input Features')
    year = st.sidebar.selectbox('Select Year:', [2010, 2013, 2014])
    region = st.sidebar.selectbox('Select Region:', df['Region'].unique())

    try:
        # Get the list of countries in the selected region
        countries_in_region = df[df['Region'] == region]['Country'].unique()

        # Get the number of countries in the selected region
        num_countries = len(countries_in_region)

        # Filter the dataset based on user inputs
        filtered_data = df[(df['Year'] == year) & (df['Region'] == region)]

        # Display the number of countries in the selected region
        st.write(f'### Number of Countries in Selected Region: {num_countries}')
    
        # Create tabs for different sections
        section = st.selectbox("Select to preview the country list and region data", ["Country List", "Region Data"])

        # Display the list of countries in the selected region
        if section == "Country List" and num_countries > 0:
            st.write('### Countries in Selected Region')
            st.write(countries_in_region)
        elif section == "Country List":
            st.write('### No Countries Found in Selected Region')

        # Display the Country Data
        if section == "Region Data":
            st.write('### Region Data')
            st.write(filtered_data)

        # Prepare the feature matrix (X) and target variable (y)
        feature_columns = ['GDP', 'Population', 'Health Expenditure', 'GDP Per Capita', 'Population Density']
        X = filtered_data[feature_columns]
        y = filtered_data['Total_hf_densities']

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
        # create an instance of the Linear Regression and train on the train set
        model = LinearRegression().fit(X_train, y_train)
        
        #  save pickle file
        pickle.dump(model, open('model.pkl','wb'))

        # load model
        reg_model = pickle.load(open('model.pkl','rb'))

        # Prediction
        user_input = {
            'GDP': st.sidebar.number_input('GDP', min_value=float(df['GDP'].min()), max_value=float(df['GDP'].max())),
            'Population': st.sidebar.number_input('Population', min_value=float(df['Population'].min()), max_value=float(df['Population'].max())),
            'Health Expenditure': st.sidebar.number_input('Health Expenditure', min_value=float(df['Health Expenditure'].min()), max_value=float(df['Health Expenditure'].max())),
            'GDP Per Capita': st.sidebar.number_input('GDP Per Capita', min_value=float(df['GDP Per Capita'].min()), max_value=float(df['GDP Per Capita'].max())),
            'Population Density': st.sidebar.number_input('Population Density', min_value=float(df['Population Density'].min()), max_value=float(df['Population Density'].max()))
        }

        # Predict Total Health Facilities Densities for user input
        predicted_value = reg_model.predict([list(user_input.values())])[0]
        predicted_text = f'<span style="color: blue ; font-weight: bold;">Predicted Total Health Facilities Densities:</span> {predicted_value:.2f}'
        st.sidebar.write(predicted_text, unsafe_allow_html=True)
    except ValueError as e:
	    st.error("An error occurred: " + str(e))


if __name__ == '__main__':
    main()
    
st.text('')
st.text('')
st.markdown('`Code:` [GitHub](https://github.com/yusufokunlola/data-crafters)')