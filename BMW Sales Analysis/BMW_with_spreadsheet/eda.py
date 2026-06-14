import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import time
import kagglehub
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import json

# Google Sheets authentication
scope = ["https://www.googleapis.com/auth/spreadsheets", 
         "https://www.googleapis.com/auth/drive"]
creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes = scope)

client = gspread.authorize(creds)


# Load credentials from local JSON file
# with open("service_account.json") as source:
#     creds_dict = json.load(source)

# creds = Credentials.from_service_account_info(creds_dict, scopes=scope)

# client = gspread.authorize(creds)
    
# Connect to google sheet
SHEET_ID = "1IUGSiE2dHLjW9owhLlLqG61EA5Llpy8GRFWeDru0VsM"
sheet = client.open_by_key(SHEET_ID).sheet1




# ------------------------------------------------
# Streamlit page setup
# ------------------------------------------------
st.set_page_config(
    page_title="BMW Sales EDA",
    page_icon="🚗",
    initial_sidebar_state = "collapsed",
    layout="centered"
)
# ------------------------------------------------
# Custom Background and Styling
# ------------------------------------------------
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://bimmers.parts/cdn/shop/files/DSC00277Large.jpg?v=1688454477&width=1445");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stSidebar"] {
    background-color: black;
}
.stButton>button {
    background-color: #00adb5;
    color: white;
    border-radius: 10px;
    height: 2.5em;
    width: 100%;
    font-weight: bold;
}
.stTextInput>div>div>input {
    border-radius: 10px;
}
h1, h2, h3 {
    color: white;
}
div.block-container {
    background-color: rgba(0,0,0,0.6);
    padding: 2em;
    border-radius: 15px;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
# Download latest version
path = kagglehub.dataset_download("ahmadrazakashif/bmw-worldwide-sales-records-20102024")
df = pd.read_csv(path + "/BMW sales data (2010-2024) (1).csv")
df.drop(columns=['Mileage_KM'], inplace=True)

categorical_features = []
numerical_features = []
for col in df.columns:
    if df[col].dtypes == object:
        categorical_features.append(col)
    else:
        numerical_features.append(col)


def dashboard():
    global categorical_features, numerical_features
    st.title("BMW SALES ANALYSIS(EDA) & PREDICTION...")
    st.sidebar.title("🚗 Exploratory Data Analysis ")
    page = st.sidebar.radio("📂", ["Overview", "Outlier Detection", "Pie-Chart","Univariate Analysis","Bivariate Analysis", 
                                   "Trend Analysis","Correlation Relationship","Key Insights", "Conclusion","Future Sales"])
    if page == "Overview":
       
       st.subheader("Dataset Overview")
       st.write(df.head(3))
       st.write("        ")
       st.write(" ")
       st.write(" ")
       st.subheader("Statistical Summary")
       st.write(df.describe())
       st.write("### Categorical Value Counts")
       st.write(df['Model'].value_counts())
       st.write(df['Region'].value_counts())
       st.write(df['Color'].value_counts())
       st.write(df['Fuel_Type'].value_counts())
       st.write(df['Transmission'].value_counts())
       st.write("### Missing Values in Each Column")
       st.write(df.isnull().sum())

       st.write("### Categorical Features and their Unique Values")
       for col in categorical_features:
        st.write(f"{col}")
        st.write(sorted(df[col].unique()))
        st.write("")
       st.subheader("Insights")
       st.write(f'''1. Data contains 50000 rows and 11 columns
                    \n2. The dataset contains a mix of categorical and numerical features.
                    \n3. Numerical features include {numerical_features[0]}, {numerical_features[1]}, {numerical_features[2]}, {numerical_features[3]}. 
                    \n4. The dataset appears to be well-structured with no missing values in the displayed overview.
                    \n5. Categorical features include Model, Region, Color, Fuel_Type, and Transmission.
                    \n6. According to this data bwm sold their cars from 2010-2024.
                    \n7. Their Engine size varies from 1.5L-5.0L.
                    \n8. Price range of cars is from 20,000 USD to 150,000 USD.''')
#-----------------------------------------------------------------------------------------------------------------------
    elif page == "Outlier Detection":
        st.title("Outlier Detection using Box Plots")
        for col in numerical_features:
            plt.figure(figsize=(8, 3))
            sns.boxplot(y = col, data = df, color="orange")
            plt.title(f"{col} Distribution ")
            plt.xlabel(f'{col}')
            plt.ylabel("Values")
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)
        st.subheader("Insights")
        st.write('''1. No outliers are present in the numerical features as per the box plot analysis
                    \n2. All the columns are in proper type
                    \n3. No need of outlier treatment as no outliers are present in the data
                    \n4. Data is clean and ready for further analysis or modeling
                    \n5. Engine size is mostly concentrated around 2000-3500 CC''')
#-----------------------------------------------------------------------------------------------------------------------
    elif page == "Pie-Chart":
        st.title("Categorical Features Pie-Chart Analysis")
        for col in categorical_features:
            plt.figure(figsize=(8, 5))
            pie_data = df[col].value_counts()
            plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
            plt.title(f"{col} Distribution ")
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)
        
        st.subheader("Insights")
        st.write('''1. All categories in each categorical column are fairly represented in the dataset.
                    \n2. The distribution of car models, regions, colors, fuel types, and transmissions appears balanced without extreme dominance by any single category.
                    \n3. This balance is beneficial for building robust machine learning models as it reduces bias towards any particular category.
                    ''')
#-----------------------------------------------------------------------------------------------------------------------
    elif page == "Univariate Analysis":
        st.title("Individual feature study")
        tabs = st.tabs(['Categorical Features', "Numerical Features"])

        with tabs[0]:  
            st.subheader("Graphical Representation")      
            for col in categorical_features:
                plt.figure(figsize= (8, 5))
                ax = sns.countplot(x = col, data = df, palette="magma", order=df[col].value_counts().reset_index()[col])        
                for cont in ax.containers:
                    ax.bar_label(cont)
                plt.title(f"{col} Distribution ")
                plt.xticks(rotation = 50)
                plt.xlabel(f'{col}')
                plt.tight_layout()
                plt.show()
                st.pyplot(plt)
            st.subheader("Insights")
            st.write('''1. all categories in each categorical col are almost equally distributed.
                    \n2. Model: The dataset includes a variety of BMW models, with some models being more popular than others where 7 Series have the highest number of records.
                    \n3. Region: Sales are distributed across multiple regions, indicating a global presence.
                    \n4. Asians are the most recorded people out here.
                    \n5. Most of the Cars are Red in color.
                    \n6. Data mainly have Hybrid type fuel and Manual Transmission.''')
            
        with tabs[1]:
            st.subheader("Numerical Analysis")
            for col in numerical_features:
                plt.figure(figsize=(8, 3))
                sns.histplot(df[col], kde=True, color="blue")
                plt.title(f"Distribution of {col}")
                plt.xlabel(col)
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.show()
                st.pyplot(plt)
            
            st.write("Insights")
            st.write('''1. data has been properly in the format no negative values are present here
                        \n2. All features have a uniform distribution.
                        \n3. from the above analysis we can say that no need of mileage col no relationship found.
                        \n4. hence we can drop the mileage column from the data for better analysis.
                        \n5. Engine size varies with 3 categories like 1.5-2.5L(low), 3.0-3.5L(Medium), 4.0-5.0L(Heavy).
                        \n6. Price Range varies from 0.2M$ - 1.2M$.
                        ''')
#-----------------------------------------------------------------------------------------------------------------------        
    elif page == "Bivariate Analysis":
        
        st.title("Sales Trend by features")
        st.write("Analyzing sales trends based on different features.")
       
        st.subheader("Graphical Representation")
        for col in categorical_features:
            idx = categorical_features.index(col)
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x = col, y = 'Sales_Volume', data = df,ci = None,  estimator = sum, order = df.groupby(col)['Sales_Volume'].sum().sort_values(ascending=False).index, palette='magma')
            for cont in ax.containers:
                ax.bar_label(cont, fmt='%.0f')
            idx = idx + 1
            plt.title(f"Total BMW Sales by {col}")
            plt.tight_layout()
            plt.show()
            st.pyplot(plt)
        st.subheader("Insights")
        st.write('''1. BMW sold almost same amount of different Models whereas 7 Series generates the highest revenue for the company and M3 lowest.
                \n2. It clearly reflecting the behaviour of customers are showing interest not only to specific model but also to every model equally.
                \n3. Mostly Asians are Beemer Lover 
                \n4. BMW Sales is approximately same among rest of the regions
                \n6. Customers are more interested in Hybrid type of cars because they might gives more mileage.
                \n7. Manual Transmission Cars leading the race.
                \n8. Consumers showing more interest in Red BMW and generating more profit by red.
                \n9. Hybrid Fuel Type of Car is in more demand as compared to rest of the fuel
                \n10. Manual driving generates more renvenue as compared to auto''')
# ------------------------------------------------------------------------------------------------------------------
    elif page == "Trend Analysis":
        st.title("Multivariate Analysis of Trend for Sales & Price on Features")
        st.write("How sales and prices varies over years.")
        tabs = st.tabs(['Sales_Volume', 'Price_USD'])
        with tabs[0]:
        
            plt.figure(figsize=(20, 20))
            plt.subplot(6, 1, 1)
            ax = sns.lineplot(x = 'Year', y = 'Sales_Volume', data = df, ci = None, estimator= sum, palette='magma')
            for cont in ax.containers:
                ax.bar_label(cont, fmt='%.0f')  
            plt.title("Total BMW Sales by Year")
            plt.tight_layout()

            plt.subplot(6, 1, 2)
            ax = sns.lineplot(x = 'Year', y = 'Sales_Volume', hue = 'Model', data = df, ci = None, estimator=sum, palette = 'flare')

            plt.subplot(6, 1, 3)
            ax = sns.lineplot(x = 'Year', y = 'Sales_Volume',hue = 'Region', data = df, estimator=sum,palette='magma', ci = None) 

            plt.subplot(6, 1, 4)
            ax = sns.lineplot(x = 'Year', y = 'Sales_Volume',hue = 'Color',  data = df, estimator=sum, ci = None) 

            plt.subplot(6, 1, 5)
            ax = sns.lineplot(x = 'Year', y = 'Sales_Volume',hue = 'Fuel_Type', data = df, estimator=sum,palette='magma', ci = None) 

            plt.subplot(6, 1, 6)
            ax = sns.lineplot(x = 'Year', y = 'Sales_Volume',hue = 'Transmission', data = df, estimator=sum,palette='flare', ci = None) 


            plt.show()
            st.pyplot(plt)
            st.subheader("Insights")
            st.write('''1. Company is being almost consistent making profit from 2010 to 2017.
                        \n2. After 2018 it leads to showing some alternate fluctuations per year till 2024 by increasing lately.
                        \n3. Peak Sales goes in 2022 but slightly decrease in next two years.
                        \n4. Recently X6 Sales increased in few years whereas M3 decreased.
                        \n5. Rest model like (7 series, 3 Series, M5, etc) remains almost at a moderate sales volume.
                        \n6. We can clearly see that Company's target market is Asia having high sales value over years.
                        \n6. Black , Silver and Grey BMW is in demand during 2024, Grey beemer generated peak renvenue in the year of 2016. 
                        \n7. Diesel demands decreased , Petrol and Electric vehicles remains at the similar level but Demand of Hybrid cars were raised.
                        \n8. They should generate more Hybrid cars their market will definetly rise in future.
                        \n9. Manual Transmission in BMW always gives driver a better experience to explore it's full potential ''')

                
        with tabs[1]:

            plt.figure(figsize=(20, 20))
            plt.subplot(6, 1, 1)
            ax = sns.lineplot(x = 'Year', y = 'Price_USD', data = df, ci = None, estimator= np.mean, palette='magma')
            for cont in ax.containers:
                ax.bar_label(cont, fmt='%.0f')  
            plt.title("Total BMW Sales by Year")
            plt.tight_layout()

            plt.subplot(6, 1, 2)
            ax = sns.lineplot(x = 'Year', y = 'Price_USD', hue = 'Model', data = df, ci = None, estimator=sum, palette = 'flare')

            plt.subplot(6, 1, 3)
            ax = sns.lineplot(x = 'Year', y = 'Price_USD',hue = 'Region', data = df, estimator=sum,palette='magma', ci = None) 

            plt.subplot(6, 1, 4)
            ax = sns.lineplot(x = 'Year', y = 'Price_USD',hue = 'Color',  data = df, estimator=sum, palette = 'flare', ci = None) 

            plt.subplot(6, 1, 5)
            ax = sns.lineplot(x = 'Year', y = 'Price_USD',hue = 'Fuel_Type', data = df, estimator=sum,palette='magma', ci = None) 

            plt.subplot(6, 1, 6)
            ax = sns.lineplot(x = 'Year', y = 'Price_USD',hue = 'Transmission', data = df, estimator=sum,palette='flare', ci = None) 
            plt.show()
            st.pyplot(plt)
            st.subheader("Insights")
            st.write("Price variation is similar to Sales_Volume")
# ------------------------------------------------------------------------------------------------------------------
    elif page == "Correlation Relationship":
        # Create a 1x2 subplot layout
        st.subheader("Examine correlation between numerical features:")
        st.write("Heatmap using Correlation")
        plt.figure(figsize=(10, 5))
        sns.heatmap(df[numerical_features].corr(), annot=True)
        plt.show()
        st.pyplot(plt)

        st.subheader("Insight")
        st.write('''1. Positive relationship between Price and Engine size for obvious reasons.
                \n2.   Reverse Relationship between Price and Sales Volume.''')
# ------------------------------------------------------------------------------------------------------------------
    elif page == "Key Insights":
        st.header("Key Insight : ")
        st.write('''| Category                  | Example Insights                                                                                                      |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Market Trends**         | Sales of hybrid/electric BMWs increasing post-2020, suggesting EV transition is key for growth.                       |
| **Regional Focus**        | North America & Europe have highest average prices; Asia has highest unit sales → suggests luxury vs volume strategy. |
| **Product Mix**           | X-Series (SUVs) showing strong global demand → allocate marketing and production resources accordingly.               |
| **Customer Preferences**  | Automatic transmissions dominate across all regions — reduce manual inventory.                                        |
| **Color & Design Trends** | Black and Blue cars dominate sales in luxury categories → color influences perception.                                |
| **Pricing Strategy**      | Moderate price increases don’t harm sales in premium regions but can reduce volume in price-sensitive markets.        |
''')
# ------------------------------------------------------------------------------------------------------------------
    elif page == "Conclusion":
        st.header("Recommendation for Company Growth")
        st.subheader("Based on EDA findings, BMW can make data-driven business decisions:")
        st.write('''
1. Expand Electric & Hybrid Lineup — as demand increases year-over-year.
\n2. Region-Specific Strategies — focus luxury marketing in North America, affordability in Asia.
\n3. Boost SUV Production (X Series) — proven high sales and revenue generation.
\n4. Leverage Data-Driven Pricing — use elasticity insights to optimize price per region.
\n5. Personalized Marketing — highlight popular colors and features by region.
\n6. Predictive Sales Planning — integrate EDA results into regression models for forecasting.''')
# ------------------------------------------------------------------------------------------------------------------
    elif page == "Future Sales":
        
        st.header("🚗 Predict Future BMW Sales Volume")
        st.write("Use the trained model to forecast future sales volume based on car features.")
        st.dataframe(pd.read_csv('model.csv').drop(columns = ['Unnamed: 0']))

        df.drop(columns=['Mileage_KM', 'Price_USD'], inplace=True, errors='ignore')

        # categorical columns
        categorical_features = [col for col in df.columns if df[col].dtype == 'object']

        # Encode categorical columns and store encoders
        encoders = {}
        for col in categorical_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le  # store encoder for future use

    
        X = df.drop(columns=['Sales_Volume'])
        y = df['Sales_Volume']

        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        file_path = 'Linear_regression'
        with open(file_path + '.pkl', 'rb') as file:
            model = pickle.load(file)

        # -------------------------------
        # Model Evaluation
        # -------------------------------
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        with st.expander("📈 Model Performance Metrics"):
            st.write(file_path.upper() + ' MODEL PERFORMANCE')
            st.write(f"**R² Score:** {r2:.3f}")
            st.write(f"**MAE:** {mae:.2f}")
            st.write(f"**RMSE:** {rmse:.2f}")
        with st.expander("📈Sales Prediction "):
             # USER INPUT SECTION
            # -------------------------------
            st.subheader("🔮 Enter Future Car Specifications to Predict Sales Volume")
            model_input = st.selectbox("Select Model", encoders['Model'].classes_)
            year_input = st.number_input("Year", min_value=2000, max_value=2030, step=1)
            region_input = st.selectbox("Select Region", encoders['Region'].classes_)
            color_input = st.selectbox("Select Color", encoders['Color'].classes_)
            fuel_input = st.selectbox("Fuel Type", encoders['Fuel_Type'].classes_)
            trans_input = st.selectbox("Transmission", encoders['Transmission'].classes_)
            engine_input = st.number_input("Engine Size (L)", min_value=1.0, max_value=6.0, step=0.1)
            sales_class_input = st.selectbox("Sales Classification", encoders['Sales_Classification'].classes_)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if st.button("Predict Sales Volume"):
                # Create DataFrame for prediction
                input_data = pd.DataFrame({
                    'Model': [model_input],
                    'Year': [year_input],
                    'Region': [region_input],
                    'Color': [color_input],
                    'Fuel_Type': [fuel_input],
                    'Transmission': [trans_input],
                    'Engine_Size_L': [engine_input],
                    'Sales_Classification': [sales_class_input]
                })

                # Encode categorical inputs using stored encoders
                for col in input_data.columns:
                    if col in encoders:
                        input_data[col] = encoders[col].transform(input_data[col])

                # Align columns with training data
                input_data = input_data[X.columns]

                # Predict
                prediction = model.predict(input_data)[0]
                st.success(f"💡 **Predicted Future Sales Volume:** {prediction:,.0f} units")

                # ✅ Now save to Google Sheet *after* prediction
                try:
                    sheet.append_row([
                        model_input, year_input, region_input, color_input,
                        fuel_input, trans_input, engine_input,
                        sales_class_input, round(prediction, 2)
                    ])
                    st.success("✅ Data successfully saved to Google Sheet!")
                except Exception as e:
                    st.error(f"⚠️ Could not save to Google Sheet: {e}")


dashboard()


