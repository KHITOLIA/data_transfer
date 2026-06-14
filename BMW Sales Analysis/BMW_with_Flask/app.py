from flask import Flask, render_template, request, redirect, url_for
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import warnings
import kagglehub
warnings.filterwarnings('ignore')
from datetime import datetime
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import io

load_dotenv()


app = Flask(__name__)

# Database config
db_user = os.getenv('MYSQL_USER')
db_password = os.getenv('MYSQL_PASSWORD')
db_host = os.getenv('MYSQL_HOST')
db_port = os.getenv('MYSQL_PORT')
db_name = os.getenv('MYSQL_DB')


engine = create_engine(
    "mysql+pymysql://avnadmin:AVNS_of8y2Dw_-xbD7tHSYEV@bmw-sales-chitkarauniversity390-8745.e.aivencloud.com:16785/defaultdb",
    connect_args={"ssl": {"ssl_mode": "REQUIRED"}}
)

# Load model
with open('Linear_regression.pkl', 'rb') as f:
    model = pickle.load(f)

# Load dataset
path = kagglehub.dataset_download("ahmadrazakashif/bmw-worldwide-sales-records-20102024")
df_original = pd.read_csv(path + "/BMW sales data (2010-2024) (1).csv")

df_original.drop(columns=['Mileage_KM', 'Price_USD'], inplace=True, errors='ignore')

# Make a copy for modeling (encoding will happen here)
df = df_original.copy()

# categorical columns
categorical_features = [col for col in df.columns if df[col].dtype == 'object']

# Encode categorical columns and store encoders
encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # store encoder for future use

X = df.drop(columns=['Sales_Volume'])

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_sales = None

    if request.method == 'POST':
        model_input = request.form['model']
        year_input = int(request.form['year'])
        region_input = request.form['region']
        color_input = request.form['color']
        fuel_input = request.form['fuel_type']
        trans_input = request.form['transmission']
        engine_input = float(request.form['engine_size'])
        sales_class_input = request.form['sales_classification']

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

        for col in input_data.columns:
            if col in encoders:
                input_data[col] = encoders[col].transform(input_data[col])

        input_data = input_data[X.columns]
        predicted_sales = round(float(model.predict(input_data)[0]), 2)

        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO sales_predictions 
                (Model, Year, Region, Color, Fuel_Type, Transmission, Engine_Size_L, Sales_Classification, Predicted_Sales_Volume)
                VALUES (:model, :year, :region, :color, :fuel_type, :transmission, :engine_size, :sales_class, :prediction)
            """), {
                'model': model_input,
                'year': year_input,
                'region': region_input,
                'color': color_input,
                'fuel_type': fuel_input,
                'transmission': trans_input,
                'engine_size': engine_input,
                'sales_class': sales_class_input,
                'prediction': predicted_sales
            })

    return render_template(
    "index.html",
    predicted_sales=predicted_sales,
    df_original=pd.read_csv(path + "/BMW sales data (2010-2024) (1).csv")
)



# small helpers
def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf8')
    buf.close()
    plt.close()
    return img_b64

# EDA 
@app.route("/eda/overview")
def eda_overview():
    # Basic summary values
    total_rows, total_cols = df.shape
    columns = df.columns.tolist()
    missing_per_col = df.isnull().sum().to_dict()
    describe_html = df.describe().round(3).to_html(classes="table table-sm table-striped", border=0)
    head_html = df.head(5).to_html(classes="table table-sm table-bordered", border=0, index=False)

    # Missing values heatmap
    plt.figure(figsize=(8, 2))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values (True = missing)")
    heatmap_b64 = plot_to_base64()

    categorical = [c for c in df_original.columns if df_original[c].dtype == 'object']
    numerical = [c for c in df_original.columns if pd.api.types.is_numeric_dtype(df_original[c])]


    overview_insights = [
        f"Total rows: {total_rows}, columns: {total_cols}",
        f"Numerical features: {', '.join(numerical)}",
        f"Categorical features: {', '.join(categorical)}",
        "Data span: years 2010–2024 (if Year column exists)",
        "Engine size range and price ranges visible in summary statistics"
    ]

    return render_template(
        "eda_overview.html",
        head_html=head_html,
        describe_html=describe_html,
        heatmap_b64=heatmap_b64,
        total_rows=total_rows,
        total_cols=total_cols,
        categorical=categorical,
        numerical=numerical,
        overview_insights=overview_insights,
        columns=columns,
        missing_per_col=missing_per_col,
        year=2025
    )

# ==========================
# EDA - Outlier Detection
# ==========================
@app.route("/eda/outliers")
def eda_outliers():
    df_num = df_original.select_dtypes(include=[np.number])
    outlier_info = []
    outlier_images = {}

    # --- OUTLIER DETECTION USING IQR & Z-SCORE ---
    for col in df_num.columns:
        data = df_num[col].dropna()

        # IQR Method
        Q1, Q3 = np.percentile(data, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()

        # Z-Score Method
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        z_outliers = (z_scores > 3).sum()

        total_outliers = len(set(data[(data < lower_bound) | (data > upper_bound)].index) |
                             set(data[z_scores > 3].index))
        perc = round((total_outliers / len(data)) * 100, 2)

        outlier_info.append({
            "column": col,
            "iqr_outliers": int(iqr_outliers),
            "z_outliers": int(z_outliers),
            "total_outliers": total_outliers,
            "percentage": perc
        })

        # --- PLOT BOX ---
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=data, color="#1c69d4")
        plt.title(f"{col} - Outlier Distribution", color='white', fontsize=13)
        plt.ylabel(col, color='white')
        plt.xticks(color='white')
        plt.yticks(color='white')
        outlier_images[col] = plot_to_base64()

    # Convert summary to dataframe
    outlier_df = pd.DataFrame(outlier_info)
    outlier_table = outlier_df.to_html(classes="table table-dark table-striped table-hover", index=False)

    # Insights generation
    if not outlier_df.empty:
        most_outliers = outlier_df.loc[outlier_df['total_outliers'].idxmax(), 'column']
        avg_perc = outlier_df['percentage'].mean()
        insights = [
            f"Columns with highest outliers: {most_outliers}",
            f"Average percentage of outliers across numerical columns: {avg_perc:.2f}%",
            f"Outlier detection performed using both IQR (1.5×IQR) and Z-score (>3) methods."
        ]
    else:
        insights = ["No numerical columns found for outlier analysis."]

    return render_template(
        "eda_outliers.html",
        outlier_images=outlier_images,
        outlier_table=outlier_table,
        insights=insights
    )

@app.route("/eda/piechart")
def eda_pie():
    categorical_features = [col for col in df_original.columns if df_original[col].dtype == 'object']
    pie_images = {}
    for col in categorical_features:
        plt.figure(figsize=(6, 6))
        df_original[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
        plt.title(f'Distribution of {col}', color='white', fontsize=14)
        plt.ylabel('')
        pie_images[col] = plot_to_base64()
        insights = [f'''Insights
All categories in each categorical column are fairly represented in the dataset.

The distribution of car models, regions, colors, fuel types, and transmissions appears balanced without extreme dominance by any single category.

This balance is beneficial for building robust machine learning models as it reduces bias towards any particular category.''']
        

    return render_template(
        "eda_piechart.html",
        pie_images=pie_images,
        insights=insights
    )
@app.route("/eda/univariate")
def univariate_analysis():
    univariate_images = {}
    for col in categorical_features:
        plt.figure(figsize= (8, 5))
        ax = sns.countplot(x = col, data = df, palette="magma", order=df[col].value_counts().reset_index()[col])        
        for cont in ax.containers:
            ax.bar_label(cont)
        plt.title(f"{col} Distribution ")
        plt.xticks(rotation = 50)
        plt.xlabel(f'{col}')
        plt.ylabel('Count')
        plt.tight_layout()
        img_b64 = plot_to_base64()
        univariate_images[col] = img_b64
        
    return render_template("eda_univariate.html", univariate_images=univariate_images)

if __name__ == "__main__":
    app.run(debug=True)