This version is structured for real-world deployment, with database connection, SQLAlchemy setup, and Streamlit integration.

---

# 🚗 BIMMER Sales Prediction Dashboard

A **Streamlit web application** that predicts BMW car sales volume based on model, year, region, and other parameters.
The app stores **user input data and predictions** directly into a **MySQL database**, providing a scalable and secure backend for analytics and reporting.

---

## 🌟 Features

✅ Interactive user input fields for sales prediction
✅ Machine learning model integration (`Linear Regression`)
✅ Automatic saving of inputs + predictions into MySQL
✅ Secure connection using environment variables or `st.secrets`
✅ Real-time data retrieval and visualization
✅ Ready for deployment on **Streamlit Cloud** or **local server**

---

## 🧩 Tech Stack

| Component      | Technology                  |
| -------------- | --------------------------- |
| **Frontend**   | Streamlit                   |
| **Backend**    | Python                      |
| **Model**      | Trained Linear Regression   |
| **Database**   | MySQL                       |
| **ORM**        | SQLAlchemy + PyMySQL        |
| **Deployment** | Streamlit Cloud / Localhost |

---

## ⚙️ Prerequisites

Before running the project, ensure you have:

* Python 3.8+
* MySQL Server (running locally or remotely)
* Streamlit
* SQLAlchemy
* PyMySQL
* pandas
* scikit-learn
* pickle

Install dependencies:

```bash
pip install streamlit sqlalchemy pymysql pandas scikit-learn
```

---

## 🛠️ Setting Up the MySQL Database

### 1️⃣ Create a new MySQL database

Open MySQL CLI or MySQL Workbench and run:

```sql
CREATE DATABASE bmw_sales;
USE bmw_sales;
```

### 2️⃣ Create a table to store sales data

```sql
CREATE TABLE sales_inputs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model VARCHAR(50),
    year INT,
    region VARCHAR(50),
    engine_size FLOAT,
    sales_class VARCHAR(50),
    predicted_sales FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

✅ This table will store each record you submit from the Streamlit form.

---

## 🔒 Secure Database Connection

Instead of hardcoding your MySQL password, store credentials in Streamlit secrets.

### Step 1 — Create `.streamlit/secrets.toml`

```toml
[mysql]
host = "localhost"
port = "3306"
user = "root"
password = "your_password"
database = "bmw_sales"
```

> ⚠️ Keep this file private and **never commit it** to GitHub.

### Step 2 — Access secrets in Streamlit

When deployed to **Streamlit Cloud**:

* Go to your app → ⚙️ **Settings → Secrets**
* Copy the `[mysql]` block above and paste your own values

---

## 🧠 Core Code Example

Here’s the complete example to connect to MySQL, save inputs, and view stored data:

```python
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import pickle
from sklearn.linear_model import LinearRegression

# -----------------------
# DATABASE CONNECTION
# -----------------------
mysql_config = st.secrets["mysql"]
engine = create_engine(
    f"mysql+pymysql://{mysql_config['user']}:{mysql_config['password']}@{mysql_config['host']}:{mysql_config['port']}/{mysql_config['database']}"
)

# -----------------------
# STREAMLIT APP UI
# -----------------------
st.title("BIMMER Sales Prediction Dashboard")

st.write("Predict BMW car sales and store them securely in a MySQL database.")

model_name = st.selectbox("Model", ["X1", "X3", "X5"])
year = st.slider("Year", 2015, 2025, 2023)
region = st.selectbox("Region", ["Europe", "Asia", "North America"])
engine_size = st.number_input("Engine Size (L)", min_value=1.0, max_value=5.0, value=2.0)
sales_class = st.selectbox("Sales Classification", ["Economy", "Luxury", "Performance"])

# -----------------------
# LOAD MODEL
# -----------------------
# Load your trained model from pickle (if applicable)
# with open("linear_model.pkl", "rb") as f:
#     model = pickle.load(f)
# For demonstration, simulate prediction:
predicted_sales = 25000 + hash(model_name + region) % 10000

st.write(f"Predicted Sales Volume: **{predicted_sales} units**")

# -----------------------
# SAVE TO DATABASE
# -----------------------
if st.button("Save Data"):
    insert_query = f"""
        INSERT INTO sales_inputs (model, year, region, engine_size, sales_class, predicted_sales)
        VALUES ('{model_name}', {year}, '{region}', {engine_size}, '{sales_class}', {predicted_sales});
    """
    with engine.begin() as conn:
        conn.execute(insert_query)
    st.success("Data successfully saved to the MySQL database!")

# -----------------------
# VIEW DATABASE RECORDS
# -----------------------
if st.checkbox("Show Saved Records"):
    query = "SELECT * FROM sales_inputs ORDER BY created_at DESC"
    df = pd.read_sql(query, engine)
    st.dataframe(df)
```

---

## 🚀 Deploying to Streamlit Cloud

1. Push your project to **GitHub**
2. Go to [Streamlit Cloud](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add your MySQL credentials in **App → Settings → Secrets**
5. Deploy your app — you’re live! 🎉

Your app will now securely read and write to the **MySQL database**.

---

## 📊 Example Database Records

| id | model | year | region | engine_size | sales_class | predicted_sales | created_at          |
| -- | ----- | ---- | ------ | ----------- | ----------- | --------------- | ------------------- |
| 1  | X3    | 2023 | Europe | 2.0         | Luxury      | 32640           | 2025-11-03 17:30:45 |
| 2  | X5    | 2024 | Asia   | 3.5         | Performance | 37520           | 2025-11-03 17:32:10 |

---

## 🧾 Notes

* Change your DB host if connecting to **remote MySQL (AWS, Render, or PlanetScale)**
* Use **SQLAlchemy ORM models** for advanced insert/query operations
* Make sure the MySQL server is running when testing locally
* For cloud DB, ensure your IP or Streamlit app has permission to connect

---

## 🌐 Live Demo

👉 [**https://bmwsaleseda.streamlit.app/**](https://bmwsaleseda.streamlit.app/)

---

## 👨‍💻 Author

**Tushar Khitoliya**
AI-ML Engineer
📧 Email: [tushar.khitoliya26@gmail.com](mailto:tushar.khitoliya26@gmail.com)
🌐 GitHub: [https://github.com/KHITOLIA](https://github.com/KHITOLIA)

---


