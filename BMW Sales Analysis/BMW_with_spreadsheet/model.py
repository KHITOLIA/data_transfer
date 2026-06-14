import pandas as pd
import numpy as np
import kagglehub
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR 
import xgboost as xgb  
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle

def training():
    le = LabelEncoder()
    scaler = StandardScaler()
    path = kagglehub.dataset_download("ahmadrazakashif/bmw-worldwide-sales-records-20102024")
    df = pd.read_csv(path + "/BMW sales data (2010-2024) (1).csv")
    df.drop(columns=['Mileage_KM', 'Price_USD'], inplace=True)

    categorical_features = []
    for col in df.columns:
        if df[col].dtypes == object:
            categorical_features.append(col)
    
    # Encode categorical features
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])



    X = df.drop(columns=['Sales_Volume'])
    y = df['Sales_Volume']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models_performance = []

    model_list = {
        'Linear_regression' : Ridge(),
        'Random_Forest' : RandomForestRegressor(),
        'XGBoost_Regression' : xgb.XGBRegressor(),
    }

    for name, model in model_list.items():
        ml_model = model.fit(x_train, y_train)
        r2 = r2_score(y_test, model.predict(x_test))
        mae = mean_absolute_error(y_test, model.predict(x_test))
        mse = mean_squared_error(y_test, model.predict(x_test))
        rmse = np.sqrt(mean_absolute_error(y_test, model.predict(x_test)))
        adjusted_r2 = 1 - (1 - r2) * (x_test.shape[0] - 1) / (x_test.shape[0] - x_test.shape[0] - 1)
        models_performance.append({'name' : name,
                        'r2' : r2,
                        'MAE' : mae,
                        'MSE' : mse,
                        'RMSE' : rmse,
                        'Adjusted_r2' : adjusted_r2})
        filename = name + '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(ml_model, file)


    model_data = pd.DataFrame(models_performance)
    model_data.to_csv('model.csv')
    return "\n=== Predict Future Sales Volume ==="
model_data = training()
print(model_data)
