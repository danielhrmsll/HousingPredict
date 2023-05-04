import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# cargar el modelo entrenado
model = joblib.load('modelo_entrenado.joblib')

# definir la interfaz de usuario
st.title('Predicción de valor medio de la vivienda en California')

# crear un formulario para ingresar los valores de entrada
with st.form(key='formulario_prediccion'):
    st.header('Ingrese los siguientes valores:')
    longitud = st.number_input('Longitud')
    latitud = st.number_input('Latitud')
    edad = st.number_input('Edad media de la vivienda')
    habitaciones = st.number_input('Número total de habitaciones')
    dormitorios = st.number_input('Número total de dormitorios')
    poblacion = st.number_input('Población')
    hogares = st.number_input('Hogares')
    ingreso = st.number_input('Ingreso medio')
    proximidad_oceano = st.selectbox('Proximidad al océano', ['<1H OCEAN', 'INLAND', 'NEAR BAY', 'NEAR OCEAN'])
    if hogares > 0:
        population_per_household = poblacion/hogares
    else:
        population_per_household = 0
    if habitaciones > 0:
        bedrooms_per_room = dormitorios/habitaciones
    else:
        bedrooms_per_room = 0
    if habitaciones > 0:
        rooms_per_household = habitaciones/hogares
    else:
        rooms_per_household = 0
    # hacer una predicción cuando se presiona el botón
    if st.form_submit_button(label='Predecir'):
    # preparar los datos de entrada para la predicción
        housing = pd.DataFrame({
            'longitude': [longitud],
            'latitude': [latitud],
            'housing_median_age': [edad],
            'total_rooms': [habitaciones],
            'total_bedrooms': [dormitorios],
            'population': [poblacion],
            'households': [hogares],
            'median_income': [ingreso],
            'ocean_proximity': [proximidad_oceano],
            'population_per_household': [population_per_household],
            'bedrooms_per_room': [bedrooms_per_room],
            'rooms_per_household': [rooms_per_household]
            })
        # Transformar la variable categórica "oceanproximity" en variables binarias
        ocean_encoder = OneHotEncoder()
        ocean_cat_1hot = ocean_encoder.fit_transform(housing["ocean_proximity"].values.reshape(-1,1))
        ocean_cat_1hot = pd.DataFrame(ocean_cat_1hot.toarray(),
                                      columns=ocean_encoder.categories_[0])
        
        # Agregar las nuevas columnas al conjunto de datos original
        housing = pd.concat([housing, ocean_cat_1hot], axis=1)
        


        imputer = SimpleImputer(strategy="median")
        housing_num = housing.drop("ocean_proximity",axis=1)
        imputer.fit(housing_num)
        imputer.statistics_
        X = imputer.transform(housing_num)
        housing_tr = pd.DataFrame(X,columns=housing_num.columns,index=housing_num.index)
        
        housing_cat = housing[["ocean_proximity"]]
        ordinal_encoder = OrdinalEncoder()
        housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
       
        print(ordinal_encoder.categories_)
        
        cat_encoder = OneHotEncoder()
        housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
        housing_cat_1hot
        housing_cat_1hot.toarray()
        
        rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
        def add_extra_features(X, add_bedrooms_per_room=True):
            rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
            population_per_household = X[:, population_ix] / X[:, household_ix]
            if add_bedrooms_per_room:
                bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
                return np.c_[X, rooms_per_household, population_per_household,
                             bedrooms_per_room]
            else:
                return np.c_[X, rooms_per_household, population_per_household]
        
        attr_adder = FunctionTransformer(add_extra_features, validate=False,
                                         kw_args={"add_bedrooms_per_room": False})
        housing_extra_attribs = attr_adder.fit_transform(housing.values)
        
        num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy="median")),
                ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
                ('std_scaler', StandardScaler()),
            ])
        housing_num_tr = num_pipeline.fit_transform(housing_num)   
     
        print(housing_num.shape)
        num_attribs = list(housing_num)
        cat_attribs = ["ocean_proximity"]
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
        ])
        housing_prepared = full_pipeline.fit_transform(housing)
        print(housing_prepared.shape)
        
        # hacer la predicción
        y_pred = model.predict(housing_prepared)
    
        # mostrar el resultado de la predicción
        st.subheader('Resultado de la predicción:')
        st.write(f'El valor medio de la vivienda en California es de ${y_pred[0]:,.2f}')
        print(y_pred)