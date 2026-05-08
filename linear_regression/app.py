import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import (r2_score, mean_absolute_error, root_mean_squared_error)

st.title("Универсальное ML-приложение")

uploaded_file = st.file_uploader("Загрузите CSV файл", type=['csv'])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Первые строки датасета")
    st.dataframe(df.head())

    st.subheader("Информация о датасете")
    st.write(df.shape)

    st.subheader("Столбцы")
    st.write(df.columns)

    st.subheader("Визуализация признаков")

    column = st.selectbox("Выберите столбец", df.columns)

    fig, ax = plt.subplots(figsize=(8, 5))

    if df[column].dtype == 'object':

        df[column].value_counts().plot(kind='bar', ax=ax)

        ax.set_ylabel("Количество")

    else:

        ax.hist(df[column], bins=30)

        ax.set_ylabel("Частота")

    ax.set_xlabel(column)

    st.pyplot(fig)

    numeric_columns = df.select_dtypes(include='number').columns

    if len(numeric_columns) >= 2:

        st.subheader("Scatterplot")

        x_col = st.selectbox("Выберите X", numeric_columns, key='scatter_x')

        y_col = st.selectbox("Выберите Y", numeric_columns, key='scatter_y')

        fig2, ax2 = plt.subplots(figsize=(8, 5))

        ax2.scatter(df[x_col], df[y_col])

        ax2.set_xlabel(x_col)
        ax2.set_ylabel(y_col)

        st.pyplot(fig2)

    st.subheader("Матрица корреляций")

    numeric_df = df.select_dtypes(include='number')

    if len(numeric_df.columns) > 1:

        fig3, ax3 = plt.subplots(figsize=(10, 7))

        sns.heatmap( numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax3)

        st.pyplot(fig3)

    st.subheader("Обучение модели")

    target = st.selectbox("Выберите target", df.columns)

    features = st.multiselect("Выберите признаки", [col for col in df.columns if col != target], default=[col for col in df.columns if col != target])

    if len(features) > 0:

        X = df[features]

        y = df[target]

        X = pd.get_dummies(X, drop_first=True)

        X = X.fillna(0)

        y = y.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = Ridge(alpha=100)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)

        mae = mean_absolute_error(y_test, y_pred)

        rmse = root_mean_squared_error(y_test, y_pred)

        st.subheader("Метрики модели")

        st.write("R² score:", round(r2, 3))
        st.write("MAE:", round(mae, 3))
        st.write("RMSE:", round(rmse, 3))

        st.subheader("Предсказание для новых данных")

        numeric_features = X.columns

        input_data = {}

        for col in numeric_features:

            input_data[col] = st.number_input(col, value=0.0)

        if st.button("Предсказать"):

            input_df = pd.DataFrame([input_data])

            for col in X.columns:

                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[X.columns]

            prediction = model.predict(input_df)

            st.success(f"Предсказание: {round(prediction[0], 2)}")

        st.subheader("Важность признаков")

        coef_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})

        coef_df['Absolute'] = abs(coef_df['Coefficient'])

        coef_df = coef_df.sort_values(by='Absolute', ascending=False).head(15)

        fig4, ax4 = plt.subplots(figsize=(10, 6))

        ax4.barh(coef_df['Feature'], coef_df['Coefficient'])

        ax4.set_xlabel("Coefficient")

        st.pyplot(fig4)