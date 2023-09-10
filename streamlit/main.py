import json
import pathlib

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import precision_recall_curve
from streamlit_shap import st_shap

import streamlit as st


@st.cache_resource
def loaded_columns():
    raw_columns = (
        pathlib.Path(__file__).parent.parent.absolute() / "ML" / "columns.json"
    ).read_text()
    return json.loads(raw_columns)["columns"]


@st.cache_resource
def loaded_model():
    return mlflow.sklearn.load_model(
        pathlib.Path(__file__).parent.parent.absolute() / "ML" / "model"
    )


@st.cache_resource
def full_data():
    path = pathlib.Path(__file__).parent.absolute() / "full.parquet"
    return pd.read_parquet(path)


@st.cache_data
def calc_prediction_full():
    df = full_data()
    last_3m = sorted(df["report_date"].unique())[-3:]

    test_df = (
        df[df["report_date"].isin(last_3m)]
        .reset_index()
        .drop(columns=["index"])
        .drop(["report_date", "client_id"], axis=1)
    )

    test_df = test_df[loaded_columns() + ["target"]]
    predictions = model.predict_proba(test_df)[:, 1]
    return test_df, predictions


@st.cache_data
def calc_prediction_test(df):
    client_id = df["client_id"]
    report_date = df["report_date"]

    df[df.select_dtypes("object").columns] = df[
        df.select_dtypes("object").columns
    ].astype(str)
    predictions = model.predict_proba(df[loaded_columns()])[:, 1]

    result = pd.DataFrame()
    result["client_id"] = client_id
    result["report_date"] = report_date
    result["prediction"] = predictions
    return result.sort_values("prediction", ascending=False).to_html()


def plot_pr_curve(test_y, predictions):
    precision, recall, _ = precision_recall_curve(test_y, predictions)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, marker=".", label="catboost")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    st.pyplot(fig)


def plot_target_segment(test_y, predictions):
    hist_df = pd.DataFrame({"target": test_y.values, "proba": predictions})
    hist_df = hist_df.sort_values(by="proba", ascending=False)

    num_segments = 20
    segment_length = len(hist_df) // num_segments
    array_21 = [21] * (len(hist_df) - num_segments * segment_length)
    hist_df["segment"] = (
        list(np.repeat(range(1, num_segments + 1), segment_length)) + array_21
    )

    hist_df = hist_df.groupby("segment")["target"].sum()
    fig, ax = plt.subplots()
    ax.set_xlabel("Сегмент")
    ax.set_ylabel("Количество контрактов")
    ax.bar(list(hist_df.index), list(hist_df.values))
    st.pyplot(fig)


def plot_for_full():
    df = full_data()
    last_3m = sorted(df["report_date"].unique())[-3:]

    test_df = (
        df[df["report_date"].isin(last_3m)]
        .reset_index()
        .drop(columns=["index"])
        .drop(["report_date", "client_id"], axis=1)
    )

    predictions = model.predict_proba(test_df.drop(["target"], axis=1))[:, 1]

    plot_pr_curve(test_df["target"], predictions)
    plot_target_segment(test_df["target"], predictions)


def plot_for_test(model, df):
    df[df.select_dtypes("object").columns] = df[
        df.select_dtypes("object").columns
    ].astype(str)
    df = df[loaded_columns()]
    plt.cla()
    shap_values = shap.TreeExplainer(model).shap_values(df)
    st_shap(shap.summary_plot(shap_values, df))


def show_prediction(df):
    st.write(calc_prediction_test(df), unsafe_allow_html=True)


model = loaded_model()
uploaded_file = st.file_uploader("Выберите файл")

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file, sep=";")

    pr, roc = 0.21453993484902384, 0.9246884335676997

    st.header("Метрики:")
    st.text(f"roc_auc: {roc}")
    st.text(f"pr_auc: {pr}")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Итоговая таблица",
            "Кривая precision recall",
            "Ранжирование клиентов",
            "Важность признаков",
        ]
    )
    full_df, predictions_full = calc_prediction_full()

    with tab1:
        show_prediction(test_df)

    with tab2:
        plot_pr_curve(full_df["target"], predictions_full)

    with tab3:
        plot_target_segment(full_df["target"], predictions_full)

    with tab4:
        plot_for_test(model, test_df)
