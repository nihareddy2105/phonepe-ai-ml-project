
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="PhonePe AI Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("PhonePe AI/ML Analytics Dashboard")
st.caption(
    "Digital Payment Volume Forecasting using PhonePe Pulse data"
)

BASE_URLS = [
    "https://raw.githubusercontent.com/PhonePe/pulse/master/"
    "data/map/transaction/hover/country/india",
    "https://raw.githubusercontent.com/PhonePe/pulse/main/"
    "data/map/transaction/hover/country/india"
]


@st.cache_data(ttl=3600)
def load_data():
    records = []

    for year in range(2018, 2027):
        for quarter in range(1, 5):
            data = None

            for base in BASE_URLS:
                url = f"{base}/{year}/{quarter}.json"

                try:
                    response = requests.get(url, timeout=15)

                    if response.status_code == 200:
                        data = response.json()
                        break
                except requests.RequestException:
                    continue

            if data is None:
                continue

            items = (
                data.get("data", {})
                .get("hoverDataList", [])
            )

            for item in items:
                metrics = item.get("metric", [])

                if not metrics:
                    continue

                count = metrics[0].get("count")

                if count is None:
                    continue

                records.append({
                    "state": item["name"],
                    "year": year,
                    "quarter": quarter,
                    "period": year * 4 + quarter,
                    "transactions": float(count)
                })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            "No data downloaded. Check the internet connection "
            "or PhonePe Pulse repository."
        )

    return (
        df.drop_duplicates(["state", "period"])
        .sort_values(["state", "period"])
        .reset_index(drop=True)
    )


@st.cache_resource
def train_model(df):
    data = df.copy()

    # Create historical features separately for each state
    grouped = data.groupby("state")["transactions"]

    data["lag1"] = grouped.shift(1)
    data["lag4"] = grouped.shift(4)

    data["rolling4"] = data.groupby("state")[
        "transactions"
    ].transform(
        lambda s: s.shift(1).rolling(4).mean()
    )

    data = data.dropna(
        subset=["lag1", "lag4", "rolling4"]
    ).copy()

    features = [
        "state",
        "year",
        "quarter",
        "lag1",
        "lag4",
        "rolling4"
    ]

    # Keep the latest four available quarters for testing
    last_period = data["period"].max()
    test_start = last_period - 3

    train = data[data["period"] < test_start].copy()
    test = data[data["period"] >= test_start].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Not enough historical data to train and test."
        )

    preprocessor = ColumnTransformer([
        (
            "state",
            OneHotEncoder(handle_unknown="ignore"),
            ["state"]
        )
    ], remainder="passthrough")

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train on log-transformed transaction counts
    model.fit(
        train[features],
        np.log1p(train["transactions"])
    )

    # Predict on the test period and convert to original scale
    test["predicted"] = np.maximum(
        0,
        np.expm1(model.predict(test[features]))
    )

    # Simple baseline: use the previous quarter's count
    test["baseline"] = test["lag1"]

    metrics = {
        "Model MAE": mean_absolute_error(
            test["transactions"], test["predicted"]
        ),
        "Model RMSE": np.sqrt(mean_squared_error(
            test["transactions"], test["predicted"]
        )),
        "Baseline MAE": mean_absolute_error(
            test["transactions"], test["baseline"]
        ),
        "Baseline RMSE": np.sqrt(mean_squared_error(
            test["transactions"], test["baseline"]
        ))
    }

    # Refit the model using all available labelled history
    model.fit(
        data[features],
        np.log1p(data["transactions"])
    )

    return model, data, test, metrics, features


try:
    with st.spinner("Downloading PhonePe Pulse data..."):
        raw = load_data()

    with st.spinner("Training the Random Forest model..."):
        model, history, test, metrics, features = train_model(raw)

    st.success("Data loaded and model trained successfully!")

    latest = raw.loc[raw["period"].idxmax()]

    col1, col2, col3 = st.columns(3)

    col1.metric("States", raw["state"].nunique())
    col2.metric("Quarterly records", f"{len(raw):,}")
    col3.metric(
        "Latest recorded transaction count",
        f"{latest['transactions']:,.0f}"
    )

    st.subheader("Model Evaluation")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Random Forest MAE", f"{metrics['Model MAE']:,.0f}")
    c2.metric("Random Forest RMSE", f"{metrics['Model RMSE']:,.0f}")
    c3.metric("Baseline MAE", f"{metrics['Baseline MAE']:,.0f}")
    c4.metric("Baseline RMSE", f"{metrics['Baseline RMSE']:,.0f}")

    if metrics["Model MAE"] < metrics["Baseline MAE"]:
        st.success(
            "Random Forest has lower MAE than the baseline."
        )
    else:
        st.warning(
            "Random Forest did not beat the baseline on MAE. "
            "Use the measured results honestly."
        )

    st.subheader("Actual vs Predicted Transactions")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        test["transactions"],
        test["predicted"],
        alpha=0.7
    )

    max_value = max(
        test["transactions"].max(),
        test["predicted"].max()
    )

    ax.plot(
        [0, max_value],
        [0, max_value],
        linestyle="--"
    )

    ax.set_xlabel("Actual transaction count")
    ax.set_ylabel("Predicted transaction count")
    ax.set_title("Test Set: Actual vs Predicted")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Explore a State")

    states = sorted(raw["state"].unique())
    selected = st.selectbox(
        "Choose an Indian state",
        states
    )

    state_history = (
        raw[raw["state"] == selected]
        .sort_values("period")
        .copy()
    )

    state_history["label"] = (
        state_history["year"].astype(str)
        + " Q"
        + state_history["quarter"].astype(str)
    )

    fig2, ax2 = plt.subplots(figsize=(10, 4))

    ax2.plot(
        state_history["label"],
        state_history["transactions"],
        marker="o"
    )

    ax2.set_title(f"{selected}: Historical Transactions")
    ax2.set_xlabel("Quarter")
    ax2.set_ylabel("Transaction count")
    ax2.tick_params(axis="x", rotation=70)

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.subheader("Next-Quarter Forecast")

    if len(state_history) >= 4:
        recent = state_history.tail(4)
        last = recent.iloc[-1]

        next_year = int(last["year"])
        next_quarter = int(last["quarter"]) + 1

        if next_quarter == 5:
            next_quarter = 1
            next_year += 1

        input_row = pd.DataFrame([{
            "state": selected,
            "year": next_year,
            "quarter": next_quarter,
            "lag1": float(last["transactions"]),
            "lag4": float(recent.iloc[0]["transactions"]),
            "rolling4": float(recent["transactions"].mean())
        }])

        # Convert the model's log prediction back to counts
        forecast = max(
            0,
            float(
                np.expm1(model.predict(input_row[features])[0])
            )
        )

        st.metric(
            f"Predicted transactions for {next_year} Q{next_quarter}",
            f"{forecast:,.0f}"
        )

        st.caption(
            "This is a model-generated estimate, not an official "
            "PhonePe forecast."
        )
    else:
        st.warning(
            "Not enough historical data to forecast this state."
        )

    st.subheader("Test Predictions")

    st.dataframe(
        test[[
            "state",
            "year",
            "quarter",
            "transactions",
            "predicted",
            "baseline"
        ]].round(2),
        use_container_width=True
    )

    st.download_button(
        "Download Test Predictions CSV",
        test.to_csv(index=False).encode("utf-8"),
        file_name="phonepe_test_predictions.csv",
        mime="text/csv"
    )

    st.markdown(
        "[Official PhonePe Pulse dataset](https://github.com/PhonePe/pulse)"
    )

    st.caption(
        "Academic project using public aggregated data. "
        "This is not PhonePe's internal forecasting system."
    )

except Exception as exc:
    st.error(f"The dashboard could not run: {exc}")
    st.info(
        "Open Manage app and check the logs if the error continues."
    )
