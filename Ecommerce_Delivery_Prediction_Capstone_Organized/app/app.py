from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="E-commerce Product Delivery Predictor",
    page_icon="📦",
    layout="wide",
)

project_folder = Path(__file__).parent
model_path = project_folder / "delivery_prediction_model.joblib"
orders_path = project_folder / "delivery_dashboard_data.csv"
importance_path = project_folder / "feature_importance.csv"
comparison_path = project_folder / "model_comparison.csv"

required_files = [model_path, orders_path, importance_path, comparison_path]
missing_files = [path.name for path in required_files if not path.exists()]
if missing_files:
    st.error("Missing project file(s): " + ", ".join(missing_files))
    st.stop()

model = joblib.load(model_path)
orders = pd.read_csv(orders_path)
feature_importance = pd.read_csv(importance_path)
model_comparison = pd.read_csv(comparison_path)

st.title("E-commerce Product Delivery Predictor")
st.caption("Explore the project charts or try a delivery prediction.")

prediction_tab, charts_tab = st.tabs(["Live prediction", "Project charts"])

with prediction_tab:
    st.write("Enter shipment details to estimate the delivery outcome.")
    st.caption("The displayed probabilities are model estimates, not guarantees.")

    with st.form("delivery_prediction_form"):
        left, right = st.columns(2)

        with left:
            warehouse = st.selectbox("Warehouse block", ["A", "B", "C", "D", "F"])
            shipment = st.selectbox("Mode of shipment", ["Flight", "Ship", "Road"])
            care_calls = st.number_input(
                "Customer care calls", min_value=0, max_value=10, value=3
            )
            product_cost = st.number_input(
                "Cost of the product", min_value=0, max_value=1000, value=200
            )
            prior_purchases = st.number_input(
                "Prior purchases", min_value=0, max_value=20, value=3
            )

        with right:
            importance = st.selectbox(
                "Product importance", ["low", "medium", "high"]
            )
            gender = st.selectbox("Gender", ["F", "M"])
            discount = st.number_input(
                "Discount offered", min_value=0, max_value=100, value=10
            )
            weight = st.number_input(
                "Weight in grams", min_value=0, max_value=10000, value=2500
            )

        submitted = st.form_submit_button("Predict delivery outcome")

    if submitted:
        order = pd.DataFrame([{
            "Warehouse_block": warehouse,
            "Mode_of_Shipment": shipment,
            "Customer_care_calls": care_calls,
            "Cost_of_the_Product": product_cost,
            "Prior_purchases": prior_purchases,
            "Product_importance": importance,
            "Gender": gender,
            "Discount_offered": discount,
            "Weight_in_gms": weight,
        }])

        prediction = int(model.predict(order)[0])
        class_values = list(model.named_steps["classifier"].classes_)
        late_index = class_values.index(1)
        late_probability = float(model.predict_proba(order)[0][late_index])
        on_time_probability = 1 - late_probability
        result = "Late" if prediction == 1 else "On time"

        st.subheader("Prediction result")
        first, second = st.columns(2)
        first.metric("Predicted delivery", result)
        second.metric("Estimated late probability", f"{late_probability:.1%}")
        st.caption(f"Estimated on-time probability: {on_time_probability:.1%}")

with charts_tab:
    st.header("Project charts")

    late_count = int((orders["Delivery_Status"] == "Late").sum())
    on_time_count = int((orders["Delivery_Status"] == "On time").sum())
    late_share = late_count / len(orders)

    total_metric, late_metric, share_metric = st.columns(3)
    total_metric.metric("Total orders", f"{len(orders):,}")
    late_metric.metric("Late orders", f"{late_count:,}")
    share_metric.metric("Late delivery share", f"{late_share:.1%}")

    top_left, top_right = st.columns(2)

    with top_left:
        st.subheader("Delivery outcome counts")
        outcome_counts = pd.DataFrame(
            {"Orders": [late_count, on_time_count]},
            index=["Late", "On time"],
        )
        st.bar_chart(outcome_counts, y="Orders")

    with top_right:
        st.subheader("Delivery outcomes by shipment mode")
        outcomes_by_mode = pd.crosstab(
            orders["Mode_of_Shipment"], orders["Delivery_Status"]
        ).reindex(columns=["Late", "On time"], fill_value=0)
        st.bar_chart(outcomes_by_mode, stack=False)

    bottom_left, bottom_right = st.columns(2)

    with bottom_left:
        st.subheader("Random Forest feature importance")
        feature_chart = feature_importance.copy()
        feature_chart["Feature"] = (
            feature_chart["Feature"]
            .str.replace("numbers__", "", regex=False)
            .str.replace("categories__", "", regex=False)
            .str.replace("_", " ", regex=False)
        )
        feature_chart["Feature"] = feature_chart["Feature"].replace({
            "Weight in gms": "Weight (g)",
            "Discount offered": "Discount",
            "Cost of the Product": "Product cost",
            "Customer care calls": "Care calls",
            "Warehouse block F": "Warehouse F",
            "Mode of Shipment Ship": "Shipment: Ship",
            "Warehouse block A": "Warehouse A",
            "Warehouse block C": "Warehouse C",
            "Gender F": "Gender: F",
        })
        feature_chart = feature_chart.sort_values("Importance", ascending=True)
        st.bar_chart(
            feature_chart,
            x="Feature",
            y="Importance",
            horizontal=True,
            sort=False,
        )

    with bottom_right:
        st.subheader("Late-delivery F1 score by model")
        score_chart = model_comparison.set_index("Model")[["F1-score (late)"]]
        st.bar_chart(score_chart, y="F1-score (late)")

    st.caption(
        "Feature importance describes how the Random Forest used the input features; "
        "it does not prove that a feature causes late delivery."
    )
