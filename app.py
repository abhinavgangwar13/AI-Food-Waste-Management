import streamlit as st
from model import load_data, prepare_dataset, train_model, estimate_surplus, build_recommendations

st.set_page_config(page_title="Food Waste Prediction", page_icon="🍽️", layout="wide")

st.title("AI-Powered Food Waste Prediction and Redistribution Intelligence")
st.caption("Prototype for reducing surplus food through demand forecasting and redistribution recommendations")

@st.cache_data
def get_data():
    df = load_data("data/food_data.csv")
    return prepare_dataset(df)

df = get_data()

@st.cache_data
def get_model():
    return train_model(df)

model = get_model()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Records", len(df))
with col2:
    st.metric("Average Demand", round(df["demand"].mean(), 2))
with col3:
    st.metric("Average Surplus", round((df["supply"] - df["demand"]).mean(), 2))

st.subheader("Demand Forecasting")
restaurant = st.selectbox("Select restaurant", sorted(df["restaurant"].unique()))
food_type = st.selectbox("Select food type", sorted(df["food_type"].unique()))

filtered = df[(df["restaurant"] == restaurant) & (df["food_type"] == food_type)]
if filtered.empty:
    st.warning("No matching data found.")
    st.stop()

latest = filtered.iloc[-1]

prediction = model.predict([[
    latest["day_of_week"],
    latest["month"],
    latest["holiday"],
    latest["weather_score"],
    latest["inventory"],
    latest["supply"]
]])[0]

surplus = estimate_surplus(latest["supply"], prediction)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Supply", f"{latest['supply']} units")
with col2:
    st.metric("Predicted Demand", f"{prediction:.2f} units")
with col3:
    st.metric("Estimated Surplus", f"{surplus:.2f} units")

st.subheader("Redistribution Recommendation")
if surplus > 0:
    st.success(f"Redistribute {surplus:.2f} units to a local shelter or kitchen.")
else:
    st.info("No immediate surplus detected.")

st.subheader("Historical Data")
st.dataframe(filtered.tail(10), use_container_width=True)

st.subheader("Project Goal")
st.markdown("""
This project predicts food demand and identifies surplus food before it becomes waste.
It supports better inventory planning and reduces avoidable food waste.
""")
