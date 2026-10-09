import streamlit as st

st.set_page_config(page_title="PhonePe AI Project")

st.title("PhonePe AI/ML Project")
st.success("Your Streamlit app is working!")

st.write("This is my AI-based digital payment forecasting assignment.")

st.subheader("Dataset")
st.write("Source: Official PhonePe Pulse public dataset")

st.link_button(
    "Open PhonePe Pulse Dataset",
    "https://github.com/PhonePe/pulse"
)

st.info("Next, we will connect the machine learning model.")
