import streamlit as st
import pandas as pd
import plotly.express as px

# Set Streamlit page title
st.set_page_config(page_title="News Topic Treemap", layout="wide")

# Sample Data
data = {
    "topic": [
        "Corporate Strategy",
        "Regulatory & Legal Issues",
        "Competitor & Market Trends",
        "Technology & Innovation"
    ],
    "count": [10, 1, 1, 3]
}

# Create DataFrame
df = pd.DataFrame(data)

# Streamlit App Title
st.title("📊 News Article Distribution by Topic")


# Create Treemap using Plotly
fig = px.treemap(df, 
                 path=["topic"], 
                 values="count", 
                 title="Treemap of News Topics",
                 color="count",
                 color_continuous_scale='Viridis')  # Dynamic color scheme

# Display Treemap in Streamlit
st.plotly_chart(fig, use_container_width=True)

