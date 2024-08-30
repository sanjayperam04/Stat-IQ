import os
import re
import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from groq import Groq
import plotly.express as px

# Set page config at the very beginning
st.set_page_config(layout="wide")

@st.cache_resource
def initialize_groq_client(api_key):
    """Initialize Groq client with API key."""
    return Groq(api_key=api_key)

client = initialize_groq_client(api_key="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") 

# Initialize conversation history for chat
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = [
        {
            "role": "system",
            "content": "You are a data analyst, data engineer, and business analyst."
        }
    ]

@st.cache_data
def convert_dataframe_types(df):
    """Ensure all DataFrame columns have consistent and appropriate types."""
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df

@st.cache_data
def load_data(file):
    """Load the dataset efficiently with caching."""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file)
    elif file.name.endswith('.json'):
        df = pd.read_json(file)
    else:
        return None

    df = convert_dataframe_types(df)
    return df

@st.cache_data
def identify_and_handle_mixed_types(df):
    """Identify columns with mixed types and handle them appropriately."""
    mixed_type_columns = []
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            mixed_type_columns.append(column)
    
    return df, mixed_type_columns

def get_response(user_query):
    """Get a response from Groq's model with retry logic and improved error handling."""
    st.session_state.conversation_history.append({
        "role": "user",
        "content": user_query
    })

    conversation_history = st.session_state.conversation_history[-10:]

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=conversation_history,
                model="Llama-3.1-70b-Versatile",
                temperature=0.5,
                max_tokens=1024,
                top_p=1,
                stop=None,
                stream=False,
            )
            assistant_response = response.choices[0].message.content
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": assistant_response
            })
            return assistant_response
        except Exception as e:
            if "rate limit" in str(e).lower():
                wait_time = 2 ** attempt
                st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            elif "invalid api key" in str(e).lower():
                st.error("Invalid API key. Please check your API key and try again.")
                return None
            else:
                st.error(f"Error: {e}")
                return None

def clean_code(code):
    """Extract only Python code from the response."""
    code_blocks = re.findall(r'```python\n(.*?)```', code, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    else:
        st.error("No Python code found in the response.")
        return ""

def execute_code(code, df):
    """Execute the dynamically generated code."""
    try:
        cleaned_code = clean_code(code)
        if not cleaned_code:
            st.error("No valid Python code to execute.")
            return None

        # Print the cleaned code for debugging
        st.write("Generated Code:")
        st.code(cleaned_code, language='python')

        # Execute the code
        local_vars = {"df": df, "st": st, "px": px, "np": np, "pd": pd}
        exec(cleaned_code, globals(), local_vars)

        # Return the cleaned DataFrame if it was modified
        return local_vars.get("df", df)
    except Exception as e:
        st.error(f"Execution error: {e}")
        return None

def generate_cleaning_code(data_description, mixed_type_columns):
    """Generate Python code for data cleaning and preprocessing."""
    prompt = f"""
    Based on the following data description, generate optimized Python code for data cleaning, Exploratory Data Analysis (EDA), and preprocessing. The code should be dynamic and scalable to handle the entire dataset. Prioritize key preprocessing steps and essential EDA techniques that will effectively support most data visualizations. Minimize unnecessary operations to ensure efficiency. The dataset is already loaded as a DataFrame named 'df'.
    
    Use st.cache_data for Streamlit and also show initial data shape and cleaned data shape too.
    
    The following columns have mixed data types and should be handled carefully:
    {mixed_type_columns}
    
    For these columns, consider the following approaches:
    1. If the column should be numeric but contains some string values, try to clean the data or convert valid entries to numeric, and handle or remove invalid entries.
    2. If the column is categorical or should remain as strings, convert the entire column to string type.
    3. If the column contains truly mixed data that can't be uniformly converted, consider creating multiple columns to separate the data types.

    Make sure to import and use numpy (as np) and pandas (as pd) in your code if needed.
    
    Data Description:
    {data_description}
    """
    code = get_response(prompt)
    return code

def generate_visualization_code(data_description):
    """Generate Python code for data visualization."""
    prompt = f"""
    Based on the cleaned dataset, generate Streamlit Python code to create a 'dataset name Dashboard' with 7 essential graphs and plots that fully summarize the dataset. The code should include various graph types like pie charts, bar graphs, histograms, and other relevant plots to provide a comprehensive overview. The layout should be inspired by a Power BI analytical dashboard, ensure a wide and aesthetically pleasing horizontal display of the graphs in Streamlit columns.
    Ensure that the generated dashboard is easy to interpret, even for someone with no knowledge of EDA or data analysis, effectively conveying the key insights from the dataset. Use only Plotly Express for the visualizations, and make sure the code is error-free. Analyze the clean dataset thoroughly before plotting.
    The dataset is already loaded as a DataFrame named 'df'.
    Do not use st.set_page_config() in the generated code.
    Use st.plotly_chart() to display the plots within the Streamlit app.
    """
    code = get_response(prompt)
    return code

def generate_business_recommendations(data_description):
    """Generate business recommendations based on the dataset."""
    prompt = f"""
    Based on the following data description, provide 10 business recommendations. These recommendations should be actionable and based on the insights derived from the dataset. Focus on key areas where improvements can be made, trends that can be leveraged, and strategies to optimize business operations.
    Data Description:
    {data_description}
    """
    response = get_response(prompt)
    recommendations = response.split('\n')
    return recommendations

# Streamlit UI
st.title("Stat IQ Dashboard")
st.write("Upload your dataset and let our model handle the analysis and visualization.")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "json"])

if uploaded_file is not None:
    # Load the data
    data = load_data(uploaded_file)

    if data is not None:
        # Identify and handle mixed type columns
        data, mixed_type_columns = identify_and_handle_mixed_types(data)

        # Create tabs for different sections of the dashboard
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Overview", "Data Analysis", "Visualizations", "Chatbot", "Business Recommendations"])

        with tab1:
            st.header("Data Overview")
            st.write(f"Number of rows: {data.shape[0]}")
            st.write(f"Number of columns: {data.shape[1]}")
            st.write("Data Sample:")
            st.write(data.head())
            st.subheader("Data Types")
            st.write(data.dtypes)
            st.subheader("Summary Statistics")
            st.write(data.describe())

        with tab2:
            st.header("Data Analysis")
            if st.button("Generate Cleaning and EDA Code"):
                data_description = data.describe(include='all').to_json()
                cleaning_code = generate_cleaning_code(data_description, mixed_type_columns)
                if cleaning_code:
                    st.write("Generated Data Cleaning and EDA Code:")
                    st.code(cleaning_code, language='python')

                    # Execute the cleaning code
                    cleaned_data = execute_code(cleaning_code, data)

                    if cleaned_data is not None:
                        st.session_state.cleaned_data = cleaned_data
                        st.success("Data cleaning and EDA completed. You can now proceed to the Visualizations tab.")

        with tab3:
            st.header("Visualizations")
            if 'cleaned_data' in st.session_state:
                if st.button("Generate Visualizations"):
                    data_description = st.session_state.cleaned_data.describe(include='all').to_json()
                    visualization_code = generate_visualization_code(data_description)
                    if visualization_code:
                        st.write("Generated Visualization Code:")
                        st.code(visualization_code, language='python')
                        execute_code(visualization_code, st.session_state.cleaned_data)
            else:
                st.warning("Please complete the Data Analysis step before generating visualizations.")

        with tab4:
            st.header("Stat-IQ GPT")
            st.write("Chat with your data and get personalized plots and graphs.")
            question = st.text_input("Ask a question or request a specific plot:")
            if st.button("Submit"):
                if question:
                    with st.spinner('Generating response...'):
                        response = get_response(question)
                        st.write("Response:", response)

                        # Check if response contains code
                        if "```python" in response:
                            cleaned_code = clean_code(response)
                            if cleaned_code:
                                execute_code(cleaned_code, data)
                else:
                    st.error("Please enter a question.")

        with tab5:
            st.header("Business Recommendations")
            if st.button("Generate Recommendations"):
                data_description = data.describe(include='all').to_json()
                recommendations = generate_business_recommendations(data_description)
                st.write("Business Recommendations:")
                for i, recommendation in enumerate(recommendations):
                    st.write(f"{recommendation}")

    else:
        st.error("Failed to load data from the uploaded file.")
