import streamlit as st
import pandas as pd
import pandas.io.excel as excel
import pandas.io.excel._base
import pandas.io.excel._odswriter
import pandas.io.excel._openpyxl
import pandas.io.excel._util
import pandas.io.excel._xlsxwriter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent.workflow import FunctionAgent, ToolCallResult, AgentStream
from llama_index.core.workflow import Context
import os
import asyncio
import matplotlib.pyplot as plt
import seaborn as sns
import io
import re
import datetime

# Set OpenAI API key from secrets
os.environ['OPENAI_API_KEY'] = st.secrets["openai_api_key"]

# Set page config
st.set_page_config(
    page_title="HungerBot Chatbot",
    page_icon="🤖",
    layout="centered"  # Options are: "centered", "wide"
)

st.sidebar.title("🍽️ HungerBot")
st.sidebar.markdown("#### An Intelligent AI Assistant for Sales Data")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🔍 Features")
st.sidebar.markdown(
    """
- Chat with your data  
- Get smart insights  
- Generate visualizations  
"""
)

st.sidebar.markdown("---")

st.sidebar.markdown("### 💬 Try These Examples")
st.sidebar.code(
    '''- describe the data
- Give me a bar plot on "Employee paid" by "source device"
- Plot a bar chart for the day wise orders
- Time at which highest sales was made on 4/25/2025?
- List the top five vendors by orders'''
)

st.sidebar.markdown("---")


# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("📊 Data Analysis Chatbot")
st.markdown("Ask questions about your data and get instant insights!")

def load_df(file_path):
    try:
        file_extension = file_path.split('.')[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(file_path, low_memory=False)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(file_path)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.stop()
        
        # Basic data validation
        if df.empty:
            st.error("The uploaded file is empty.")
            st.stop()
            
            
        # Display basic data info
        # st.sidebar.success(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        st.sidebar.markdown("### Data Preview")

        # --- Robust datetime handling ---
        # Try to find a datetime column
        datetime_col = None
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_col = col
                break
            # Try to parse if column name suggests datetime
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    datetime_col = col
                    break
                except Exception:
                    continue

        if datetime_col:
            df['Order_date'] = df[datetime_col].dt.strftime('%Y-%m-%d')
            df['Order_clock_time'] = df[datetime_col].dt.strftime('%H:%M:%S')
            # Optionally drop the original datetime column
            # df = df.drop(columns=[datetime_col])
        else:
            st.sidebar.info("No datetime column detected. Date/time features will be unavailable.")

        st.sidebar.dataframe(df.head(3))
        
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.stop()

# Load the data and index
def load_data():
    try:
        st.session_state.storage_context = StorageContext.from_defaults(
            persist_dir='storage/sample.20652'
        )
        st.session_state.supa_index = load_index_from_storage(st.session_state.storage_context)
    except Exception as e:
        st.error(f"Error loading index: {e}")
        st.session_state.supa_index = None

    return st.session_state.supa_index


    

# Load data
df = load_df('dataset/data.csv')


query_engine = PandasQueryEngine(
    df=df,
    verbose=False,
    synthesize_response=True,
    )

if "supa_index" not in st.session_state:
    st.session_state.supa_index = load_data()

if st.session_state.supa_index is not None:
    Supa_Engine = st.session_state.supa_index.as_query_engine(similarity_top_k=3)
else:
    st.error("Could not load the index. Please check if the path is correct and the storage is available.")
    st.stop()


Supa_Engine = st.session_state.supa_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine=Supa_Engine,
        name="Sales-Analyser",
        description=(
            "Use this tool to answer questions about the uploaded tabular data. "
            "It can perform statistical analysis, filtering, aggregation, and generate insights. "
            "Always use the actual column names from the provided DataFrame."
        ),
    )
]

def panda_retriver(query: str) -> str:
    """
    Process a natural language query and retrieve insights from the pandas DataFrame using the PandasQueryEngine.

    This function takes a natural language query and uses the PandasQueryEngine to:
    1. Convert the query into pandas operations
    2. Execute the operations on the loaded DataFrame
    3. Synthesize a human-readable response

    Args:
        query (str): A natural language query about the data, such as statistical questions,
                    data filtering requests, or analytical queries.

    Returns:
        str: A string containing the processed response from the PandasQueryEngine,
             which includes both the analysis results and a natural language explanation.

    Note:
        - The function uses the global query_engine instance configured with verbose=True
        - Responses are synthesized for better readability
        - The query should be related to the data in the loaded DataFrame
    """

    ## Todo : Subquestion Query to retrive from all
    response = query_engine.query(query)
    return str(response)


tools = query_engine_tools + [panda_retriver]

def get_column_summary(df: pd.DataFrame, n_examples: int = 2) -> str:
    """Generate a summary of columns, types, and example values for prompt."""
    summary = []
    for col in df.columns:
        col_type = str(df[col].dtype)
        examples = df[col].dropna().unique()[:n_examples]
        summary.append(
            f'- "{col}": type={col_type}, examples={list(examples)}'
        )
    return "\n".join(summary)

def get_column_names(df: pd.DataFrame) -> str:
    """Return a comma-separated string of column names."""
    return ", ".join([f'"{col}"' for col in df.columns])

def is_visualization_query(query: str) -> bool:
    """Detect if the query is asking for a visualization."""
    viz_keywords = [
        "plot", "chart", "graph", "visualize", "visualization", "draw", "histogram", "bar", "line", "scatter", "pie"
    ]
    return any(word in query.lower() for word in viz_keywords)

def render_visualization(df: pd.DataFrame, query: str):
    """
    Generate a visualization based on the query using pandas/seaborn/matplotlib.
    Handles numeric and categorical data robustly and asks for clarification if needed.
    """
    try:
        # Bar plot: try to find a categorical x and numeric y, or count if y is not numeric
        if "bar" in query or "bar plot" in query or "bar chart" in query:
            # Try to find a categorical column for x
            x_col = next((col for col in df.columns if df[col].dtype == 'object' or 'cat' in str(df[col].dtype)), df.columns[0])
            # Try to find a numeric column for y
            y_col = next((col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])), None)

            if y_col:
                # If numeric y, aggregate by sum
                agg_df = df.groupby(x_col)[y_col].sum().reset_index()
                plt.figure(figsize=(8, 4))
                sns.barplot(x=agg_df[x_col], y=agg_df[y_col])
                plt.ylabel(y_col)
                plt.xlabel(x_col)
                plt.title(f"{y_col} by {x_col}")
                plt.xticks(rotation=45)
            else:
                # If no numeric y, ask for clarification before plotting counts
                st.info(f"You asked for a bar plot of '{x_col}', but there is no numeric column to plot. "
                        "Would you like to see a count plot (number of occurrences for each category)?")
                # Optionally, you could add a button for the user to confirm
                if st.button(f"Show count plot for {x_col}"):
                    counts = df[x_col].value_counts()
                    plt.figure(figsize=(8, 4))
                    sns.barplot(x=counts.index, y=counts.values)
                    plt.ylabel("Count")
                    plt.xlabel(x_col)
                    plt.title(f"Count by {x_col}")
                    plt.xticks(rotation=45)
                else:
                    return
        elif "hist" in query or "histogram" in query:
            col = next((col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])), None)
            if col:
                plt.figure(figsize=(8, 4))
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f"Histogram of {col}")
            else:
                st.error("No numeric column found for histogram.")
                return
        elif "scatter" in query or "scatter plot" in query:
            num_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            if len(num_cols) >= 2:
                plt.figure(figsize=(8, 4))
                sns.scatterplot(data=df, x=num_cols[0], y=num_cols[1])
                plt.title(f"Scatter plot of {num_cols[0]} vs {num_cols[1]}")
            else:
                st.error("Not enough numeric columns for scatter plot.")
                return
        else:
            st.info("Visualization type not recognized or not explicitly requested. Please specify the type of plot you want.")
            return
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format="png")
        st.image(buf)
        plt.close()
    except Exception as e:
        st.error(f"Could not generate visualization: {e}")

# In your main prompt, dynamically describe the data:
def get_system_prompt(df: pd.DataFrame) -> str:
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""
You are HungerBox's advanced Data Analysis Assistant, designed to help users analyze and gain insights from complex sales and operational data. You are integrated into a Retrieval-Augmented Generation (RAG) system and have access to the user's uploaded data as a pandas DataFrame named `df`.

Current date and time: {current_datetime}

Your core responsibilities:
- **Data Fidelity:** Always use the provided DataFrame `df` for all analysis, calculations, and visualizations. Never create or load a new DataFrame or use sample data.
- **Column Awareness:** Only use the actual column names and data types present in `df`. If you are unsure about a column name, refer to the provided list of columns.
- **Business Context:** Assume the data is related to sales, orders, transactions, customers, vendors, products, payments, and other business operations typical for a food-tech company like HungerBox. However, always adapt to the actual columns and data provided.
- **Directness:** Always answer the user's question directly and concisely. If the question is ambiguous or cannot be answered, ask for clarification or explain why.
- **Details:** Always answer the user's question directly, **with a detailed, step-by-step explanation** of your reasoning and findings. For every response, include a clear summary of what the result means in a business context, and how the user can interpret or act on it.
- **Visualization:** 
    - Only generate a visualization if the user explicitly requests it (e.g., plot, chart, graph, visualization, histogram, bar, line, scatter, pie) or if the question strongly implies a need for visual representation (e.g., trend, distribution, compare, relationship).
    - If a visualization is required, return only the Python code (in a code block) that generates the requested plot using matplotlib or seaborn, and then provide a concise explanation of what the chart shows. Do not explain the code itself.
    - If the user does not ask for a visualization, provide only a direct, concise, and complete answer to their question, with no code block.
    - If a visualization is not possible, explain why.
- **Analysis:** 
    - Support a wide range of queries: aggregations, trends, comparisons, filtering, segmentation, cohort analysis, time series, outlier detection, and more.
    - For time-based queries, automatically detect and use the appropriate datetime column.
    - For categorical analysis, use the relevant columns (e.g., product, vendor, location, status).
    - For numeric analysis, use columns such as sales, quantity, value, amount, etc.
- **Currency:** Always display and format all prices, sales, and monetary values in Indian Rupees (₹). If a value is a price or amount, clearly indicate it is in rupees.
- **Clarity & Formatting:** 
    - Format all responses in markdown for readability.
    - Use tables, bullet points, and clear headings where appropriate.
    - Never include code that creates or loads a DataFrame; always use the existing `df` variable.
- **Error Handling:** 
    - If the question is ambiguous, ask for clarification.
    - If a column or data is missing, explain the limitation.
    - If an error occurs, provide a helpful message.
- **Privacy:** Never reveal or infer sensitive information beyond what is present in the data.

**Data context:**
- Number of rows: {len(df)}, Number of columns: {len(df.columns)}
- Column names: {get_column_names(df)}
- Columns and types:
{get_column_summary(df)}
- Sample data:
{df.head(2).to_markdown()}
"""

def run_and_render_code_from_response(response: str):
    """
    Detects Python code blocks in the response, executes them, and renders any matplotlib plots.
    Shows the rest of the response as markdown.
    """
    code_blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", response)
    non_code = re.sub(r"```(?:python)?\s*[\s\S]*?```", "", response).strip()
    
    # Run each code block
    for code in code_blocks:
        try:
            plt.close('all')
            exec_globals = {
                "plt": plt,
                "sns": sns,
                "pd": pd,
                "df": df,
            }
            exec(code, exec_globals)
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png")
            st.image(buf)
            plt.close()
        except Exception as e:
            st.error(f"Error running code snippet: {e}")
    
    if non_code:
        st.markdown(non_code)


# Update agent initialization to use the new dynamic prompt
agent = FunctionAgent(
    tools=tools,
    llm=OpenAI(model="gpt-4o"), 
    system_prompt=get_system_prompt(df)
)

st.session_state.ctx = Context(agent)
async def main(prompt):
    handler = agent.run(prompt, ctx=st.session_state.ctx, stepwise=False)

    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
            print(
                f"Call {ev.tool_name} with args {ev.tool_kwargs}\nReturned: {ev.tool_output}"
            )
        elif isinstance(ev, AgentStream):
            print(ev.delta, end="", flush=True)

    response = await handler
    return response

async def process_chat(prompt):
    try:
        # Run the agent with proper error handling
        response = await main(prompt=prompt)
        if response is None:
            st.error("No response generated. Please try rephrasing your question.")
        else:
            run_and_render_code_from_response(str(response))
            st.session_state.messages.append({"role": "assistant", "content": str(response)})
    except Exception as e:
        error_message = f"Error processing your query: {str(e)}"
        st.error(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})

# Display chat messages
if __name__ == "__main__":
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your data"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                asyncio.run(process_chat(prompt))



