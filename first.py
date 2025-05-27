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

# Set OpenAI API key from secrets
os.environ['OPENAI_API_KEY'] = st.secrets["openai_api_key"]

# Set page config
st.set_page_config(
    page_title="Data Analysis Chatbot",
    page_icon="🤖",
    layout="centered"  # Options are: "centered", "wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title and description
st.title("📊 Data Analysis Chatbot")
st.markdown("Ask questions about your data and get instant insights!")

# Sidebar instructions
with st.sidebar:
    st.header("📁 Data Upload")
    st.markdown("""
    ### Instructions:
    1. Upload your CSV or Excel file using the file uploader below
    2. Supported formats: CSV (.csv) and Excel (.xls, .xlsx)
    3. Make sure your data is clean and properly formatted
    4. Maximum file size: 200MB
    
    ### Tips:
    - For best results, ensure your data has clear column headers
    - Remove any unnecessary columns before uploading
    - Check for missing values in your dataset
    """)
    # File uploader with size limit
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB in bytes
    file = st.file_uploader(
        "Upload your data file",
        type=["csv", "xls", "xlsx"],
        help="Upload a CSV or Excel file to analyze"
    )

def load_df(file):
    if file is None:
        st.warning("Please upload a file to begin analysis.")
        st.stop()
    
    try:
        # Check file size
        file_size = file.size
        if file_size > MAX_FILE_SIZE:
            st.error(f"File size exceeds the maximum limit of 200MB. Current size: {file_size/1024/1024:.2f}MB")
            st.stop()
        
        # Read file based on extension
        file_extension = file.name.split('.')[-1].lower()
        if file_extension == 'csv':
            df = pd.read_csv(file)
        elif file_extension in ['xls', 'xlsx']:
            print("hello")
            df = pd.read_excel(file)
            print("hello2")
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.stop()
        
        # Basic data validation
        if df.empty:
            st.error("The uploaded file is empty.")
            st.stop()
            
        # Save DataFrame to dataset directory if not present
        os.makedirs('dataset', exist_ok=True)
        dataset_path = os.path.join('dataset', f"{os.path.splitext(file.name)[0]}.{file_extension}")
        if not os.path.exists(dataset_path):
            df.to_csv(dataset_path, index=False)
            
        # Display basic data info
        st.sidebar.success(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        st.sidebar.markdown("### Data Preview")
        st.sidebar.dataframe(df.head(3))
        
        return df
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.stop()

# Load the data and index
@st.cache_data
def load_data():
    try:
        st.session_state.storage_context = StorageContext.from_defaults(
            persist_dir=f"./storage/sample.{os.getpid()}"
        )
        st.session_state.supa_index = load_index_from_storage(st.session_state.storage_context)


        index_loaded = True
    except:
        index_loaded = False

    if not index_loaded:
        print("Loading data...")
        # load data
        supa_docs = SimpleDirectoryReader(
            input_dir="dataset"
        ).load_data()

        # build index
        st.session_state.supa_index = VectorStoreIndex.from_documents(supa_docs)

    # persist index
    st.session_state.supa_index.storage_context.persist(persist_dir=f"./storage/sample.{os.getpid()}")

    return st.session_state.supa_index

    

# Load data
df = load_df(file)
query_engine = PandasQueryEngine(df=df, verbose=False, synthesize_response=True)
st.session_state.supa_index = load_data()
Supa_Engine = st.session_state.supa_index.as_query_engine(similarity_top_k=3)

query_engine_tools = [
    QueryEngineTool.from_defaults(
        query_engine=Supa_Engine,
        name="Sales-Analyser",
        description=(
            "Provides information about sales data of a company profit and sales "
            "Use a detailed plain text question as input to the tool."
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

agent = FunctionAgent(
    tools=tools,
    llm=OpenAI(model="gpt-4o"), 
    system_prompt= """
        You are an advanced Data Analysis Assistant with expertise in both pandas data analysis and semantic search capabilities. Your primary responsibilities are:

        1. Data Analysis using panda_retriver:
        - Process natural language queries about the uploaded dataset
        - Perform statistical analysis, filtering, and data manipulation
        - Provide clear explanations of your analysis results
        - Handle both simple and complex analytical queries

        2. Semantic Search using query_engine_tools(name: "Sales-Analyser"):
        - Understand and process context-based queries
        - Retrieve relevant information from the vector store
        - Provide comprehensive answers based on the available data
        - Maintain context awareness in conversations

        3. Response Guidelines:
        - Always provide clear, concise, and accurate responses
        - Include relevant data points and statistics when applicable
        - Explain your reasoning and methodology
        - Format numerical results appropriately
        - Use markdown formatting for better readability

        4. Error Handling:
        - If a Pandas Query Engine is not able to answer the question, then use the query_engine_tools tool to answer the question by searching the data in the vector store.
        - Gracefully handle ambiguous or unclear queries
        - Provide helpful suggestions when queries cannot be processed
        - Guide users to rephrase questions when needed

        5. Best Practices:
        - Prioritize accuracy over speed
        - Maintain professional and helpful tone
        - Consider data privacy and security
        - Provide context-aware responses

        Remember to:
        - Use the appropriate tool (panda_retriver or similarity_retrive) based on the query type
        - Combine insights from both tools when necessary
        - Always verify the accuracy of your responses
        - Provide step-by-step explanations for complex analyses
    """
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
            st.write(str(response))
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



