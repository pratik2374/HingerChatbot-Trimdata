# Data Analysis Chatbot 🤖

An advanced data analysis chatbot that combines the power of pandas data analysis with semantic search capabilities. Built with Streamlit and powered by OpenAI's GPT-4, this tool allows users to interact with their data through natural language queries.

## Features

- **Natural Language Data Analysis**: Ask questions about your data in plain English
- **Multi-Format Support**: Upload and analyze both CSV and Excel files
- **Advanced Data Processing**:
  - Automatic date/time parsing
  - Data validation and cleaning
  - Intelligent column handling
- **Dual Analysis Engine**:
  - Pandas-based data analysis
  - Semantic search capabilities
- **Interactive Chat Interface**: Real-time responses with markdown formatting
- **Data Persistence**: Automatic storage and indexing of uploaded datasets

## Prerequisites

- Python 3.x
- OpenAI API key
- Required Python packages:
  ```bash
  pip install streamlit pandas llama-index openai
  ```

## Installation

1. Clone the repository
2. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
3. Set up your OpenAI API key in Streamlit secrets:
  - Create a `.streamlit/secrets.toml` file
  - Add your API key: `openai_api_key = "your-api-key-here"`

## Workflow

1. Implement EDA on Dataset manually or you can use [this tool](https://csvtrimapp.streamlit.app/)

2. Rename your columns as follows : 
  - `"Created At"` or any other date/time column → **Order_time**
  - `"Status"` → **Delivery_Status**
  - `"Total Value"` → **Sales**

3. Set up your OpenAI API key in Streamlit secrets:
  - Create a `.streamlit/secrets.toml` file
  - Add your API key: `openai_api_key = "your-api-key-here"`

4. Run the application:
  ```bash
  streamlit run first.py
  ```

5. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

6. Upload your data and wait some time for processing:
   - Supported formats: CSV (.csv), Excel (.xls, .xlsx)
   - Maximum file size: 200MB
   - Ensure your data has clear column headers

7. Start asking questions about your data:
   - Statistical analysis
   - Data filtering
   - Trend analysis
   - Custom queries

## Data Processing

The chatbot automatically processes your data with the following features:
- Converts date/time columns to appropriate formats
- Handles missing values
- Creates derived columns for better analysis
- Maintains data integrity throughout the analysis

## Analysis Capabilities

### Pandas Analysis
- Statistical calculations
- Data filtering and grouping
- Time series analysis
- Custom aggregations

### Semantic Search
- Context-aware queries
- Related information retrieval
- Comprehensive data exploration
- Historical data analysis

## File Structure

- `first.py`: Main application file
- `dataset/`: Directory for storing uploaded datasets
- `storage/`: Directory for vector store and indices
- `.streamlit/secrets.toml`: Configuration file for API keys

## Notes

- The chatbot uses GPT-4 for advanced natural language processing
- All data processing is done in real-time
- Results are cached for better performance
- The interface is responsive and works on both desktop and mobile devices

## Security

- API keys are stored securely in Streamlit secrets
- File size limits prevent memory issues
- Data validation ensures safe processing

## License

This project is open source and available under the MIT License. 