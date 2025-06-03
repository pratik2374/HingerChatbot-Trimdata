import uuid
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

# Set page config
st.set_page_config(
    page_title="CSV EDA Tool",
    page_icon="✂️",
    layout="wide"
)

# Custom CSS for card-based UI
st.markdown("""
<style>
.column-card {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 10px;
    margin: 5px;
    cursor: pointer;
    transition: all 0.3s ease;
}
.column-card:hover {
    background-color: #e0e2e6;
}
.selected {
    background-color: #4CAF50;
    color: white;
}
.stButton {
    text-align: center;
}
.stDownloadButton {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("CSV EDA Tool")
st.write("Upload a CSV file and select columns to keep")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Initialize session states
    if 'selected_columns' not in st.session_state:
        st.session_state.selected_columns = set()
    if 'column_renames' not in st.session_state:
        st.session_state.column_renames = {}
    if 'filter_refunds' not in st.session_state:
        filter_refunds = False
    if 'row_range' not in st.session_state:
        st.session_state.row_range = (0, len(df))
    if 'shuffle_df' not in st.session_state:
        st.session_state.shuffle_df = False
    
    # Display original dataframe
    st.subheader("Original DataFrame")
    st.dataframe(df)

     # Find columns with all empty rows
    empty_columns = [col for col in df.columns if df[col].isna().all()]
    if empty_columns:
        st.warning("The following columns have all empty rows:")
        for col in empty_columns:
            st.write(f"- {col}")
    
    # Download button for original dataframe
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Original CSV",
            data=csv,
            file_name=f"HB_original_{uuid.uuid4().hex[:8]}.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_original"
        )
    
    st.markdown("---")
    
    # Create columns for the card-based UI
    st.subheader("Select Columns to Keep")
    
    # Create a container for the cards
    cols = st.columns(4)  # Adjust number of columns as needed
    
    # Display column cards
    for i, col in enumerate(df.columns):
        col_idx = i % 4  # Distribute cards across columns
        with cols[col_idx]:
            is_selected = col in st.session_state.selected_columns
            card_class = "selected" if is_selected else ""
            
            # Create clickable card
            if st.button(
                f"{col}",
                key=f"btn_{col}",
                help=f"Click to {'deselect' if is_selected else 'select'} {col}",
                use_container_width=True,
                type="primary" if is_selected else "secondary"
            ):
                if is_selected:
                    st.session_state.selected_columns.remove(col)
                    # Remove rename if column is deselected
                    if col in st.session_state.column_renames:
                        del st.session_state.column_renames[col]
                else:
                    st.session_state.selected_columns.add(col)
                st.rerun()
    
    # Create filtered dataframe
    if st.session_state.selected_columns:
        # Get the selected columns
        selected_cols = list(st.session_state.selected_columns)
        filtered_df = df[selected_cols].copy()
        
        # Add row range selector
        st.markdown("---")
        st.subheader("Select Row Range")
        
        # Create slider for row range
        min_val, max_val = st.slider(
            "Select rows to keep",
            min_value=0,
            max_value=len(filtered_df),
            value=st.session_state.row_range,
            help="Drag the handles to select the range of rows to keep"
        )
        
        # Update session state
        st.session_state.row_range = (min_val, max_val)
        
        # Apply row range filter
        filtered_df = filtered_df.iloc[min_val:max_val]
        
        # Display row count info
        st.info(f"Showing rows {min_val} to {max_val} (Total: {len(filtered_df)} rows)")
        
        # Add shuffle checkbox
        st.session_state.shuffle_df = st.checkbox(
            "Shuffle the dataframe",
            value=st.session_state.shuffle_df,
            key="st.session_state.shuffle_df"
        )
        
        # Apply shuffle if checked
        if st.session_state.shuffle_df:
            filtered_df = filtered_df.sample(frac=1, random_state=42).reset_index(drop=True)
            st.session_state.shuffle_df = True
            st.info("Dataframe has been shuffled")
            st.session_state.shuffle_df = False
        else:
            st.session_state.shuffle_df = False
        
        # Apply renames
        rename_dict = {old: new for old, new in st.session_state.column_renames.items() 
                      if old in selected_cols}
        if rename_dict:
            filtered_df = filtered_df.rename(columns=rename_dict)
        
        # Display filtered dataframe
        st.subheader("Filtered DataFrame")
        st.dataframe(filtered_df)
        
        # Download button for filtered dataframe
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download Filtered CSV",
                data=csv,
                file_name=f"HB_filtered_{uuid.uuid4().hex[:8]}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_filtered"
            )
        
        # Add rename interface
        st.markdown("---")  # Add a separator
        st.subheader("Rename")
        
        # Create three columns for the rename interface
        rename_col1, rename_col2, rename_col3 = st.columns([2, 1, 2])
        
        with rename_col1:
            # Dropdown for selecting column to rename
            column_to_rename = st.selectbox(
                "Select column to rename",
                options=filtered_df.columns,
                key="column_selector"
            )
        
        with rename_col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            # Rename button
            if st.button("Rename →", use_container_width=True):
                if 'new_column_name' in st.session_state:
                    new_name = st.session_state.new_column_name
                    if new_name and new_name != column_to_rename:
                        st.session_state.column_renames[column_to_rename] = new_name
                        st.rerun()
        
        with rename_col3:
            # Text input for new name
            new_name = st.text_input(
                "Enter new name",
                value=st.session_state.column_renames.get(column_to_rename, column_to_rename),
                key="new_column_name"
            )
        
        # Display final dataframe with renames
        if st.session_state.column_renames:
            st.subheader("Final DataFrame with Renamed Columns")
            final_df = filtered_df.copy()
            final_df = final_df.rename(columns=st.session_state.column_renames)
            st.dataframe(final_df)
            
            # Download button for final dataframe
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                csv = final_df.to_csv(index=False)
                st.download_button(
                    label="Download Final CSV",
                    data=csv,
                    file_name=f"HB_final_{uuid.uuid4().hex[:8]}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_final"
                )
        
        # Add refund filter checkbox at the end
        st.markdown("---")
        st.subheader("Additional Filters")
        filter_refunds = st.checkbox(
            "Remove rows where Refunded Amount > Refundable Amount",
            key="filter_refunds"
        )
        
        # Display info message and filtered data if checkbox is checked
        if filter_refunds:
            if 'Refunded Amount' in filtered_df.columns and 'Refundable Amount' in filtered_df.columns:
                # Store original row count
                original_rows = len(filtered_df)
                
                # Apply filter
                filtered_df = filtered_df[filtered_df['Refunded Amount'] <= filtered_df['Refundable Amount']]
                filter_refunds = True
                
                # Calculate rows removed
                rows_removed = original_rows - len(filtered_df)
                
                # Display rows removed information
                st.info(f"Removed {rows_removed} rows where Refunded Amount > Refundable Amount")
                
                # Display filtered dataframe
                st.subheader("Final Filtered DataFrame")
                st.dataframe(filtered_df)
                
                # Download button for filtered dataframe
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download Filtered CSV",
                        data=csv,
                        file_name=f"HB_filtered_{uuid.uuid4().hex[:8]}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key="download_refund_filtered"
                    )
            else:
                st.warning("Both 'Refunded Amount' and 'Refundable Amount' columns must be selected to apply this filter.")
                filter_refunds = False
    else:
        st.info("Select columns to keep from the cards above")
