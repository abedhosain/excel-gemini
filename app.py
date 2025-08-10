import streamlit as st
import json
import os
from typing import List, Dict, Any, Optional, Tuple
import openpyxl
from openpyxl import load_workbook
import google.generativeai as genai
import tempfile
import logging
from io import BytesIO
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="📊 Excel AI Analyzer with TF-IDF",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TFIDFSearchEngine:
    """TF-IDF based search engine for Excel data"""
    
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        self.row_metadata = []
        
    def build_index(self, files_data: List[Dict[str, Any]]) -> None:
        """Build TF-IDF index from all Excel data"""
        self.documents = []
        self.row_metadata = []
        
        for file_data in files_data:
            if 'error' in file_data:
                continue
                
            file_name = file_data['file_name']
            
            for sheet_name, sheet_data in file_data['sheets'].items():
                if 'error' in sheet_data:
                    continue
                
                # Process each row as a document
                for row_idx, row in enumerate(sheet_data['data']):
                    # Combine all cell values into a single text document
                    text_content = []
                    for header, value in row.items():
                        if value and str(value).strip():
                            text_content.append(f"{header}: {str(value)}")
                    
                    if text_content:
                        document = " | ".join(text_content)
                        self.documents.append(document)
                        
                        # Store metadata for each row
                        self.row_metadata.append({
                            'file_name': file_name,
                            'sheet_name': sheet_name,
                            'row_index': row_idx,
                            'row_data': row,
                            'headers': list(row.keys())
                        })
        
        if self.documents:
            # Build TF-IDF matrix
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant rows using TF-IDF similarity"""
        if not self.vectorizer or not self.documents:
            return []
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include relevant results
                result = self.row_metadata[idx].copy()
                result['similarity_score'] = float(similarities[idx])
                result['matched_text'] = self.documents[idx]
                results.append(result)
        
        return results
    
    def get_column_data(self, column_name: str, file_name: str = None, sheet_name: str = None) -> List[Dict[str, Any]]:
        """Get all data from a specific column"""
        results = []
        
        for metadata in self.row_metadata:
            # Filter by file/sheet if specified
            if file_name and metadata['file_name'] != file_name:
                continue
            if sheet_name and metadata['sheet_name'] != sheet_name:
                continue
            
            # Check if column exists and has data
            row_data = metadata['row_data']
            if column_name in row_data and row_data[column_name]:
                result = {
                    'file_name': metadata['file_name'],
                    'sheet_name': metadata['sheet_name'],
                    'row_index': metadata['row_index'],
                    'column_name': column_name,
                    'value': row_data[column_name],
                    'full_row': row_data
                }
                results.append(result)
        
        return results

class ExcelProcessor:
    """Enhanced Excel file processor with TF-IDF capabilities"""
    
    def __init__(self):
        self.max_rows_per_sheet = 100
        self.max_chars_per_cell = 500
        self.search_engine = TFIDFSearchEngine()
        
    def read_excel_file(self, file_content: bytes, file_name: str) -> Dict[str, Any]:
        """Read Excel file from bytes content"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            workbook = load_workbook(tmp_path, data_only=True)
            
            file_data = {
                'file_name': file_name,
                'sheets': {},
                'summary': {
                    'total_sheets': len(workbook.sheetnames),
                    'sheet_names': workbook.sheetnames
                }
            }
            
            for sheet_name in workbook.sheetnames:
                sheet_data = self._process_sheet(workbook[sheet_name], sheet_name)
                file_data['sheets'][sheet_name] = sheet_data
            
            try:
                os.unlink(tmp_path)
            except:
                pass
                
            return file_data
            
        except Exception as e:
            logger.error(f"Error reading Excel file {file_name}: {str(e)}")
            return {'error': f"Failed to read {file_name}: {str(e)}"}
    
    def _process_sheet(self, sheet, sheet_name: str) -> Dict[str, Any]:
        """Process individual sheet with enhanced data capture"""
        try:
            if sheet.max_row is None or sheet.max_row == 0:
                return {
                    'sheet_name': sheet_name,
                    'summary': {'total_rows': 0, 'total_columns': 0, 'headers': [], 'sample_data': []},
                    'data': []
                }
            
            max_row = min(sheet.max_row, self.max_rows_per_sheet + 1)
            max_col = sheet.max_column or 0
            
            # Extract headers with better cleaning
            headers = []
            for col in range(1, max_col + 1):
                cell_value = sheet.cell(row=1, column=col).value
                if cell_value is not None:
                    header = str(cell_value).strip()
                    # Clean header name
                    header = re.sub(r'[^\w\s]', '', header)
                    header = re.sub(r'\s+', ' ', header)
                    headers.append(header if header else f"Column_{col}")
                else:
                    headers.append(f"Column_{col}")
            
            # Extract data rows with better preservation
            data_rows = []
            for row in range(2, max_row + 1):
                row_data = {}
                has_data = False
                
                for col, header in enumerate(headers, 1):
                    cell_value = sheet.cell(row=row, column=col).value
                    
                    if cell_value is not None:
                        cell_str = str(cell_value).strip()
                        # Preserve more content but limit for token efficiency
                        if len(cell_str) > self.max_chars_per_cell:
                            cell_str = cell_str[:self.max_chars_per_cell] + "..."
                        row_data[header] = cell_str
                        has_data = True
                    else:
                        row_data[header] = ""
                
                if has_data:
                    # Add row identifier
                    row_data['_row_number'] = row - 1  # 1-based numbering
                    data_rows.append(row_data)
            
            # Enhanced summary with column analysis
            summary = {
                'total_rows': len(data_rows),
                'total_columns': len(headers),
                'headers': headers,
                'data_types': self._analyze_data_types(data_rows, headers),
                'column_stats': self._get_column_stats(data_rows, headers),
                'sample_data': data_rows[:5] if data_rows else []  # Show more samples
            }
            
            return {
                'sheet_name': sheet_name,
                'summary': summary,
                'data': data_rows
            }
            
        except Exception as e:
            logger.error(f"Error processing sheet {sheet_name}: {str(e)}")
            return {'error': f"Failed to process sheet {sheet_name}: {str(e)}"}
    
    def _analyze_data_types(self, data_rows: List[Dict], headers: List[str]) -> Dict[str, str]:
        """Enhanced data type analysis"""
        data_types = {}
        
        for header in headers:
            if header.startswith('_'):  # Skip internal fields
                continue
                
            sample_values = [str(row.get(header, "")).strip() for row in data_rows[:20] if row.get(header, "")]
            
            if not sample_values:
                data_types[header] = "empty"
                continue
            
            # Enhanced type detection
            numeric_count = 0
            date_count = 0
            
            for val in sample_values:
                # Check numeric
                if re.match(r'^-?\d+\.?\d*$', val.replace(',', '')):
                    numeric_count += 1
                # Check date patterns
                elif re.match(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}', val):
                    date_count += 1
            
            total_samples = len(sample_values)
            if numeric_count > total_samples * 0.7:
                data_types[header] = "numeric"
            elif date_count > total_samples * 0.5:
                data_types[header] = "date"
            else:
                data_types[header] = "text"
                
        return data_types
    
    def _get_column_stats(self, data_rows: List[Dict], headers: List[str]) -> Dict[str, Dict]:
        """Get statistics for each column"""
        stats = {}
        
        for header in headers:
            if header.startswith('_'):
                continue
                
            values = [row.get(header, "") for row in data_rows if row.get(header, "")]
            
            stats[header] = {
                'non_empty_count': len(values),
                'unique_count': len(set(values)) if values else 0,
                'sample_values': list(set(values))[:5] if values else []
            }
            
        return stats
    
    def create_enhanced_summary(self, files_data: List[Dict[str, Any]]) -> str:
        """Create enhanced summary with TF-IDF search capabilities"""
        summary_parts = []
        
        summary_parts.append("=== ENHANCED EXCEL ANALYSIS WITH TF-IDF SEARCH ===")
        summary_parts.append(f"Total files processed: {len(files_data)}")
        summary_parts.append("📊 Data is indexed and searchable using TF-IDF similarity")
        
        # Build search index
        self.search_engine.build_index(files_data)
        total_rows = len(self.search_engine.documents)
        summary_parts.append(f"🔍 Total searchable rows: {total_rows}")
        
        for file_data in files_data:
            if 'error' in file_data:
                summary_parts.append(f"\n❌ {file_data.get('file_name', 'Unknown')}: {file_data['error']}")
                continue
                
            file_name = file_data['file_name']
            summary_parts.append(f"\n📁 FILE: {file_name}")
            summary_parts.append(f"   Sheets: {file_data['summary']['total_sheets']} ({', '.join(file_data['summary']['sheet_names'])})")
            
            for sheet_name, sheet_data in file_data['sheets'].items():
                if 'error' in sheet_data:
                    summary_parts.append(f"   ❌ Sheet '{sheet_name}': {sheet_data['error']}")
                    continue
                    
                summary = sheet_data['summary']
                summary_parts.append(f"\n   📊 SHEET: {sheet_name}")
                summary_parts.append(f"      Dimensions: {summary['total_rows']} rows × {summary['total_columns']} columns")
                summary_parts.append(f"      Headers: {', '.join(summary['headers'][:10])}")
                
                # Add column statistics
                if 'column_stats' in summary:
                    summary_parts.append("      Column Statistics:")
                    for col, stats in list(summary['column_stats'].items())[:5]:
                        if not col.startswith('_'):
                            summary_parts.append(f"        {col}: {stats['non_empty_count']} values, {stats['unique_count']} unique")
                
                # Enhanced sample data with row numbers
                if summary['sample_data']:
                    summary_parts.append("      Sample rows (with row numbers):")
                    for row in summary['sample_data'][:3]:
                        row_num = row.get('_row_number', '?')
                        clean_row = {k: str(v)[:40] for k, v in row.items() if not k.startswith('_') and v}
                        summary_parts.append(f"        Row {row_num}: {clean_row}")
        
        # Add search instructions
        summary_parts.append("\n🔍 SEARCH CAPABILITIES:")
        summary_parts.append("You can now search for specific data using:")
        summary_parts.append("- Natural language queries (e.g., 'sales data from Q1')")
        summary_parts.append("- Column-specific searches (e.g., 'revenue numbers')")
        summary_parts.append("- Content-based similarity matching")
        summary_parts.append("- Exact row/column retrieval by name")
        
        return "\n".join(summary_parts)

class GeminiLLM:
    """Enhanced Gemini LLM integration with TF-IDF search"""
    
    def __init__(self, api_key: str, search_engine: TFIDFSearchEngine = None):
        genai.configure(api_key=api_key)
        self.model = "gemini-2.0-flash"
        self.search_engine = search_engine
    
    def analyze_excel_data(self, excel_summary: str, user_query: str = "", search_results: List[Dict] = None) -> str:
        """Enhanced analysis with search results"""
        try:
            # Build enhanced prompt with search capabilities
            search_context = ""
            if search_results:
                search_context = "\n🔍 RELEVANT SEARCH RESULTS:\n"
                for i, result in enumerate(search_results[:5], 1):
                    search_context += f"Result {i} (Score: {result['similarity_score']:.3f}):\n"
                    search_context += f"  File: {result['file_name']}, Sheet: {result['sheet_name']}, Row: {result['row_index'] + 1}\n"
                    search_context += f"  Data: {result['row_data']}\n\n"
            
            prompt = f"""
You are an expert data analyst with TF-IDF search capabilities. You can search through Excel data intelligently.

EXCEL DATA SUMMARY:
{excel_summary}

{search_context}

USER QUERY: {user_query if user_query else "Provide comprehensive analysis with search examples"}

🔍 SEARCH CAPABILITIES:
- You can search for specific rows/data using natural language
- TF-IDF similarity matching finds relevant content
- You can access any column or row data on demand
- Search results include row numbers and exact locations

ANALYSIS REQUIREMENTS:
1. Key insights from the data
2. Data quality observations
3. Patterns or trends you notice
4. Recommendations for further analysis
5. Examples of how to search this data
6. Specific actionable insights

If the user asks about specific data, suggest relevant search terms or demonstrate how to find it.

Be specific, actionable, and show how the search functionality can help explore the data further.
"""

            model = genai.GenerativeModel(self.model)
            response = model.generate_content(prompt)
            
            return response.text
                
        except Exception as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            return f"Error analyzing data with Gemini: {str(e)}"

# Initialize session state
if 'files_data' not in st.session_state:
    st.session_state.files_data = []
if 'excel_summary' not in st.session_state:
    st.session_state.excel_summary = ""
if 'llm_analysis' not in st.session_state:
    st.session_state.llm_analysis = ""
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'excel_processor' not in st.session_state:
    st.session_state.excel_processor = ExcelProcessor()

# Main App
def main():
    st.title("📊 Excel AI Analyzer with TF-IDF Search")
    st.markdown("Upload Excel files, build searchable index, and get AI insights with intelligent data retrieval!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key = st.text_input(
            "🔑 Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )
        
        if api_key:
            st.success("✅ API Key set")
        else:
            st.warning("⚠️ Enter Gemini API key")
        
        st.divider()
        
        # Processing settings
        st.header("⚙️ Processing Settings")
        max_rows = st.slider("Max rows per sheet", 50, 500, 100)
        max_chars = st.slider("Max characters per cell", 200, 1000, 500)
        
        # Update processor settings
        st.session_state.excel_processor.max_rows_per_sheet = max_rows
        st.session_state.excel_processor.max_chars_per_cell = max_chars
        
        st.divider()
        
        # TF-IDF Search Section
        st.header("🔍 TF-IDF Search")
        
        if st.session_state.files_data:
            search_query = st.text_input(
                "Search your data:",
                placeholder="e.g., sales revenue Q1, customer data, highest values"
            )
            
            search_button = st.button("🔍 Search", type="secondary")
            
            if search_button and search_query:
                with st.spinner("Searching..."):
                    results = st.session_state.excel_processor.search_engine.search(search_query, top_k=10)
                    st.session_state.search_results = results
                    
                if results:
                    st.success(f"Found {len(results)} results")
                    
                    # Show top results in sidebar
                    for i, result in enumerate(results[:3], 1):
                        with st.expander(f"Result {i} (Score: {result['similarity_score']:.3f})"):
                            st.write(f"**File:** {result['file_name']}")
                            st.write(f"**Sheet:** {result['sheet_name']}")
                            st.write(f"**Row:** {result['row_index'] + 1}")
                            st.json(result['row_data'], expanded=False)
                else:
                    st.warning("No results found")
        else:
            st.info("Process files first to enable search")
        
        st.divider()
        
        # Clear button
        if st.button("🗑️ Clear All Data", type="secondary"):
            for key in ['files_data', 'excel_summary', 'llm_analysis', 'search_results']:
                st.session_state[key] = [] if 'results' in key or 'data' in key else ""
            st.session_state.excel_processor = ExcelProcessor()
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Excel Files")
        
        uploaded_files = st.file_uploader(
            "Choose Excel files",
            type=['xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload multiple Excel files (.xlsx or .xls format)"
        )
        
        if uploaded_files:
            st.success(f"📄 {len(uploaded_files)} file(s) uploaded")
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size / 1024 / 1024:.2f} MB)")
    
    with col2:
        st.header("❓ Your Question (Optional)")
        
        user_query = st.text_area(
            "Ask about your Excel data",
            placeholder="e.g., What are the sales trends? Find customers with highest revenue. Show me data quality issues.",
            height=150
        )
    
    # Processing section
    st.divider()
    
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        process_button = st.button(
            "🚀 Process & Index",
            type="primary",
            disabled=not uploaded_files,
            help="Process Excel files and build TF-IDF search index"
        )
    
    with col2:
        analyze_button = st.button(
            "🤖 AI Analysis",
            type="primary",
            disabled=not (st.session_state.excel_summary and api_key),
            help="Get AI insights from processed data"
        )
    
    with col3:
        if st.session_state.search_results:
            st.metric("Search Results", len(st.session_state.search_results))
        else:
            st.metric("Search Results", 0)
    
    with col4:
        if st.session_state.excel_processor.search_engine.documents:
            st.metric("Indexed Rows", len(st.session_state.excel_processor.search_engine.documents))
        else:
            st.metric("Indexed Rows", 0)
    
    # Process files
    if process_button and uploaded_files:
        with st.spinner("🔄 Processing files and building search index..."):
            files_data = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                file_content = uploaded_file.read()
                file_data = st.session_state.excel_processor.read_excel_file(file_content, uploaded_file.name)
                files_data.append(file_data)
            
            st.session_state.files_data = files_data
            st.session_state.excel_summary = st.session_state.excel_processor.create_enhanced_summary(files_data)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete! TF-IDF index built.")
            
            st.success(f"Successfully processed {len(files_data)} files with {len(st.session_state.excel_processor.search_engine.documents)} searchable rows!")
    
    # AI Analysis
    if analyze_button and st.session_state.excel_summary and api_key:
        with st.spinner("🤖 Analyzing with AI..."):
            try:
                gemini_llm = GeminiLLM(api_key, st.session_state.excel_processor.search_engine)
                
                # Include search results if available
                search_results = st.session_state.search_results if st.session_state.search_results else None
                
                analysis = gemini_llm.analyze_excel_data(
                    st.session_state.excel_summary, 
                    user_query,
                    search_results
                )
                st.session_state.llm_analysis = analysis
                st.success("✅ AI analysis complete!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Results section
    if st.session_state.excel_summary or st.session_state.llm_analysis or st.session_state.search_results:
        st.divider()
        st.header("📊 Results")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Summary", "🤖 AI Analysis", "🔍 Search Results", "📈 Row Browser"])
        
        with tab1:
            if st.session_state.excel_summary:
                st.subheader("📊 Enhanced Data Summary with TF-IDF Index")
                st.text_area(
                    "Processed data summary",
                    value=st.session_state.excel_summary,
                    height=400,
                    disabled=True
                )
                
                st.download_button(
                    label="📥 Download Summary",
                    data=st.session_state.excel_summary,
                    file_name="excel_summary_with_tfidf.txt",
                    mime="text/plain"
                )
            else:
                st.info("👆 Process Excel files first")
        
        with tab2:
            if st.session_state.llm_analysis:
                st.subheader("🤖 AI Analysis with Search Context")
                st.markdown(st.session_state.llm_analysis)
                
                st.download_button(
                    label="📥 Download Analysis",
                    data=st.session_state.llm_analysis,
                    file_name="ai_analysis.txt",
                    mime="text/plain"
                )
            else:
                st.info("👆 Run AI analysis to see insights")
        
        with tab3:
            if st.session_state.search_results:
                st.subheader("🔍 TF-IDF Search Results")
                
                for i, result in enumerate(st.session_state.search_results, 1):
                    with st.expander(f"📄 Result {i} - Similarity: {result['similarity_score']:.3f}"):
                        col1, col2, col3 = st.columns([1, 1, 1])
                        col1.metric("File", result['file_name'])
                        col2.metric("Sheet", result['sheet_name'])
                        col3.metric("Row Number", result['row_index'] + 1)
                        
                        st.subheader("Row Data:")
                        # Create a nice table view
                        row_df = pd.DataFrame([result['row_data']])
                        st.dataframe(row_df, use_container_width=True)
                        
                        st.subheader("Matched Text:")
                        st.text(result['matched_text'])
            else:
                st.info("👆 Use the search feature in the sidebar")
        
        with tab4:
            if st.session_state.files_data:
                st.subheader("📈 Row Browser - View All Data")
                
                # File and sheet selector
                file_options = [f['file_name'] for f in st.session_state.files_data if 'error' not in f]
                if file_options:
                    selected_file = st.selectbox("Select File:", file_options)
                    
                    # Get selected file data
                    file_data = next(f for f in st.session_state.files_data if f.get('file_name') == selected_file)
                    sheet_options = list(file_data['sheets'].keys())
                    
                    if sheet_options:
                        selected_sheet = st.selectbox("Select Sheet:", sheet_options)
                        
                        # Display sheet data
                        sheet_data = file_data['sheets'][selected_sheet]
                        if 'error' not in sheet_data and sheet_data['data']:
                            st.subheader(f"📊 {selected_sheet} - All Rows")
                            
                            # Convert to DataFrame for better display
                            df = pd.DataFrame(sheet_data['data'])
                            
                            # Remove internal columns
                            display_df = df.drop(columns=[col for col in df.columns if col.startswith('_')])
                            
                            # Add row numbers
                            display_df.insert(0, 'Row #', df['_row_number'] if '_row_number' in df.columns else range(1, len(df) + 1))
                            
                            st.dataframe(display_df, use_container_width=True, height=400)
                            
                            # Download option
                            csv = display_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download as CSV",
                                data=csv,
                                file_name=f"{selected_file}_{selected_sheet}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No data to display for this sheet")
            else:
                st.info("👆 Process Excel files first")

if __name__ == "__main__":
    main()
