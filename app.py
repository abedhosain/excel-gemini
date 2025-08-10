import streamlit as st
import json
import os
from typing import List, Dict, Any, Optional
import openpyxl
from openpyxl import load_workbook
import google.generativeai as genai
import tempfile
import logging
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="📊 Excel AI Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ExcelProcessor:
    """Efficient Excel file processor optimized for LLM token usage"""
    
    def __init__(self):
        self.max_rows_per_sheet = 100  # Limit rows to control token usage
        self.max_chars_per_cell = 500  # Limit cell content length
        
    def read_excel_file(self, file_content: bytes, file_name: str) -> Dict[str, Any]:
        """Read Excel file from bytes content"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            # Load workbook
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
            
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
            return file_data
            
        except Exception as e:
            logger.error(f"Error reading Excel file {file_name}: {str(e)}")
            return {'error': f"Failed to read {file_name}: {str(e)}"}
    
    def _process_sheet(self, sheet, sheet_name: str) -> Dict[str, Any]:
        """Process individual sheet with optimization for LLM"""
        try:
            if sheet.max_row is None or sheet.max_row == 0:
                return {
                    'sheet_name': sheet_name,
                    'summary': {'total_rows': 0, 'total_columns': 0, 'headers': [], 'sample_data': []},
                    'data': []
                }
            
            # Get sheet dimensions
            max_row = min(sheet.max_row, self.max_rows_per_sheet + 1)  # +1 for header
            max_col = sheet.max_column or 0
            
            # Extract headers (first row)
            headers = []
            for col in range(1, max_col + 1):
                cell_value = sheet.cell(row=1, column=col).value
                headers.append(str(cell_value) if cell_value is not None else f"Column_{col}")
            
            # Extract data rows
            data_rows = []
            for row in range(2, max_row + 1):
                row_data = {}
                has_data = False
                
                for col, header in enumerate(headers, 1):
                    cell_value = sheet.cell(row=row, column=col).value
                    
                    # Clean and limit cell content
                    if cell_value is not None:
                        cell_str = str(cell_value)
                        if len(cell_str) > self.max_chars_per_cell:
                            cell_str = cell_str[:self.max_chars_per_cell] + "..."
                        row_data[header] = cell_str
                        has_data = True
                    else:
                        row_data[header] = ""
                
                if has_data:
                    data_rows.append(row_data)
            
            # Create summary statistics
            summary = {
                'total_rows': len(data_rows),
                'total_columns': len(headers),
                'headers': headers,
                'data_types': self._analyze_data_types(data_rows, headers),
                'sample_data': data_rows[:3] if data_rows else []  # First 3 rows as sample
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
        """Analyze data types for each column"""
        data_types = {}
        
        for header in headers:
            sample_values = [row.get(header, "") for row in data_rows[:10] if row.get(header, "")]
            
            if not sample_values:
                data_types[header] = "empty"
                continue
                
            # Simple type detection
            numeric_count = sum(1 for val in sample_values if str(val).replace('.', '').replace('-', '').isdigit())
            
            if numeric_count > len(sample_values) * 0.7:
                data_types[header] = "numeric"
            else:
                data_types[header] = "text"
                
        return data_types
    
    def create_llm_optimized_summary(self, files_data: List[Dict[str, Any]]) -> str:
        """Create a token-efficient summary for LLM"""
        summary_parts = []
        
        summary_parts.append("=== EXCEL FILES ANALYSIS SUMMARY ===")
        summary_parts.append(f"Total files processed: {len(files_data)}")
        
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
                summary_parts.append(f"      Columns: {', '.join(summary['headers'][:10])}{'...' if len(summary['headers']) > 10 else ''}")
                
                # Add sample data
                if summary['sample_data']:
                    summary_parts.append("      Sample data:")
                    for i, row in enumerate(summary['sample_data'][:2], 1):
                        row_preview = {k: str(v)[:50] + ("..." if len(str(v)) > 50 else "") for k, v in row.items()}
                        summary_parts.append(f"        Row {i}: {json.dumps(row_preview, ensure_ascii=False)}")
        
        return "\n".join(summary_parts)

class GeminiLLM:
    """Gemini LLM integration"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = "gemini-1.5-flash"
    
    def analyze_excel_data(self, excel_summary: str, user_query: str = "") -> str:
        """Send Excel data to Gemini for analysis"""
        try:
            prompt = f"""
You are an expert data analyst. I have processed multiple Excel files and need your analysis.

EXCEL DATA SUMMARY:
{excel_summary}

USER QUERY: {user_query if user_query else "Please provide a comprehensive analysis of this Excel data, including key insights, patterns, and recommendations."}

Please provide:
1. Key insights from the data
2. Data quality observations
3. Patterns or trends you notice
4. Recommendations for further analysis
5. Any potential issues or anomalies

Be specific and actionable in your response.
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

# Initialize processors
excel_processor = ExcelProcessor()

# Main App
def main():
    st.title("📊 Excel to LLM Processor")
    st.markdown("Upload multiple Excel files to analyze with AI. Optimized for efficient token usage.")
    
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
            st.success("✅ API Key set successfully")
        else:
            st.warning("⚠️ Please enter your Gemini API key")
        
        st.divider()
        
        # Processing settings
        st.header("⚙️ Processing Settings")
        max_rows = st.slider("Max rows per sheet", 50, 200, 100)
        max_chars = st.slider("Max characters per cell", 200, 1000, 500)
        
        # Update processor settings
        excel_processor.max_rows_per_sheet = max_rows
        excel_processor.max_chars_per_cell = max_chars
        
        st.divider()
        
        # Clear button
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.files_data = []
            st.session_state.excel_summary = ""
            st.session_state.llm_analysis = ""
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Excel Files")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose Excel files",
            type=['xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload multiple Excel files (.xlsx or .xls format)"
        )
        
        if uploaded_files:
            st.success(f"📄 {len(uploaded_files)} file(s) uploaded")
            
            # Display file information
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size / 1024 / 1024:.2f} MB)")
    
    with col2:
        st.header("❓ Your Question (Optional)")
        
        user_query = st.text_area(
            "Ask a specific question about your Excel data",
            placeholder="e.g., What are the sales trends? Are there any data quality issues? What insights can you provide?",
            height=150
        )
    
    # Processing section
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        process_button = st.button(
            "🚀 Process Files",
            type="primary",
            disabled=not (uploaded_files and api_key),
            help="Process Excel files and extract data"
        )
    
    with col2:
        analyze_button = st.button(
            "🤖 Analyze with AI",
            type="primary",
            disabled=not (st.session_state.excel_summary and api_key),
            help="Send processed data to Gemini for analysis"
        )
    
    with col3:
        st.metric("Files Processed", len(st.session_state.files_data))
    
    # Process files
    if process_button and uploaded_files:
        with st.spinner("🔄 Processing Excel files..."):
            files_data = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Read file content
                file_content = uploaded_file.read()
                
                # Process Excel file
                file_data = excel_processor.read_excel_file(file_content, uploaded_file.name)
                files_data.append(file_data)
            
            # Store in session state
            st.session_state.files_data = files_data
            
            # Create summary
            st.session_state.excel_summary = excel_processor.create_llm_optimized_summary(files_data)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            st.success(f"Successfully processed {len(files_data)} files!")
    
    # Analyze with AI
    if analyze_button and st.session_state.excel_summary and api_key:
        with st.spinner("🤖 Analyzing data with Gemini AI..."):
            try:
                gemini_llm = GeminiLLM(api_key)
                analysis = gemini_llm.analyze_excel_data(st.session_state.excel_summary, user_query)
                st.session_state.llm_analysis = analysis
                st.success("✅ AI analysis complete!")
            except Exception as e:
                st.error(f"❌ Error during AI analysis: {str(e)}")
    
    # Results section
    if st.session_state.excel_summary or st.session_state.llm_analysis:
        st.divider()
        st.header("📊 Results")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📋 Data Summary", "🤖 AI Analysis", "📈 Detailed Data"])
        
        with tab1:
            if st.session_state.excel_summary:
                st.subheader("📊 Excel Data Summary")
                st.text_area(
                    "Processed data summary",
                    value=st.session_state.excel_summary,
                    height=400,
                    disabled=True
                )
                
                # Download button for summary
                st.download_button(
                    label="📥 Download Summary",
                    data=st.session_state.excel_summary,
                    file_name="excel_summary.txt",
                    mime="text/plain"
                )
            else:
                st.info("👆 Process Excel files first to see the data summary")
        
        with tab2:
            if st.session_state.llm_analysis:
                st.subheader("🤖 AI Analysis Results")
                st.markdown(st.session_state.llm_analysis)
                
                # Download button for analysis
                st.download_button(
                    label="📥 Download Analysis",
                    data=st.session_state.llm_analysis,
                    file_name="ai_analysis.txt",
                    mime="text/plain"
                )
            else:
                st.info("👆 Run AI analysis to see insights and recommendations")
        
        with tab3:
            if st.session_state.files_data:
                st.subheader("📈 Detailed File Information")
                
                for file_data in st.session_state.files_data:
                    if 'error' in file_data:
                        st.error(f"❌ {file_data.get('file_name', 'Unknown')}: {file_data['error']}")
                        continue
                    
                    with st.expander(f"📁 {file_data['file_name']} ({file_data['summary']['total_sheets']} sheets)"):
                        for sheet_name, sheet_data in file_data['sheets'].items():
                            if 'error' in sheet_data:
                                st.error(f"❌ Sheet '{sheet_name}': {sheet_data['error']}")
                                continue
                            
                            st.subheader(f"📊 Sheet: {sheet_name}")
                            summary = sheet_data['summary']
                            
                            # Sheet metrics
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Rows", summary['total_rows'])
                            col2.metric("Columns", summary['total_columns'])
                            col3.metric("Data Types", len(set(summary.get('data_types', {}).values())))
                            
                            # Headers
                            st.write("**Headers:**", ", ".join(summary['headers'][:10]))
                            if len(summary['headers']) > 10:
                                st.write(f"... and {len(summary['headers']) - 10} more")
                            
                            # Sample data
                            if summary['sample_data']:
                                st.write("**Sample Data:**")
                                for i, row in enumerate(summary['sample_data'][:3], 1):
                                    st.json(row, expanded=False)
            else:
                st.info("👆 Process Excel files first to see detailed data")

if __name__ == "__main__":
    main()
