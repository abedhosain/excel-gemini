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
import math
import re
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="📊 On-Demand Excel AI Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SimpleTFIDFSearchEngine:
    """Lightweight TF-IDF search engine for on-demand data retrieval"""
    
    def __init__(self):
        self.documents = []
        self.row_metadata = []
        self.idf_scores = {}
        self.tfidf_vectors = []
        
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        return [token for token in tokens if token not in stop_words and len(token) > 1]
    
    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency"""
        tf = {}
        total_tokens = len(tokens)
        token_counts = Counter(tokens)
        
        for token, count in token_counts.items():
            tf[token] = count / total_tokens
        
        return tf
    
    def _compute_idf(self) -> None:
        """Compute inverse document frequency for all terms"""
        doc_count = len(self.documents)
        term_doc_count = defaultdict(int)
        
        for doc in self.documents:
            tokens = self._tokenize(doc)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                term_doc_count[token] += 1
        
        for term, count in term_doc_count.items():
            self.idf_scores[term] = math.log(doc_count / count)
    
    def _compute_tfidf_vector(self, tokens: List[str]) -> Dict[str, float]:
        """Compute TF-IDF vector for a document"""
        tf_scores = self._compute_tf(tokens)
        tfidf_vector = {}
        
        for token in tf_scores:
            if token in self.idf_scores:
                tfidf_vector[token] = tf_scores[token] * self.idf_scores[token]
        
        return tfidf_vector
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Compute cosine similarity between two TF-IDF vectors"""
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        if not common_terms:
            return 0.0
        
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        mag1 = math.sqrt(sum(val**2 for val in vec1.values()))
        mag2 = math.sqrt(sum(val**2 for val in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
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
                
                for row_idx, row in enumerate(sheet_data['data']):
                    text_content = []
                    for header, value in row.items():
                        if value and str(value).strip() and not header.startswith('_'):
                            text_content.append(f"{header}: {str(value)}")
                    
                    if text_content:
                        document = " | ".join(text_content)
                        self.documents.append(document)
                        
                        self.row_metadata.append({
                            'file_name': file_name,
                            'sheet_name': sheet_name,
                            'row_index': row_idx,
                            'row_data': row,
                            'headers': list(row.keys())
                        })
        
        if self.documents:
            self._compute_idf()
            
            self.tfidf_vectors = []
            for doc in self.documents:
                tokens = self._tokenize(doc)
                tfidf_vector = self._compute_tfidf_vector(tokens)
                self.tfidf_vectors.append(tfidf_vector)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant rows using TF-IDF similarity"""
        if not self.documents:
            return []
        
        query_tokens = self._tokenize(query)
        query_vector = self._compute_tfidf_vector(query_tokens)
        
        if not query_vector:
            return []
        
        similarities = []
        for i, doc_vector in enumerate(self.tfidf_vectors):
            similarity = self._cosine_similarity(query_vector, doc_vector)
            similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, similarity_score in similarities[:top_k]:
            if similarity_score > 0:
                result = self.row_metadata[idx].copy()
                result['similarity_score'] = similarity_score
                result['matched_text'] = self.documents[idx]
                results.append(result)
        
        return results

class OnDemandDataEngine:
    """Data engine that executes Gemini's search instructions"""
    
    def __init__(self, excel_processor: 'ExcelProcessor'):
        self.excel_processor = excel_processor
        self.search_engine = excel_processor.search_engine
        
    def execute_search_instruction(self, instruction: str) -> Dict[str, Any]:
        """Execute search instruction from Gemini"""
        try:
            instruction_lower = instruction.lower().strip()
            
            # Parse different types of search instructions
            if instruction.startswith('SEARCH:'):
                query = instruction[7:].strip()
                return self._execute_tfidf_search(query)
            
            elif instruction.startswith('GET_ROW:'):
                params = instruction[8:].strip().split()
                if len(params) >= 2:
                    sheet_name = params[0]
                    try:
                        row_number = int(params[1])
                        return self._get_specific_row(sheet_name, row_number)
                    except ValueError:
                        return {'success': False, 'error': 'Invalid row number'}
                        
            elif instruction.startswith('GET_COLUMN:'):
                params = instruction[11:].strip().split()
                if len(params) >= 2:
                    sheet_name = params[0]
                    column_name = ' '.join(params[1:])
                    return self._get_column_data(sheet_name, column_name)
            
            elif instruction.startswith('CALCULATE:'):
                query = instruction[10:].strip()
                return self._calculate_from_search(query)
            
            elif instruction.startswith('FIND:'):
                query = instruction[5:].strip()
                return self._execute_tfidf_search(query)
            
            else:
                # Default to TF-IDF search
                return self._execute_tfidf_search(instruction)
                
        except Exception as e:
            return {'success': False, 'error': f"Error executing instruction: {str(e)}"}
    
    def _execute_tfidf_search(self, query: str) -> Dict[str, Any]:
        """Execute TF-IDF search based on query"""
        search_results = self.search_engine.search(query, top_k=5)
        
        if not search_results:
            return {'success': False, 'error': f"No results found for: {query}"}
        
        # Return only essential data
        formatted_results = []
        for result in search_results:
            relevant_data = {k: v for k, v in result['row_data'].items() 
                           if not k.startswith('_') and v and str(v).strip()}
            
            formatted_results.append({
                'location': f"{result['file_name']}/{result['sheet_name']}/Row {result['row_index']+1}",
                'score': f"{result['similarity_score']:.3f}",
                'data': relevant_data
            })
        
        return {
            'success': True,
            'type': 'search_results',
            'results': formatted_results,
            'query': query,
            'count': len(search_results)
        }
    
    def _get_specific_row(self, sheet_name: str, row_number: int) -> Dict[str, Any]:
        """Get specific row using openpyxl metadata"""
        for metadata in self.search_engine.row_metadata:
            if (metadata['sheet_name'].lower() == sheet_name.lower() and 
                metadata['row_data'].get('_row_number') == row_number):
                
                clean_data = {k: v for k, v in metadata['row_data'].items() 
                             if not k.startswith('_') and v and str(v).strip()}
                
                return {
                    'success': True,
                    'type': 'specific_row',
                    'data': clean_data,
                    'location': f"Row {row_number} from '{sheet_name}'"
                }
        
        return {'success': False, 'error': f"Row {row_number} not found in {sheet_name}"}
    
    def _get_column_data(self, sheet_name: str, column_name: str) -> Dict[str, Any]:
        """Get column data using openpyxl metadata"""
        values = []
        
        for metadata in self.search_engine.row_metadata:
            if metadata['sheet_name'].lower() == sheet_name.lower():
                row_data = metadata['row_data']
                # Find column with partial matching
                matching_column = None
                for col in row_data.keys():
                    if column_name.lower() in col.lower():
                        matching_column = col
                        break
                
                if matching_column and row_data[matching_column] and str(row_data[matching_column]).strip():
                    values.append({
                        'row': row_data.get('_row_number', '?'),
                        'value': row_data[matching_column]
                    })
        
        if not values:
            return {'success': False, 'error': f"Column '{column_name}' not found in {sheet_name}"}
        
        # Limit results
        display_values = values[:8]
        
        return {
            'success': True,
            'type': 'column_data',
            'data': display_values,
            'total_count': len(values),
            'column_name': column_name,
            'sheet_name': sheet_name
        }
    
    def _calculate_from_search(self, query: str) -> Dict[str, Any]:
        """Calculate statistics from search results"""
        search_results = self.search_engine.search(query, top_k=20)
        
        if not search_results:
            return {'success': False, 'error': f"No data found for calculation: {query}"}
        
        numeric_values = []
        sources = []
        
        for result in search_results:
            for key, value in result['row_data'].items():
                if not key.startswith('_') and value:
                    try:
                        clean_value = str(value).replace(',', '').replace('$', '').replace('₹', '').strip()
                        num_val = float(clean_value)
                        numeric_values.append(num_val)
                        sources.append({
                            'value': num_val,
                            'location': f"{result['file_name']}/{result['sheet_name']}/Row {result['row_index']+1}/{key}"
                        })
                    except:
                        continue
        
        if not numeric_values:
            return {'success': False, 'error': "No numeric values found for calculation"}
        
        stats = {
            'count': len(numeric_values),
            'sum': sum(numeric_values),
            'average': sum(numeric_values) / len(numeric_values),
            'min': min(numeric_values),
            'max': max(numeric_values)
        }
        
        return {
            'success': True,
            'type': 'calculation',
            'statistics': stats,
            'sources_count': len(sources),
            'query': query
        }

class ExcelProcessor:
    """Excel processor that only provides structure info"""
    
    def __init__(self):
        self.max_rows_per_sheet = 100
        self.max_chars_per_cell = 500
        self.search_engine = SimpleTFIDFSearchEngine()
        
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
        """Process sheet and store data for search"""
        try:
            if sheet.max_row is None or sheet.max_row == 0:
                return {
                    'sheet_name': sheet_name,
                    'summary': {'total_rows': 0, 'total_columns': 0, 'headers': []},
                    'data': []
                }
            
            max_row = min(sheet.max_row, self.max_rows_per_sheet + 1)
            max_col = sheet.max_column or 0
            
            # Extract headers
            headers = []
            for col in range(1, max_col + 1):
                cell_value = sheet.cell(row=1, column=col).value
                if cell_value is not None:
                    header = str(cell_value).strip()
                    header = re.sub(r'[^\w\s]', '', header)
                    header = re.sub(r'\s+', ' ', header)
                    headers.append(header if header else f"Column_{col}")
                else:
                    headers.append(f"Column_{col}")
            
            # Extract data rows
            data_rows = []
            for row in range(2, max_row + 1):
                row_data = {}
                has_data = False
                
                for col, header in enumerate(headers, 1):
                    cell_value = sheet.cell(row=row, column=col).value
                    
                    if cell_value is not None:
                        cell_str = str(cell_value).strip()
                        if len(cell_str) > self.max_chars_per_cell:
                            cell_str = cell_str[:self.max_chars_per_cell] + "..."
                        row_data[header] = cell_str
                        has_data = True
                    else:
                        row_data[header] = ""
                
                if has_data:
                    row_data['_row_number'] = row - 1
                    data_rows.append(row_data)
            
            # Return only structure info
            summary = {
                'total_rows': len(data_rows),
                'total_columns': len(headers),
                'headers': headers
            }
            
            return {
                'sheet_name': sheet_name,
                'summary': summary,
                'data': data_rows  # Store for search but don't expose
            }
            
        except Exception as e:
            logger.error(f"Error processing sheet {sheet_name}: {str(e)}")
            return {'error': f"Failed to process sheet {sheet_name}: {str(e)}"}
    
    def create_structure_only_summary(self, files_data: List[Dict[str, Any]]) -> str:
        """Create structure-only summary for Gemini"""
        summary_parts = []
        
        summary_parts.append("=== EXCEL FILES STRUCTURE (NO DATA PROVIDED) ===")
        summary_parts.append(f"Total files: {len(files_data)}")
        
        # Build search index silently
        self.search_engine.build_index(files_data)
        total_rows = len(self.search_engine.documents)
        summary_parts.append(f"Total indexed rows available for search: {total_rows}")
        
        for file_data in files_data:
            if 'error' in file_data:
                summary_parts.append(f"\n❌ {file_data.get('file_name', 'Unknown')}: {file_data['error']}")
                continue
                
            file_name = file_data['file_name']
            summary_parts.append(f"\n📁 FILE: {file_name}")
            summary_parts.append(f"   Available sheets: {', '.join(file_data['summary']['sheet_names'])}")
            
            for sheet_name, sheet_data in file_data['sheets'].items():
                if 'error' in sheet_data:
                    summary_parts.append(f"   ❌ Sheet '{sheet_name}': {sheet_data['error']}")
                    continue
                    
                summary = sheet_data['summary']
                summary_parts.append(f"\n   📊 SHEET: {sheet_name}")
                summary_parts.append(f"      Structure: {summary['total_rows']} rows × {summary['total_columns']} columns")
                summary_parts.append(f"      Available columns: {', '.join(summary['headers'])}")
        
        summary_parts.append("\n🔍 TO GET ACTUAL DATA, USE THESE INSTRUCTIONS:")
        summary_parts.append("- SEARCH: [query] - Search using TF-IDF")
        summary_parts.append("- GET_ROW: [sheet_name] [row_number] - Get specific row")
        summary_parts.append("- GET_COLUMN: [sheet_name] [column_name] - Get column data")
        summary_parts.append("- CALCULATE: [query] - Calculate from search results")
        summary_parts.append("- FIND: [query] - Find specific data")
        summary_parts.append("\nNO ACTUAL DATA IS PROVIDED - YOU MUST SEARCH FOR EVERYTHING!")
        
        return "\n".join(summary_parts)

class GeminiSearchDirector:
    """Gemini that directs TF-IDF and openpyxl searches"""
    
    def __init__(self, api_key: str, data_engine: OnDemandDataEngine):
        genai.configure(api_key=api_key)
        self.model = "gemini-2.0-flash-exp"
        self.data_engine = data_engine
        
    def analyze_with_search_instructions(self, structure_summary: str, user_query: str = "") -> str:
        """Gemini analyzes by instructing search operations"""
        try:
            # Phase 1: Gemini plans search strategy
            planning_prompt = f"""
You are a data analyst with access to Excel files. You have ONLY the structure information below - NO actual data.

{structure_summary}

USER QUESTION: {user_query if user_query else "Analyze this Excel data"}

You must instruct the TF-IDF search engine and openpyxl system to retrieve data. Available instructions:

- SEARCH: [query] - Use TF-IDF to find relevant data
- GET_ROW: [sheet_name] [row_number] - Get specific row from sheet
- GET_COLUMN: [sheet_name] [column_name] - Get all values from column  
- CALCULATE: [query] - Find and calculate numeric data
- FIND: [query] - Find specific information

Plan your search strategy and provide 4-5 specific search instructions to answer the user's question.

Format your response as:
SEARCH STRATEGY: [explain your approach]

SEARCH INSTRUCTIONS:
1. [instruction 1]
2. [instruction 2] 
3. [instruction 3]
4. [instruction 4]
5. [instruction 5]

Remember: You have NO data yet - you must search for everything!
"""

            model = genai.GenerativeModel(self.model)
            planning_response = model.generate_content(planning_prompt)
            
            # Extract search instructions
            search_instructions = self._extract_search_instructions(planning_response.text)
            
            # Phase 2: Execute search instructions
            retrieved_data = ""
            if search_instructions:
                retrieved_data = "\n📊 SEARCH EXECUTION RESULTS:\n"
                
                for i, instruction in enumerate(search_instructions[:5], 1):
                    retrieved_data += f"\n🔍 Instruction {i}: {instruction}\n"
                    
                    result = self.data_engine.execute_search_instruction(instruction)
                    formatted_result = self._format_search_result(result)
                    retrieved_data += f"Result: {formatted_result}\n"
            
            # Phase 3: Final analysis with retrieved data
            if retrieved_data:
                analysis_prompt = f"""
You planned this search strategy:
{planning_response.text}

Here are the results from your search instructions:
{retrieved_data}

Now provide comprehensive analysis using ONLY the data you retrieved above. 

Provide:
1. Direct answers to the user's question
2. Key insights from the retrieved data
3. Specific findings with exact numbers and locations
4. Data-driven recommendations
5. Any patterns or trends you discovered

Reference only the specific data you retrieved through your search instructions.
"""
                
                final_response = model.generate_content(analysis_prompt)
                
                return f"""
🎯 SEARCH STRATEGY & EXECUTION:
{planning_response.text}

{retrieved_data}

📈 ANALYSIS BASED ON RETRIEVED DATA:
{final_response.text}
"""
            else:
                return f"{planning_response.text}\n\n❌ No data was successfully retrieved."
                
        except Exception as e:
            logger.error(f"Error in search-directed analysis: {str(e)}")
            return f"Error in analysis: {str(e)}"
    
    def _extract_search_instructions(self, text: str) -> List[str]:
        """Extract search instructions from Gemini's response"""
        instructions = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for instruction patterns
            if any(cmd in line.upper() for cmd in ['SEARCH:', 'GET_ROW:', 'GET_COLUMN:', 'CALCULATE:', 'FIND:']):
                clean_instruction = line
                # Remove numbering if present
                clean_instruction = re.sub(r'^\d+\.\s*', '', clean_instruction)
                instructions.append(clean_instruction.strip())
        
        return instructions[:5]  # Limit to 5 instructions
    
    def _format_search_result(self, result: Dict[str, Any]) -> str:
        """Format search results for Gemini"""
        if not result.get('success', False):
            return f"❌ {result.get('error', 'No data found')}"
        
        result_type = result.get('type', 'unknown')
        
        if result_type == 'search_results':
            results = result['results'][:3]  # Show top 3
            formatted = f"✅ Found {result['count']} results:\n"
            for res in results:
                formatted += f"   📍 {res['location']} (Score: {res['score']})\n"
                # Show first 3 data items
                data_items = list(res['data'].items())[:3]
                data_str = ' | '.join([f"{k}: {v}" for k, v in data_items])
                formatted += f"   📊 Data: {data_str}\n"
            return formatted
        
        elif result_type == 'specific_row':
            data = result['data']
            if data:
                data_items = list(data.items())[:5]  # Show first 5 fields
                data_str = ' | '.join([f"{k}: {v}" for k, v in data_items])
                return f"✅ {result['location']}\n   📊 Data: {data_str}"
            else:
                return f"✅ {result['location']} - No data in this row"
        
        elif result_type == 'column_data':
            data = result['data'][:3]  # Show first 3 values
            formatted = f"✅ Found {result['total_count']} values in column '{result['column_name']}'\n"
            for item in data:
                formatted += f"   Row {item['row']}: {item['value']}\n"
            return formatted
        
        elif result_type == 'calculation':
            stats = result['statistics']
            return f"✅ Calculation from {result['sources_count']} values:\n   Sum: {stats['sum']:,.2f} | Avg: {stats['average']:,.2f} | Count: {stats['count']}"
        
        else:
            return f"✅ Retrieved: {str(result)[:150]}..."

# Initialize session state
if 'files_data' not in st.session_state:
    st.session_state.files_data = []
if 'structure_summary' not in st.session_state:
    st.session_state.structure_summary = ""
if 'llm_analysis' not in st.session_state:
    st.session_state.llm_analysis = ""
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'excel_processor' not in st.session_state:
    st.session_state.excel_processor = ExcelProcessor()
if 'data_engine' not in st.session_state:
    st.session_state.data_engine = None

# Main App
def main():
    st.title("📊 On-Demand Excel AI Analyzer")
    st.markdown("**Gemini instructs TF-IDF & openpyxl to search and retrieve data on-demand!**")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key
        api_key = st.text_input(
            "🔑 Gemini API Key",
            type="password",
            help="Gemini will direct data searches"
        )
        
        if api_key:
            st.success("✅ Ready for directed search")
        else:
            st.warning("⚠️ Enter Gemini API key")
        
        st.divider()
        
        # Search Engine Status
        st.header("🔍 Search Engine Status")
        if st.session_state.data_engine:
            indexed_rows = len(st.session_state.excel_processor.search_engine.documents)
            st.metric("Indexed Rows", indexed_rows)
            st.success("TF-IDF engine ready")
        else:
            st.info("Process files to enable search")
        
        st.divider()
        
        # Manual search test
        if st.session_state.data_engine:
            st.header("🧪 Test Search Instructions")
            test_instruction = st.text_input("Test instruction:", placeholder="SEARCH: revenue data")
            if st.button("Execute"):
                if test_instruction:
                    result = st.session_state.data_engine.execute_search_instruction(test_instruction)
                    if result.get('success'):
                        st.success("✅ Search executed")
                        st.json(result, expanded=False)
                    else:
                        st.error(f"❌ {result.get('error')}")
        
        if st.button("🗑️ Clear All", type="secondary"):
            for key in ['files_data', 'structure_summary', 'llm_analysis', 'search_results']:
                st.session_state[key] = [] if 'results' in key or 'data' in key else ""
            st.session_state.excel_processor = ExcelProcessor()
            st.session_state.data_engine = None
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Excel Files")
        uploaded_files = st.file_uploader(
            "Upload Excel files",
            type=['xlsx', 'xls'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"📄 {len(uploaded_files)} file(s) ready")
    
    with col2:
        st.header("❓ Your Analysis Question")
        user_query = st.text_area(
            "What do you want to analyze?",
            placeholder="e.g., What are the key financial insights? Find revenue trends. Calculate total expenses.",
            height=120
        )
    
    # Action buttons
    st.divider()
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        process_button = st.button(
            "🚀 Index Files",
            type="primary",
            disabled=not uploaded_files,
            help="Process files and build search index"
        )
    
    with col2:
        analyze_button = st.button(
            "🎯 Gemini Search & Analyze",
            type="primary",
            disabled=not (st.session_state.structure_summary and api_key),
            help="Gemini will direct TF-IDF searches"
        )
    
    with col3:
        if st.session_state.data_engine:
            indexed_count = len(st.session_state.excel_processor.search_engine.documents)
            st.metric("Ready for Search", indexed_count)
        else:
            st.metric("Ready for Search", 0)
    
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
            st.session_state.structure_summary = st.session_state.excel_processor.create_structure_only_summary(files_data)
            
            # Initialize search engine
            st.session_state.data_engine = OnDemandDataEngine(st.session_state.excel_processor)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Search index ready for Gemini instructions!")
            
            indexed_rows = len(st.session_state.excel_processor.search_engine.documents)
            st.success(f"🔍 Indexed {indexed_rows} rows ready for on-demand search!")
    
    # Gemini directs searches
    if analyze_button and st.session_state.structure_summary and api_key:
        with st.spinner("🤖 Gemini is planning and executing search strategy..."):
            try:
                if not st.session_state.data_engine:
                    st.session_state.data_engine = OnDemandDataEngine(st.session_state.excel_processor)
                
                gemini_director = GeminiSearchDirector(api_key, st.session_state.data_engine)
                
                analysis = gemini_director.analyze_with_search_instructions(
                    st.session_state.structure_summary, 
                    user_query
                )
                st.session_state.llm_analysis = analysis
                st.success("✅ Gemini completed search-directed analysis!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Results
    if st.session_state.structure_summary or st.session_state.llm_analysis:
        st.divider()
        st.header("📊 Results")
        
        tab1, tab2, tab3 = st.tabs(["📋 File Structure", "🎯 Gemini Analysis", "🔍 Search Testing"])
        
        with tab1:
            if st.session_state.structure_summary:
                st.subheader("📊 Excel File Structure (No Data)")
                st.text_area(
                    "Structure available to Gemini",
                    value=st.session_state.structure_summary,
                    height=400,
                    disabled=True
                )
                
                st.info("👆 This is ALL Gemini sees initially - no actual data!")
                
                st.download_button(
                    "📥 Download Structure",
                    data=st.session_state.structure_summary,
                    file_name="excel_structure_only.txt",
                    mime="text/plain"
                )
            else:
                st.info("👆 Process Excel files first")
        
        with tab2:
            if st.session_state.llm_analysis:
                st.subheader("🎯 Gemini's Search-Directed Analysis")
                st.markdown(st.session_state.llm_analysis)
                
                st.info("👆 Gemini planned searches, executed them via TF-IDF/openpyxl, then analyzed the results!")
                
                st.download_button(
                    "📥 Download Analysis",
                    data=st.session_state.llm_analysis,
                    file_name="gemini_search_analysis.txt",
                    mime="text/plain"
                )
            else:
                st.info("👆 Run Gemini analysis to see search-directed insights")
        
        with tab3:
            if st.session_state.data_engine:
                st.subheader("🔍 Test Search Instructions")
                st.markdown("**Test the same search instructions Gemini uses:**")
                
                # Instruction examples
                example_instructions = [
                    "SEARCH: revenue data",
                    "GET_ROW: Balance_Sheet 1",
                    "GET_COLUMN: Income_Statement Revenue",
                    "CALCULATE: total expenses",
                    "FIND: accommodation costs"
                ]
                
                selected_instruction = st.selectbox(
                    "Choose instruction to test:",
                    [""] + example_instructions + ["Custom..."]
                )
                
                if selected_instruction == "Custom...":
                    custom_instruction = st.text_input("Enter custom instruction:")
                    test_instruction = custom_instruction
                else:
                    test_instruction = selected_instruction
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if st.button("🔍 Execute Instruction") and test_instruction:
                        result = st.session_state.data_engine.execute_search_instruction(test_instruction)
                        
                        if result.get('success'):
                            st.success("✅ Instruction executed")
                        else:
                            st.error(f"❌ {result.get('error')}")
                
                with col2:
                    if test_instruction:
                        st.write(f"**Testing:** `{test_instruction}`")
                
                # Show result if there's one
                if st.button("🔍 Execute Instruction") and test_instruction:
                    with st.spinner("Executing search instruction..."):
                        result = st.session_state.data_engine.execute_search_instruction(test_instruction)
                        
                        if result.get('success'):
                            st.subheader("📊 Search Result:")
                            
                            result_type = result.get('type', 'unknown')
                            
                            if result_type == 'search_results':
                                st.write(f"**Found {result['count']} results:**")
                                for i, res in enumerate(result['results'], 1):
                                    with st.expander(f"Result {i} - {res['location']} (Score: {res['score']})"):
                                        st.json(res['data'])
                            
                            elif result_type == 'specific_row':
                                st.write(f"**Location:** {result['location']}")
                                if result['data']:
                                    st.json(result['data'])
                                else:
                                    st.write("No data in this row")
                            
                            elif result_type == 'column_data':
                                st.write(f"**Column:** {result['column_name']} from {result['sheet_name']}")
                                st.write(f"**Total values:** {result['total_count']}")
                                if result['data']:
                                    df = pd.DataFrame(result['data'])
                                    st.dataframe(df)
                            
                            elif result_type == 'calculation':
                                st.write("**Calculation Results:**")
                                stats = result['statistics']
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Sum", f"{stats['sum']:,.2f}")
                                col2.metric("Average", f"{stats['average']:,.2f}")
                                col3.metric("Count", stats['count'])
                        else:
                            st.error(f"❌ Error: {result.get('error')}")
                
                st.divider()
                
                st.subheader("💡 Available Search Instructions")
                st.markdown("""
                **Instructions that Gemini can use:**
                
                - `SEARCH: [query]` - TF-IDF search for relevant data
                - `GET_ROW: [sheet_name] [row_number]` - Get specific row
                - `GET_COLUMN: [sheet_name] [column_name]` - Get column data
                - `CALCULATE: [query]` - Find and calculate numeric data
                - `FIND: [query]` - Find specific information
                
                **Examples:**
                - `SEARCH: accommodation expenses` - Find accommodation-related entries
                - `GET_ROW: Balance_Sheet 5` - Get row 5 from Balance Sheet
                - `GET_COLUMN: Income_Statement Revenue` - Get all Revenue values
                - `CALCULATE: total expenses august` - Calculate total expenses
                - `FIND: highest revenue` - Find highest revenue entries
                """)
            else:
                st.info("👆 Process Excel files first to enable search testing")

if __name__ == "__main__":
    main()
