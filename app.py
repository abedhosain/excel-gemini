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
    page_title="📊 Conversational Excel AI Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class SimpleTFIDFSearchEngine:
    """Lightweight TF-IDF search engine without sklearn dependency"""
    
    def __init__(self):
        self.documents = []
        self.row_metadata = []
        self.vocabulary = {}
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

class SmartDataRetriever:
    """Smart data retrieval that works like a conversational AI interface"""
    
    def __init__(self, excel_processor: 'ExcelProcessor'):
        self.excel_processor = excel_processor
        self.search_engine = excel_processor.search_engine
        
    def process_natural_request(self, request: str) -> Dict[str, Any]:
        """Process natural language data requests like talking to Claude"""
        try:
            request_lower = request.lower().strip()
            
            # Get available files and sheets
            available_files = [f['file_name'] for f in st.session_state.files_data if 'error' not in f]
            available_sheets = []
            for file_data in st.session_state.files_data:
                if 'error' not in file_data:
                    available_sheets.extend(list(file_data['sheets'].keys()))
            
            # Pattern 1: "Show me row X" or "Get row X from sheet"
            if 'row' in request_lower and any(char.isdigit() for char in request):
                row_num = self._extract_number(request)
                sheet_name = self._find_best_sheet_match(request, available_sheets)
                file_name = self._find_file_containing_sheet(sheet_name)
                
                if row_num and sheet_name and file_name:
                    return self._get_specific_row(file_name, sheet_name, row_num)
                else:
                    return self._search_for_data(request)
            
            # Pattern 2: "Show me column" or "Get all values from column"
            elif any(word in request_lower for word in ['column', 'all values', 'list all']):
                column_name = self._extract_column_name(request)
                if column_name:
                    return self._get_column_data_smart(column_name)
                else:
                    return self._search_for_data(request)
            
            # Pattern 3: "Calculate" or "Sum" or "Total"
            elif any(word in request_lower for word in ['total', 'sum', 'average', 'calculate', 'statistics']):
                return self._calculate_from_request(request)
            
            # Pattern 4: "Find" or "Search" 
            elif any(word in request_lower for word in ['find', 'search', 'show me', 'get', 'where']):
                return self._search_for_data(request)
            
            # Pattern 5: "What is" questions
            elif any(word in request_lower for word in ['what is', 'what are', 'show', 'display']):
                return self._search_for_data(request)
            
            else:
                return self._search_for_data(request)
                
        except Exception as e:
            return {'success': False, 'error': f"Error processing request: {str(e)}"}
    
    def _extract_number(self, text: str) -> Optional[int]:
        """Extract first number from text"""
        numbers = re.findall(r'\b\d+\b', text)
        return int(numbers[0]) if numbers else None
    
    def _find_best_sheet_match(self, request: str, available_sheets: List[str]) -> Optional[str]:
        """Find the best matching sheet name from the request"""
        request_lower = request.lower()
        
        # Look for exact mentions
        for sheet in available_sheets:
            if sheet.lower() in request_lower:
                return sheet
        
        # Look for partial matches
        for sheet in available_sheets:
            sheet_words = sheet.lower().split()
            for word in sheet_words:
                if len(word) > 3 and word in request_lower:
                    return sheet
        
        # Default to first sheet
        return available_sheets[0] if available_sheets else None
    
    def _find_file_containing_sheet(self, sheet_name: str) -> Optional[str]:
        """Find which file contains the given sheet"""
        for file_data in st.session_state.files_data:
            if 'error' not in file_data and sheet_name in file_data['sheets']:
                return file_data['file_name']
        return None
    
    def _get_specific_row(self, file_name: str, sheet_name: str, row_number: int) -> Dict[str, Any]:
        """Get specific row data"""
        for metadata in self.search_engine.row_metadata:
            if (metadata['file_name'] == file_name and 
                metadata['sheet_name'] == sheet_name and 
                metadata['row_data'].get('_row_number') == row_number):
                
                # Return only non-empty, relevant data
                clean_data = {k: v for k, v in metadata['row_data'].items() 
                             if not k.startswith('_') and v and str(v).strip()}
                
                return {
                    'success': True,
                    'type': 'specific_row',
                    'data': clean_data,
                    'location': f"Row {row_number} from '{sheet_name}' in {file_name}",
                    'description': f"Data from row {row_number}:"
                }
        
        return {'success': False, 'error': f"Row {row_number} not found in {sheet_name}"}
    
    def _extract_column_name(self, request: str) -> Optional[str]:
        """Extract column name from request"""
        # Get all available column names
        all_columns = set()
        for file_data in st.session_state.files_data:
            if 'error' not in file_data:
                for sheet_data in file_data['sheets'].values():
                    if 'error' not in sheet_data:
                        all_columns.update(sheet_data['summary'].get('headers', []))
        
        # Look for column mentions in request
        request_lower = request.lower()
        for col in all_columns:
            if col.lower() in request_lower:
                return col
        
        return None
    
    def _get_column_data_smart(self, column_name: str) -> Dict[str, Any]:
        """Get column data with smart filtering"""
        all_values = []
        
        for metadata in self.search_engine.row_metadata:
            row_data = metadata['row_data']
            if column_name in row_data and row_data[column_name] and str(row_data[column_name]).strip():
                all_values.append({
                    'value': row_data[column_name],
                    'row': row_data.get('_row_number', '?'),
                    'sheet': metadata['sheet_name'],
                    'file': metadata['file_name']
                })
        
        # Limit results
        display_values = all_values[:8]  # Show first 8
        
        return {
            'success': True,
            'type': 'column_data',
            'data': display_values,
            'total_count': len(all_values),
            'column_name': column_name,
            'description': f"Found {len(all_values)} values in '{column_name}' column" + (f" (showing first 8)" if len(all_values) > 8 else "")
        }
    
    def _calculate_from_request(self, request: str) -> Dict[str, Any]:
        """Calculate statistics based on natural language request"""
        search_results = self.search_engine.search(request, top_k=30)
        
        if not search_results:
            return {'success': False, 'error': f"No relevant data found for calculation"}
        
        # Extract numeric values
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
                            'original': value,
                            'location': f"{result['file_name']}/{result['sheet_name']}/Row {result['row_index']+1}/{key}"
                        })
                    except:
                        continue
        
        if not numeric_values:
            return {'success': False, 'error': "No numeric values found for calculation"}
        
        # Calculate statistics
        stats = {
            'count': len(numeric_values),
            'sum': sum(numeric_values),
            'average': sum(numeric_values) / len(numeric_values),
            'min': min(numeric_values),
            'max': max(numeric_values),
            'median': sorted(numeric_values)[len(numeric_values)//2]
        }
        
        sample_sources = sources[:3]
        
        return {
            'success': True,
            'type': 'calculation',
            'statistics': stats,
            'sample_sources': sample_sources,
            'total_values_used': len(numeric_values),
            'description': f"Calculated from {len(numeric_values)} numeric values"
        }
    
    def _search_for_data(self, request: str) -> Dict[str, Any]:
        """Perform intelligent search and return only relevant results"""
        search_results = self.search_engine.search(request, top_k=5)  # Limit to top 5
        
        if not search_results:
            return {'success': False, 'error': f"No results found for: {request}"}
        
        # Format results concisely
        formatted_results = []
        for result in search_results:
            relevant_data = {k: v for k, v in result['row_data'].items() 
                           if not k.startswith('_') and v and str(v).strip()}
            
            formatted_results.append({
                'location': f"{result['file_name']}/{result['sheet_name']}/Row {result['row_index']+1}",
                'relevance_score': f"{result['similarity_score']:.3f}",
                'data': relevant_data
            })
        
        return {
            'success': True,
            'type': 'search_results',
            'results': formatted_results,
            'query': request,
            'description': f"Found {len(search_results)} relevant results"
        }

class ExcelProcessor:
    """Enhanced Excel file processor with TF-IDF capabilities"""
    
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
                    row_data['_row_number'] = row - 1  # 1-based numbering
                    data_rows.append(row_data)
            
            # Enhanced summary
            summary = {
                'total_rows': len(data_rows),
                'total_columns': len(headers),
                'headers': headers,
                'data_types': self._analyze_data_types(data_rows, headers),
                'column_stats': self._get_column_stats(data_rows, headers),
                'sample_data': data_rows[:5] if data_rows else []
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
            if header.startswith('_'):
                continue
                
            sample_values = [str(row.get(header, "")).strip() for row in data_rows[:20] if row.get(header, "")]
            
            if not sample_values:
                data_types[header] = "empty"
                continue
            
            numeric_count = 0
            date_count = 0
            
            for val in sample_values:
                if re.match(r'^-?\d+\.?\d*$', val.replace(',', '')):
                    numeric_count += 1
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
    
    def create_minimal_summary(self, files_data: List[Dict[str, Any]]) -> str:
        """Create minimal summary - just structure info, no actual data"""
        summary_parts = []
        
        summary_parts.append("=== EXCEL FILES STRUCTURE ===")
        summary_parts.append(f"Total files: {len(files_data)}")
        
        # Build search index first
        self.search_engine.build_index(files_data)
        total_rows = len(self.search_engine.documents)
        summary_parts.append(f"Total searchable rows: {total_rows}")
        
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
                summary_parts.append(f"      Available columns: {', '.join(summary['headers'][:15])}")  # Just column names, no data
                
                # NO SAMPLE DATA - LLM must request specific data
        
        summary_parts.append("\n💬 TO GET ACTUAL DATA:")
        summary_parts.append("You must make specific requests like:")
        summary_parts.append("- 'Show me row 5 from Balance Sheet'")
        summary_parts.append("- 'Get values from Revenue column'") 
        summary_parts.append("- 'Calculate total expenses'")
        summary_parts.append("- 'Find accommodation related entries'")
        summary_parts.append("\nNO DATA IS PROVIDED UPFRONT - YOU MUST REQUEST EVERYTHING!")
        
        return "\n".join(summary_parts)

class ConversationalGeminiLLM:
    """Gemini LLM that requests data conversationally like a user talking to Claude"""
    
    def __init__(self, api_key: str, data_retriever: SmartDataRetriever):
        genai.configure(api_key=api_key)
        self.model = "gemini-2.0-flash-exp"
        self.data_retriever = data_retriever
        
    def analyze_with_conversational_requests(self, excel_summary: str, user_query: str = "") -> str:
        """Analyze data by making conversational requests for specific information"""
        try:
            # Minimal analysis phase - no data provided upfront
            initial_prompt = f"""
You are an expert data analyst. You have access to Excel files with this STRUCTURE ONLY (no actual data):

{excel_summary}

USER QUESTION: {user_query if user_query else "Please analyze this Excel data"}

IMPORTANT: You have ONLY the file structure above. No actual data has been provided to you. To analyze anything, you MUST request specific data by making natural language requests.

You can request data like:
- "Show me row 5 from Balance Sheet"
- "Get all values from Revenue column"  
- "Calculate total of all expenses"
- "Find entries related to accommodation"
- "What is in the first 3 rows of Income Statement?"

Your task:
1. Based on the user's question and file structure, identify what specific data you need
2. Make 3-4 natural language requests to get that data
3. DO NOT make any analysis without first requesting the actual data

Format your response as:
ANALYSIS PLAN: [explain what you want to analyze]

DATA REQUESTS:
1. [specific request 1]
2. [specific request 2] 
3. [specific request 3]
4. [specific request 4]

Remember: You have NO actual data yet - you must request everything!
"""

            model = genai.GenerativeModel(self.model)
            initial_response = model.generate_content(initial_prompt)
            
            # Extract data requests from the response
            data_requests = self._extract_natural_requests(initial_response.text)
            
            # Process each request and get ONLY the specific data requested
            retrieved_data = ""
            if data_requests:
                retrieved_data = "\n📋 SPECIFIC DATA RETRIEVED:\n"
                
                for i, request in enumerate(data_requests[:4], 1):  # Limit to 4 requests
                    retrieved_data += f"\n🔹 Request {i}: \"{request}\"\n"
                    
                    result = self.data_retriever.process_natural_request(request)
                    formatted_result = self._format_conversational_result(result)
                    retrieved_data += f"Response: {formatted_result}\n"
            
            # Final analysis with ONLY the retrieved data
            if retrieved_data:
                final_prompt = f"""
You are a data analyst. Here is what you requested and received:

YOUR INITIAL PLAN:
{initial_response.text}

ACTUAL DATA YOU REQUESTED AND RECEIVED:
{retrieved_data}

Now provide comprehensive analysis using ONLY the specific data above. Do not assume any other data exists.

Provide:
1. Key insights from the specific data retrieved
2. Exact numbers and values with their locations
3. Specific findings with precise references
4. Data-driven recommendations based only on what you received
5. If you need more data for better analysis, mention what additional requests would help

Be specific and reference only the exact data points you retrieved above.
"""
                
                final_response = model.generate_content(final_prompt)
                
                return f"""
🎯 ANALYSIS APPROACH:
{initial_response.text}

{retrieved_data}

📊 ANALYSIS BASED ON RETRIEVED DATA:
{final_response.text}
"""
            else:
                return f"{initial_response.text}\n\n❌ No data was successfully retrieved. Please try different requests."
                
        except Exception as e:
            logger.error(f"Error in conversational analysis: {str(e)}")
            return f"Error in analysis: {str(e)}"
    
    def _extract_natural_requests(self, text: str) -> List[str]:
        """Extract natural language data requests from LLM response"""
        requests = []
        lines = text.split('\n')
        
        # Look for requests in various formats
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and headers
            if not line or line.startswith('#') or line.startswith('**') or 'ANALYSIS:' in line.upper():
                continue
            
            # Look for request indicators
            if any(indicator in line.lower() for indicator in [
                'show me', 'get', 'find', 'calculate', 'what is', 'how much',
                'display', 'retrieve', 'sum', 'total', 'list all'
            ]):
                # Clean up the request
                clean_request = line.replace('-', '').replace('*', '').replace('•', '').strip()
                # Remove numbering if present
                clean_request = re.sub(r'^\d+\.\s*', '', clean_request)
                if len(clean_request) > 10:  # Minimum length filter
                    requests.append(clean_request)
        
        # Also look for numbered or bulleted requests
        for line in lines:
            line = line.strip()
            if (line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')) and 
                len(line) > 10 and
                any(word in line.lower() for word in ['show', 'get', 'find', 'calculate'])):
                clean_request = line[2:].strip() if line[1] == '.' else line[1:].strip()
                requests.append(clean_request)
        
        return requests[:4]  # Limit to 4 requests
    
    def _format_conversational_result(self, result: Dict[str, Any]) -> str:
        """Format data retrieval results in a conversational way"""
        if not result.get('success', False):
            return f"❌ {result.get('error', 'No data found')}"
        
        result_type = result.get('type', 'unknown')
        
        if result_type == 'specific_row':
            data = result['data']
            location = result['location']
            if data:
                formatted_data = []
                for key, value in list(data.items())[:5]:  # Show first 5 fields
                    formatted_data.append(f"{key}: {value}")
                return f"✅ {location}\n   Data: {' | '.join(formatted_data)}"
            else:
                return f"✅ {location} - No data in this row"
        
        elif result_type == 'column_data':
            column_name = result['column_name']
            total_count = result['total_count']
            data = result['data'][:3]  # Show first 3 values
            
            values_preview = []
            for item in data:
                values_preview.append(f"Row {item['row']}: {item['value']}")
            
            return f"✅ Found {total_count} values in '{column_name}' column\n   Sample: {' | '.join(values_preview)}"
        
        elif result_type == 'calculation':
            stats = result['statistics']
            return f"✅ Calculation complete:\n   Sum: {stats['sum']:,.2f} | Average: {stats['average']:,.2f} | Count: {stats['count']} values"
        
        elif result_type == 'search_results':
            results = result['results'][:2]  # Show first 2 results
            formatted_results = []
            for res in results:
                location = res['location']
                data_preview = list(res['data'].items())[:2]  # First 2 fields
                data_str = ' | '.join([f"{k}: {v}" for k, v in data_preview])
                formatted_results.append(f"{location}: {data_str}")
            
            return f"✅ Found {len(result['results'])} results:\n   " + '\n   '.join(formatted_results)
        
        else:
            return f"✅ Data retrieved: {str(result)[:200]}..."

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
if 'data_retriever' not in st.session_state:
    st.session_state.data_retriever = None

# Main App
def main():
    st.title("📊 Conversational Excel AI Analyzer")
    st.markdown("Upload Excel files and let AI request specific data conversationally - just like talking to Claude!")
    
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
        
        # Manual Search Section
        st.header("🔍 Manual Search")
        
        if st.session_state.files_data:
            search_query = st.text_input(
                "Search your data:",
                placeholder="e.g., sales revenue Q1, customer data, highest values"
            )
            
            search_button = st.button("🔍 Search", type="secondary")
            
            if search_button and search_query:
                with st.spinner("Searching..."):
                    results = st.session_state.excel_processor.search_engine.search(search_query, top_k=5)
                    st.session_state.search_results = results
                    
                if results:
                    st.success(f"Found {len(results)} results")
                    
                    # Show top results in sidebar
                    for i, result in enumerate(results[:2], 1):
                        with st.expander(f"Result {i} (Score: {result['similarity_score']:.3f})"):
                            st.write(f"**File:** {result['file_name']}")
                            st.write(f"**Sheet:** {result['sheet_name']}")
                            st.write(f"**Row:** {result['row_index'] + 1}")
                            # Show only first 3 data items
                            clean_data = {k: v for k, v in list(result['row_data'].items())[:3] if not k.startswith('_') and v}
                            st.json(clean_data, expanded=False)
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
            st.session_state.data_retriever = None
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
            placeholder="e.g., What are the key financial insights? Find the highest expenses. Analyze revenue trends.",
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
            help="Process Excel files and build search index"
        )
    
    with col2:
        analyze_button = st.button(
            "🤖 Conversational AI Analysis",
            type="primary",
            disabled=not (st.session_state.excel_summary and api_key),
            help="AI will request specific data conversationally"
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
            
            # Initialize data retriever
            st.session_state.data_retriever = SmartDataRetriever(st.session_state.excel_processor)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete! Ready for conversational analysis.")
            
            st.success(f"Successfully processed {len(files_data)} files with {len(st.session_state.excel_processor.search_engine.documents)} searchable rows!")
    
    # AI Analysis with Conversational Requests
    if analyze_button and st.session_state.excel_summary and api_key:
        with st.spinner("🤖 AI is analyzing data and making conversational requests..."):
            try:
                if not st.session_state.data_retriever:
                    st.session_state.data_retriever = SmartDataRetriever(st.session_state.excel_processor)
                
                conversational_llm = ConversationalGeminiLLM(api_key, st.session_state.data_retriever)
                
                analysis = conversational_llm.analyze_with_conversational_requests(
                    st.session_state.excel_summary, 
                    user_query
                )
                st.session_state.llm_analysis = analysis
                st.success("✅ Conversational AI analysis complete!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Results section
    if st.session_state.excel_summary or st.session_state.llm_analysis or st.session_state.search_results:
        st.divider()
        st.header("📊 Results")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Summary", "🤖 AI Analysis", "🔍 Search Results", "💬 Test Requests"])
        
        with tab1:
            if st.session_state.excel_summary:
                st.subheader("📊 Excel Data Summary")
                st.text_area(
                    "Processed data summary",
                    value=st.session_state.excel_summary,
                    height=400,
                    disabled=True
                )
                
                st.download_button(
                    label="📥 Download Summary",
                    data=st.session_state.excel_summary,
                    file_name="excel_summary_conversational.txt",
                    mime="text/plain"
                )
            else:
                st.info("👆 Process Excel files first")
        
        with tab2:
            if st.session_state.llm_analysis:
                st.subheader("🤖 Conversational AI Analysis")
                st.markdown(st.session_state.llm_analysis)
                
                st.download_button(
                    label="📥 Download Analysis",
                    data=st.session_state.llm_analysis,
                    file_name="ai_analysis_conversational.txt",
                    mime="text/plain"
                )
            else:
                st.info("👆 Run AI analysis to see insights")
        
        with tab3:
            if st.session_state.search_results:
                st.subheader("🔍 Manual Search Results")
                
                for i, result in enumerate(st.session_state.search_results, 1):
                    with st.expander(f"📄 Result {i} - Similarity: {result['similarity_score']:.3f}"):
                        col1, col2, col3 = st.columns([1, 1, 1])
                        col1.metric("File", result['file_name'])
                        col2.metric("Sheet", result['sheet_name'])
                        col3.metric("Row Number", result['row_index'] + 1)
                        
                        st.subheader("Row Data:")
                        # Show only non-empty data
                        clean_data = {k: v for k, v in result['row_data'].items() if not k.startswith('_') and v}
                        if clean_data:
                            row_df = pd.DataFrame([clean_data])
                            st.dataframe(row_df, use_container_width=True)
                        else:
                            st.write("No data to display")
            else:
                st.info("👆 Use the search feature in the sidebar")
        
        with tab4:
            if st.session_state.data_retriever:
                st.subheader("💬 Test Conversational Data Requests")
                st.markdown("Test how the AI requests data - just like talking to Claude:")
                
                # Manual data request interface
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    manual_request = st.text_input(
                        "Enter a natural language data request:",
                        placeholder="e.g., Show me row 5 from balance sheet, Get all revenue values, Calculate total expenses",
                        key="manual_request"
                    )
                
                with col2:
                    if st.button("💬 Ask", key="execute_request"):
                        if manual_request and st.session_state.data_retriever:
                            with st.spinner("Processing request..."):
                                result = st.session_state.data_retriever.process_natural_request(manual_request)
                                
                                # Format result nicely
                                if result.get('success'):
                                    st.success("✅ Request processed successfully!")
                                    
                                    result_type = result.get('type', 'unknown')
                                    
                                    if result_type == 'specific_row':
                                        st.write(f"**Location:** {result['location']}")
                                        if result['data']:
                                            st.json(result['data'])
                                        else:
                                            st.write("No data in this row")
                                    
                                    elif result_type == 'column_data':
                                        st.write(f"**Column:** {result['column_name']}")
                                        st.write(f"**Total Values:** {result['total_count']}")
                                        if result['data']:
                                            df = pd.DataFrame(result['data'])
                                            st.dataframe(df)
                                    
                                    elif result_type == 'calculation':
                                        st.write("**Statistics:**")
                                        stats = result['statistics']
                                        col1, col2, col3 = st.columns(3)
                                        col1.metric("Sum", f"{stats['sum']:,.2f}")
                                        col2.metric("Average", f"{stats['average']:,.2f}")
                                        col3.metric("Count", stats['count'])
                                    
                                    elif result_type == 'search_results':
                                        st.write(f"**Found {len(result['results'])} results:**")
                                        for res in result['results']:
                                            st.write(f"📍 {res['location']}")
                                            if res['data']:
                                                st.json(res['data'])
                                else:
                                    st.error(f"❌ {result.get('error', 'No results found')}")
                
                st.divider()
                
                # Example requests
                st.subheader("💡 Example Conversational Requests")
                example_requests = [
                    "Show me row 3 from the first sheet",
                    "Get all values from the Amount column",
                    "Calculate the total of all expenses",
                    "Find all entries related to revenue",
                    "What is in row 10 of the balance sheet?"
                ]
                
                for i, example in enumerate(example_requests, 1):
                    col1, col2 = st.columns([3, 1])
                    col1.write(f"{i}. {example}")
                    if col2.button("Try", key=f"example_{i}"):
                        result = st.session_state.data_retriever.process_natural_request(example)
                        if result.get('success'):
                            st.success(f"✅ Found: {result.get('description', 'Data retrieved')}")
                        else:
                            st.error(f"❌ {result.get('error', 'No results')}")
            else:
                st.info("Process Excel files first to enable conversational requests")

if __name__ == "__main__":
    main()
