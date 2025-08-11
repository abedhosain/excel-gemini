import streamlit as st
import json
import os
from typing import List, Dict, Any, Optional, Tuple
import openpyxl
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
import google.generativeai as genai
import tempfile
import logging
from io import BytesIO
import pandas as pd
import math
import re
from collections import Counter, defaultdict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="📊 AI Excel Editor - Gemini Instructor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ExcelSearchEngine:
    """Advanced search engine for Excel data with editing capabilities"""
    
    def __init__(self):
        self.workbooks = {}  # Store actual workbook objects
        self.sheet_data = {}  # Store processed data for search
        self.file_paths = {}  # Store temporary file paths
        self.idf_scores = {}
        self.tfidf_vectors = []
        self.row_metadata = []
        
    def _tokenize(self, text: str) -> List[str]:
        """Advanced tokenization"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        return [token for token in tokens if token not in stop_words and len(token) > 1]
    
    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency"""
        tf = {}
        total_tokens = len(tokens)
        if total_tokens == 0:
            return tf
        token_counts = Counter(tokens)
        
        for token, count in token_counts.items():
            tf[token] = count / total_tokens
        
        return tf
    
    def _compute_idf(self) -> None:
        """Compute inverse document frequency"""
        doc_count = len(self.row_metadata)
        if doc_count == 0:
            return
            
        term_doc_count = defaultdict(int)
        
        for metadata in self.row_metadata:
            doc_text = " ".join([f"{k}: {v}" for k, v in metadata['row_data'].items() 
                               if not k.startswith('_') and v and str(v).strip()])
            tokens = self._tokenize(doc_text)
            unique_tokens = set(tokens)
            for token in unique_tokens:
                term_doc_count[token] += 1
        
        for term, count in term_doc_count.items():
            self.idf_scores[term] = math.log(doc_count / count) if count > 0 else 0
    
    def load_excel_files(self, uploaded_files) -> None:
        """Load Excel files and build search index"""
        self.workbooks = {}
        self.sheet_data = {}
        self.file_paths = {}
        self.row_metadata = []
        
        for uploaded_file in uploaded_files:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            self.file_paths[uploaded_file.name] = tmp_path
            
            # Load workbook
            workbook = load_workbook(tmp_path, data_only=False)
            self.workbooks[uploaded_file.name] = workbook
            
            # Process each sheet
            file_sheets = {}
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                sheet_data = self._process_sheet(sheet, uploaded_file.name, sheet_name)
                file_sheets[sheet_name] = sheet_data
            
            self.sheet_data[uploaded_file.name] = file_sheets
        
        # Build search index
        self._build_search_index()
    
    def _process_sheet(self, sheet, file_name: str, sheet_name: str) -> Dict[str, Any]:
        """Process sheet and extract all data"""
        try:
            if sheet.max_row is None or sheet.max_row == 0:
                return {'headers': [], 'data': [], 'max_row': 0, 'max_col': 0}
            
            max_row = sheet.max_row
            max_col = sheet.max_column or 0
            
            # Extract headers (row 1)
            headers = []
            for col in range(1, max_col + 1):
                cell_value = sheet.cell(row=1, column=col).value
                if cell_value is not None:
                    header = str(cell_value).strip()
                    headers.append(header if header else f"Column_{col}")
                else:
                    headers.append(f"Column_{col}")
            
            # Extract all data
            data_rows = []
            for row_num in range(2, max_row + 1):
                row_data = {}
                has_data = False
                
                for col_num, header in enumerate(headers, 1):
                    cell = sheet.cell(row=row_num, column=col_num)
                    cell_value = cell.value
                    
                    if cell_value is not None:
                        row_data[header] = str(cell_value).strip()
                        has_data = True
                    else:
                        row_data[header] = ""
                
                if has_data:
                    row_data['_row_number'] = row_num
                    row_data['_file_name'] = file_name
                    row_data['_sheet_name'] = sheet_name
                    data_rows.append(row_data)
                    
                    # Add to metadata for search
                    self.row_metadata.append({
                        'file_name': file_name,
                        'sheet_name': sheet_name,
                        'row_number': row_num,
                        'excel_row': row_num,  # Actual Excel row number
                        'row_data': row_data,
                        'headers': headers
                    })
            
            return {
                'headers': headers,
                'data': data_rows,
                'max_row': max_row,
                'max_col': max_col
            }
            
        except Exception as e:
            logger.error(f"Error processing sheet {sheet_name}: {str(e)}")
            return {'headers': [], 'data': [], 'max_row': 0, 'max_col': 0, 'error': str(e)}
    
    def _build_search_index(self) -> None:
        """Build TF-IDF search index"""
        if not self.row_metadata:
            return
            
        self._compute_idf()
        
        self.tfidf_vectors = []
        for metadata in self.row_metadata:
            doc_text = " ".join([f"{k}: {v}" for k, v in metadata['row_data'].items() 
                               if not k.startswith('_') and v and str(v).strip()])
            tokens = self._tokenize(doc_text)
            tfidf_vector = self._compute_tfidf_vector(tokens)
            self.tfidf_vectors.append(tfidf_vector)
    
    def _compute_tfidf_vector(self, tokens: List[str]) -> Dict[str, float]:
        """Compute TF-IDF vector"""
        tf_scores = self._compute_tf(tokens)
        tfidf_vector = {}
        
        for token in tf_scores:
            if token in self.idf_scores:
                tfidf_vector[token] = tf_scores[token] * self.idf_scores[token]
        
        return tfidf_vector
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Compute cosine similarity"""
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        if not common_terms:
            return 0.0
        
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        mag1 = math.sqrt(sum(val**2 for val in vec1.values()))
        mag2 = math.sqrt(sum(val**2 for val in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant rows"""
        if not self.row_metadata:
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
                results.append(result)
        
        return results
    
    def get_sheet_dataframe(self, file_name: str, sheet_name: str) -> pd.DataFrame:
        """Get sheet data as DataFrame for display"""
        if file_name not in self.sheet_data or sheet_name not in self.sheet_data[file_name]:
            return pd.DataFrame()
        
        sheet_data = self.sheet_data[file_name][sheet_name]
        if not sheet_data['data']:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df_data = []
        for row in sheet_data['data']:
            clean_row = {k: v for k, v in row.items() if not k.startswith('_')}
            df_data.append(clean_row)
        
        return pd.DataFrame(df_data)
    
    def get_file_structure(self) -> Dict[str, Any]:
        """Get structure of all loaded files"""
        structure = {}
        for file_name, sheets in self.sheet_data.items():
            structure[file_name] = {}
            for sheet_name, sheet_data in sheets.items():
                structure[file_name][sheet_name] = {
                    'headers': sheet_data['headers'],
                    'row_count': len(sheet_data['data']),
                    'max_row': sheet_data['max_row'],
                    'max_col': sheet_data['max_col']
                }
        return structure

class ExcelEditor:
    """Excel editing engine that executes Gemini's instructions"""
    
    def __init__(self, search_engine: ExcelSearchEngine):
        self.search_engine = search_engine
        self.edit_history = []
    
    def execute_instruction(self, instruction: str) -> Dict[str, Any]:
        """Execute editing instruction from Gemini"""
        try:
            instruction = instruction.strip()
            
            # Parse different instruction types
            if instruction.startswith('SEARCH:'):
                query = instruction[7:].strip()
                return self._search_data(query)
            
            elif instruction.startswith('READ_ROW:'):
                params = instruction[9:].strip()
                return self._read_row(params)
            
            elif instruction.startswith('READ_CELL:'):
                params = instruction[11:].strip()
                return self._read_cell(params)
            
            elif instruction.startswith('EDIT_CELL:'):
                params = instruction[11:].strip()
                return self._edit_cell(params)
            
            elif instruction.startswith('EDIT_ROW:'):
                params = instruction[10:].strip()
                return self._edit_row(params)
            
            elif instruction.startswith('INSERT_ROW:'):
                params = instruction[12:].strip()
                return self._insert_row(params)
            
            elif instruction.startswith('DELETE_ROW:'):
                params = instruction[12:].strip()
                return self._delete_row(params)
            
            elif instruction.startswith('FIND_AND_REPLACE:'):
                params = instruction[17:].strip()
                return self._find_and_replace(params)
            
            elif instruction.startswith('GET_STRUCTURE:'):
                return self._get_structure()
            
            else:
                # Default to search
                return self._search_data(instruction)
                
        except Exception as e:
            return {'success': False, 'error': f"Error executing instruction: {str(e)}"}
    
    def _search_data(self, query: str) -> Dict[str, Any]:
        """Search for data using TF-IDF"""
        results = self.search_engine.search(query, top_k=5)
        
        if not results:
            return {'success': False, 'error': f"No results found for: {query}"}
        
        formatted_results = []
        for result in results:
            clean_data = {k: v for k, v in result['row_data'].items() 
                         if not k.startswith('_') and v and str(v).strip()}
            
            formatted_results.append({
                'location': f"{result['file_name']}/{result['sheet_name']}/Row {result['excel_row']}",
                'score': f"{result['similarity_score']:.3f}",
                'data': clean_data,
                'metadata': {
                    'file_name': result['file_name'],
                    'sheet_name': result['sheet_name'],
                    'row_number': result['excel_row']
                }
            })
        
        return {
            'success': True,
            'type': 'search_results',
            'results': formatted_results,
            'query': query,
            'count': len(results)
        }
    
    def _read_row(self, params: str) -> Dict[str, Any]:
        """Read specific row: 'filename sheetname rownumber'"""
        parts = params.split()
        if len(parts) < 3:
            return {'success': False, 'error': 'Invalid parameters. Use: filename sheetname rownumber'}
        
        file_name = parts[0]
        sheet_name = parts[1]
        try:
            row_number = int(parts[2])
        except ValueError:
            return {'success': False, 'error': 'Invalid row number'}
        
        # Find the row in metadata
        for metadata in self.search_engine.row_metadata:
            if (metadata['file_name'] == file_name and 
                metadata['sheet_name'] == sheet_name and 
                metadata['excel_row'] == row_number):
                
                clean_data = {k: v for k, v in metadata['row_data'].items() 
                             if not k.startswith('_')}
                
                return {
                    'success': True,
                    'type': 'row_data',
                    'data': clean_data,
                    'location': f"{file_name}/{sheet_name}/Row {row_number}"
                }
        
        return {'success': False, 'error': f"Row {row_number} not found in {file_name}/{sheet_name}"}
    
    def _read_cell(self, params: str) -> Dict[str, Any]:
        """Read specific cell: 'filename sheetname rownumber columnname'"""
        parts = params.split()
        if len(parts) < 4:
            return {'success': False, 'error': 'Invalid parameters. Use: filename sheetname rownumber columnname'}
        
        file_name = parts[0]
        sheet_name = parts[1]
        try:
            row_number = int(parts[2])
        except ValueError:
            return {'success': False, 'error': 'Invalid row number'}
        
        column_name = ' '.join(parts[3:])
        
        # Get the workbook
        if file_name not in self.search_engine.workbooks:
            return {'success': False, 'error': f"File {file_name} not found"}
        
        workbook = self.search_engine.workbooks[file_name]
        if sheet_name not in workbook.sheetnames:
            return {'success': False, 'error': f"Sheet {sheet_name} not found"}
        
        sheet = workbook[sheet_name]
        
        # Find column by name
        col_num = None
        for col in range(1, sheet.max_column + 1):
            header_cell = sheet.cell(row=1, column=col)
            if header_cell.value and str(header_cell.value).strip() == column_name:
                col_num = col
                break
        
        if col_num is None:
            return {'success': False, 'error': f"Column '{column_name}' not found"}
        
        # Read the cell
        cell = sheet.cell(row=row_number, column=col_num)
        cell_value = cell.value if cell.value is not None else ""
        
        return {
            'success': True,
            'type': 'cell_data',
            'value': str(cell_value),
            'location': f"{file_name}/{sheet_name}/Row {row_number}/{column_name}"
        }
    
    def _edit_cell(self, params: str) -> Dict[str, Any]:
        """Edit specific cell: 'filename sheetname rownumber columnname newvalue'"""
        parts = params.split()
        if len(parts) < 5:
            return {'success': False, 'error': 'Invalid parameters. Use: filename sheetname rownumber columnname newvalue'}
        
        file_name = parts[0]
        sheet_name = parts[1]
        try:
            row_number = int(parts[2])
        except ValueError:
            return {'success': False, 'error': 'Invalid row number'}
        
        column_name = parts[3]
        new_value = ' '.join(parts[4:])
        
        # Get the workbook
        if file_name not in self.search_engine.workbooks:
            return {'success': False, 'error': f"File {file_name} not found"}
        
        workbook = self.search_engine.workbooks[file_name]
        if sheet_name not in workbook.sheetnames:
            return {'success': False, 'error': f"Sheet {sheet_name} not found"}
        
        sheet = workbook[sheet_name]
        
        # Find column by name
        col_num = None
        for col in range(1, sheet.max_column + 1):
            header_cell = sheet.cell(row=1, column=col)
            if header_cell.value and str(header_cell.value).strip() == column_name:
                col_num = col
                break
        
        if col_num is None:
            return {'success': False, 'error': f"Column '{column_name}' not found"}
        
        # Get old value
        cell = sheet.cell(row=row_number, column=col_num)
        old_value = cell.value if cell.value is not None else ""
        
        # Try to convert to appropriate type
        try:
            if new_value.isdigit():
                new_value = int(new_value)
            elif new_value.replace('.', '').isdigit():
                new_value = float(new_value)
        except:
            pass  # Keep as string
        
        # Edit the cell
        cell.value = new_value
        
        # Save the workbook
        file_path = self.search_engine.file_paths[file_name]
        workbook.save(file_path)
        
        # Record edit history
        edit_record = {
            'timestamp': datetime.now().isoformat(),
            'action': 'edit_cell',
            'location': f"{file_name}/{sheet_name}/Row {row_number}/{column_name}",
            'old_value': str(old_value),
            'new_value': str(new_value)
        }
        self.edit_history.append(edit_record)
        
        # Update search engine data
        self._update_search_data_after_edit(file_name, sheet_name, row_number, column_name, new_value)
        
        return {
            'success': True,
            'type': 'cell_edited',
            'location': f"{file_name}/{sheet_name}/Row {row_number}/{column_name}",
            'old_value': str(old_value),
            'new_value': str(new_value)
        }
    
    def _update_search_data_after_edit(self, file_name: str, sheet_name: str, row_number: int, column_name: str, new_value: Any):
        """Update search engine data after edit"""
        for metadata in self.search_engine.row_metadata:
            if (metadata['file_name'] == file_name and 
                metadata['sheet_name'] == sheet_name and 
                metadata['excel_row'] == row_number):
                
                metadata['row_data'][column_name] = str(new_value)
                break
        
        # Update sheet data as well
        if file_name in self.search_engine.sheet_data and sheet_name in self.search_engine.sheet_data[file_name]:
            sheet_data = self.search_engine.sheet_data[file_name][sheet_name]
            for row in sheet_data['data']:
                if row.get('_row_number') == row_number:
                    row[column_name] = str(new_value)
                    break
        
        # Rebuild search index
        self.search_engine._build_search_index()
    
    def _edit_row(self, params: str) -> Dict[str, Any]:
        """Edit entire row: 'filename sheetname rownumber column1:value1 column2:value2 ...'"""
        parts = params.split()
        if len(parts) < 4:
            return {'success': False, 'error': 'Invalid parameters'}
        
        file_name = parts[0]
        sheet_name = parts[1]
        try:
            row_number = int(parts[2])
        except ValueError:
            return {'success': False, 'error': 'Invalid row number'}
        
        # Parse column:value pairs
        updates = {}
        for part in parts[3:]:
            if ':' in part:
                col, val = part.split(':', 1)
                updates[col] = val
        
        if not updates:
            return {'success': False, 'error': 'No valid column:value pairs found'}
        
        results = []
        for column_name, new_value in updates.items():
            result = self._edit_cell(f"{file_name} {sheet_name} {row_number} {column_name} {new_value}")
            results.append(result)
        
        return {
            'success': True,
            'type': 'row_edited',
            'location': f"{file_name}/{sheet_name}/Row {row_number}",
            'updates': len(updates),
            'results': results
        }
    
    def _get_structure(self) -> Dict[str, Any]:
        """Get structure of all files"""
        structure = self.search_engine.get_file_structure()
        return {
            'success': True,
            'type': 'structure',
            'data': structure
        }
    
    def _insert_row(self, params: str) -> Dict[str, Any]:
        """Insert new row (placeholder for future implementation)"""
        return {'success': False, 'error': 'Insert row not yet implemented'}
    
    def _delete_row(self, params: str) -> Dict[str, Any]:
        """Delete row (placeholder for future implementation)"""
        return {'success': False, 'error': 'Delete row not yet implemented'}
    
    def _find_and_replace(self, params: str) -> Dict[str, Any]:
        """Find and replace (placeholder for future implementation)"""
        return {'success': False, 'error': 'Find and replace not yet implemented'}
    
    def get_edit_history(self) -> List[Dict[str, Any]]:
        """Get edit history"""
        return self.edit_history
    
    def save_files(self) -> Dict[str, Any]:
        """Save all modified files"""
        try:
            saved_files = []
            for file_name, workbook in self.search_engine.workbooks.items():
                file_path = self.search_engine.file_paths[file_name]
                workbook.save(file_path)
                saved_files.append(file_name)
            
            return {
                'success': True,
                'saved_files': saved_files,
                'count': len(saved_files)
            }
        except Exception as e:
            return {'success': False, 'error': f"Error saving files: {str(e)}"}

class GeminiInstructor:
    """Gemini acts as intelligent instructor for Excel operations"""
    
    def __init__(self, api_key: str, excel_editor: ExcelEditor):
        genai.configure(api_key=api_key)
        self.model = "gemini-2.0-flash-exp"
        self.excel_editor = excel_editor
    
    def process_user_request(self, user_prompt: str, file_structure: Dict[str, Any]) -> str:
        """Process user request and generate instructions for Excel editor"""
        try:
            system_prompt = f"""
You are an expert Excel AI instructor. You have access to Excel files through a specialized editing engine.

AVAILABLE FILES STRUCTURE:
{json.dumps(file_structure, indent=2)}

AVAILABLE INSTRUCTIONS YOU CAN GIVE:
1. SEARCH: [query] - Search for data using AI
2. READ_ROW: [filename] [sheetname] [rownumber] - Read specific row
3. READ_CELL: [filename] [sheetname] [rownumber] [columnname] - Read specific cell
4. EDIT_CELL: [filename] [sheetname] [rownumber] [columnname] [newvalue] - Edit cell
5. EDIT_ROW: [filename] [sheetname] [rownumber] [col1:val1] [col2:val2] - Edit multiple cells in row
6. GET_STRUCTURE: - Get file structure

USER REQUEST: {user_prompt}

Your task:
1. Understand what the user wants to do
2. Plan the necessary Excel operations
3. Generate specific instructions for the Excel editor
4. Execute the instructions step by step
5. Provide a comprehensive response

Start by understanding the request and planning your approach. Then execute the necessary instructions.

Format your response as:
UNDERSTANDING: [what you understand from the user request]
PLAN: [your step-by-step plan]
INSTRUCTIONS:
1. [instruction 1]
2. [instruction 2]
...

Then I'll execute these instructions and analyze the results.
"""

            model = genai.GenerativeModel(self.model)
            planning_response = model.generate_content(system_prompt)
            
            # Extract instructions
            instructions = self._extract_instructions(planning_response.text)
            
            # Execute instructions
            execution_results = ""
            if instructions:
                execution_results = "\n🔧 EXECUTION RESULTS:\n"
                
                for i, instruction in enumerate(instructions, 1):
                    execution_results += f"\n📋 Instruction {i}: {instruction}\n"
                    
                    result = self.excel_editor.execute_instruction(instruction)
                    formatted_result = self._format_result(result)
                    execution_results += f"✅ Result: {formatted_result}\n"
            
            # Final analysis
            if execution_results:
                analysis_prompt = f"""
Your initial planning:
{planning_response.text}

Execution results:
{execution_results}

Now provide a comprehensive response to the user that includes:
1. Summary of what was accomplished
2. Key findings from the data
3. Any changes made to the Excel files
4. Specific details with numbers and locations
5. Recommendations or next steps

Make it clear and actionable for the user.
"""
                
                final_response = model.generate_content(analysis_prompt)
                
                return f"""
🎯 PLANNING & UNDERSTANDING:
{planning_response.text}

{execution_results}

📊 COMPREHENSIVE ANALYSIS:
{final_response.text}
"""
            else:
                return f"{planning_response.text}\n\n❌ No instructions were successfully executed."
                
        except Exception as e:
            logger.error(f"Error in Gemini instruction: {str(e)}")
            return f"Error processing request: {str(e)}"
    
    def _extract_instructions(self, text: str) -> List[str]:
        """Extract instructions from Gemini's response"""
        instructions = []
        lines = text.split('\n')
        
        in_instructions_section = False
        for line in lines:
            line = line.strip()
            
            if 'INSTRUCTIONS:' in line.upper():
                in_instructions_section = True
                continue
            
            if in_instructions_section and line:
                # Look for instruction patterns
                if any(cmd in line.upper() for cmd in ['SEARCH:', 'READ_ROW:', 'READ_CELL:', 'EDIT_CELL:', 'EDIT_ROW:', 'GET_STRUCTURE:']):
                    clean_instruction = line
                    # Remove numbering if present
                    clean_instruction = re.sub(r'^\d+\.\s*', '', clean_instruction)
                    instructions.append(clean_instruction.strip())
                elif line.startswith(('1.', '2.', '3.', '4.', '5.')):
                    # Extract instruction from numbered list
                    clean_instruction = re.sub(r'^\d+\.\s*', '', line)
                    if any(cmd in clean_instruction.upper() for cmd in ['SEARCH:', 'READ_ROW:', 'READ_CELL:', 'EDIT_CELL:', 'EDIT_ROW:', 'GET_STRUCTURE:']):
                        instructions.append(clean_instruction.strip())
        
        return instructions[:10]  # Limit to 10 instructions
    
    def _format_result(self, result: Dict[str, Any]) -> str:
        """Format execution results"""
        if not result.get('success', False):
            return f"❌ {result.get('error', 'Unknown error')}"
        
        result_type = result.get('type', 'unknown')
        
        if result_type == 'search_results':
            results = result['results'][:3]  # Show top 3
            formatted = f"Found {result['count']} results:\n"
            for res in results:
                formatted += f"   📍 {res['location']} (Score: {res['score']})\n"
                data_items = list(res['data'].items())[:3]
                data_str = ' | '.join([f"{k}: {v}" for k, v in data_items])
                formatted += f"   📊 {data_str}\n"
            return formatted
        
        elif result_type == 'row_data':
            data = result['data']
            if data:
                data_items = list(data.items())[:5]
                data_str = ' | '.join([f"{k}: {v}" for k, v in data_items])
                return f"Row data from {result['location']}: {data_str}"
            else:
                return f"No data in {result['location']}"
        
        elif result_type == 'cell_data':
            return f"Cell value at {result['location']}: {result['value']}"
        
        elif result_type == 'cell_edited':
            return f"✏️ Edited {result['location']}: '{result['old_value']}' → '{result['new_value']}'"
        
        elif result_type == 'row_edited':
            return f"✏️ Edited {result['updates']} cells in {result['location']}"
        
        elif result_type == 'structure':
            return f"File structure retrieved with {len(result['data'])} files"
        
        else:
            return f"✅ {str(result)[:100]}..."

# Initialize session state
if 'search_engine' not in st.session_state:
    st.session_state.search_engine = ExcelSearchEngine()
if 'excel_editor' not in st.session_state:
    st.session_state.excel_editor = None
if 'gemini_instructor' not in st.session_state:
    st.session_state.gemini_instructor = None
if 'files_loaded' not in st.session_state:
    st.session_state.files_loaded = False
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = ""
if 'edit_history' not in st.session_state:
    st.session_state.edit_history = []

# API Key
API_KEY = "AIzaSyCSDx-q3PgkvMQktdi4tScbT1wOLgZ9jQg"

def main():
    st.title("📊 AI Excel Editor - Gemini Instructor")
    st.markdown("**Gemini understands your requests and instructs specialized ML models to search and edit Excel files perfectly!**")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        # File status
        if st.session_state.files_loaded:
            file_count = len(st.session_state.search_engine.workbooks)
            row_count = len(st.session_state.search_engine.row_metadata)
            st.success(f"✅ {file_count} files loaded")
            st.info(f"📊 {row_count} rows indexed")
        else:
            st.warning("⚠️ No files loaded")
        
        # Editor status
        if st.session_state.excel_editor:
            edit_count = len(st.session_state.excel_editor.edit_history)
            st.info(f"✏️ {edit_count} edits made")
        
        st.divider()
        
        # Quick actions
        st.header("🚀 Quick Actions")
        if st.session_state.files_loaded:
            if st.button("💾 Save All Changes"):
                result = st.session_state.excel_editor.save_files()
                if result['success']:
                    st.success(f"✅ Saved {result['count']} files")
                else:
                    st.error(f"❌ {result['error']}")
            
            if st.button("📋 View Edit History"):
                history = st.session_state.excel_editor.get_edit_history()
                if history:
                    st.write("**Recent Edits:**")
                    for edit in history[-5:]:  # Show last 5 edits
                        st.text(f"• {edit['action']}: {edit['location']}")
                else:
                    st.info("No edits made yet")
        
        st.divider()
        
        # Test instructions
        if st.session_state.excel_editor:
            st.header("🧪 Test Instructions")
            test_instruction = st.text_input(
                "Test instruction:",
                placeholder="SEARCH: revenue data"
            )
            if st.button("Execute Test") and test_instruction:
                result = st.session_state.excel_editor.execute_instruction(test_instruction)
                if result['success']:
                    st.success("✅ Executed successfully")
                    st.json(result, expanded=False)
                else:
                    st.error(f"❌ {result['error']}")
        
        if st.button("🗑️ Clear All", type="secondary"):
            for key in ['files_loaded', 'analysis_result', 'edit_history']:
                if key in st.session_state:
                    del st.session_state[key]
            st.session_state.search_engine = ExcelSearchEngine()
            st.session_state.excel_editor = None
            st.session_state.gemini_instructor = None
            st.rerun()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Excel Files")
        uploaded_files = st.file_uploader(
            "Upload Excel files for AI analysis and editing",
            type=['xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload Excel files that Gemini will analyze and edit"
        )
        
        if uploaded_files:
            st.success(f"📄 {len(uploaded_files)} file(s) ready")
            
            if st.button("🔄 Load Files", type="primary"):
                with st.spinner("Loading and indexing Excel files..."):
                    st.session_state.search_engine.load_excel_files(uploaded_files)
                    st.session_state.excel_editor = ExcelEditor(st.session_state.search_engine)
                    st.session_state.gemini_instructor = GeminiInstructor(API_KEY, st.session_state.excel_editor)
                    st.session_state.files_loaded = True
                    
                st.success("✅ Files loaded and ready for AI operations!")
                st.rerun()
    
    with col2:
        st.header("🤖 AI Request")
        user_request = st.text_area(
            "Tell Gemini what you want to do with your Excel files:",
            placeholder="""Examples:
• Find all revenue data and show me the totals
• Edit the sales figures in Q1 sheet to increase by 10%
• Search for customer information and update phone numbers
• Find expenses above $1000 and highlight them
• Add a new column for profit margins and calculate them""",
            height=150
        )
        
        if st.button("🎯 Execute AI Request", type="primary", disabled=not (st.session_state.files_loaded and user_request)):
            with st.spinner("🤖 Gemini is understanding your request and instructing the Excel editor..."):
                try:
                    file_structure = st.session_state.search_engine.get_file_structure()
                    result = st.session_state.gemini_instructor.process_user_request(user_request, file_structure)
                    st.session_state.analysis_result = result
                    st.success("✅ AI request completed!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Excel File Viewer
    if st.session_state.files_loaded:
        st.divider()
        st.header("📊 Excel File Viewer")
        
        # File and sheet selector
        file_names = list(st.session_state.search_engine.workbooks.keys())
        selected_file = st.selectbox("Select File:", file_names)
        
        if selected_file:
            sheet_names = list(st.session_state.search_engine.sheet_data[selected_file].keys())
            selected_sheet = st.selectbox("Select Sheet:", sheet_names)
            
            if selected_sheet:
                # Display sheet data
                df = st.session_state.search_engine.get_sheet_dataframe(selected_file, selected_sheet)
                
                if not df.empty:
                    st.subheader(f"📋 {selected_file} / {selected_sheet}")
                    
                    # Show basic info
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Rows", len(df))
                    col2.metric("Columns", len(df.columns))
                    col3.metric("Total Cells", len(df) * len(df.columns))
                    
                    # Display the dataframe
                    st.dataframe(
                        df,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download option
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download as CSV",
                        data=csv,
                        file_name=f"{selected_file}_{selected_sheet}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No data to display in this sheet")
    
    # Results
    if st.session_state.analysis_result:
        st.divider()
        st.header("🎯 AI Analysis Results")
        
        st.markdown(st.session_state.analysis_result)
        
        # Download results
        st.download_button(
            "📥 Download Analysis Report",
            data=st.session_state.analysis_result,
            file_name="ai_excel_analysis.txt",
            mime="text/plain"
        )
    
    # Instructions and Examples
    if not st.session_state.files_loaded:
        st.divider()
        st.header("💡 How It Works")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔄 Process Flow")
            st.markdown("""
            1. **Upload Excel files** - Your data is loaded and indexed
            2. **Make AI request** - Tell Gemini what you want to do
            3. **Gemini understands** - Analyzes your request intelligently  
            4. **ML model executes** - Specialized engine searches and edits
            5. **View results** - See changes and analysis in real-time
            """)
        
        with col2:
            st.subheader("🤖 AI Capabilities")
            st.markdown("""
            **Search & Analysis:**
            - Find specific data using natural language
            - Calculate totals, averages, trends
            - Identify patterns and insights
            
            **Editing & Modification:**
            - Edit individual cells or entire rows
            - Update values based on conditions
            - Apply formulas and calculations
            
            **Intelligence:**
            - Understands context and intent
            - Handles complex multi-step operations
            - Provides detailed explanations
            """)
        
        st.subheader("📝 Example Requests")
        st.markdown("""
        **Analysis Requests:**
        - "Show me all revenue data from Q1 and calculate the total"
        - "Find customers with outstanding balances over $5000"
        - "What are the top 5 products by sales volume?"
        
        **Editing Requests:**
        - "Update all prices in the product sheet by increasing them 5%"
        - "Change the status of all pending orders to 'confirmed'"
        - "Add a new column for profit margin and calculate it for each product"
        
        **Complex Operations:**
        - "Find all employees in the sales department and update their commission rate to 8%"
        - "Identify duplicate customer entries and merge them"
        - "Calculate year-over-year growth for each product category"
        """)

if __name__ == "__main__":
    main()
