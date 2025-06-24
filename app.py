import streamlit as st
import pandas as pd
import json
import os
from openai import AzureOpenAI
from typing import Dict, List, Tuple, Optional
import logging
import io
import chardet
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileProcessor:
    """Handles different file formats and processing logic."""
    
    @staticmethod
    def detect_encoding(file_content: bytes) -> str:
        """Detect file encoding for CSV files."""
        try:
            result = chardet.detect(file_content)
            return result['encoding'] or 'utf-8'
        except:
            return 'utf-8'
    
    @staticmethod
    def detect_csv_delimiter(file_content: str, sample_size: int = 1024) -> str:
        """Detect CSV delimiter by analyzing the first few lines."""
        sample = file_content[:sample_size]
        lines = sample.split('\n')[:5]  # Check first 5 lines
        
        delimiters = [',', ';', '\t', '|']
        delimiter_scores = {}
        
        for delimiter in delimiters:
            scores = []
            for line in lines:
                if line.strip():
                    scores.append(line.count(delimiter))
            
            if scores:
                # Check consistency across lines
                avg_score = sum(scores) / len(scores)
                consistency = 1 - (max(scores) - min(scores)) / (max(scores) + 1)
                delimiter_scores[delimiter] = avg_score * consistency
        
        if delimiter_scores:
            return max(delimiter_scores, key=delimiter_scores.get)
        return ','
    
    @classmethod
    def process_csv_file(cls, uploaded_file) -> Tuple[pd.DataFrame, int]:
        """Process CSV file with automatic encoding and delimiter detection."""
        # Read file content
        file_content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer
        
        # Detect encoding
        encoding = cls.detect_encoding(file_content)
        
        # Decode content
        try:
            content_str = file_content.decode(encoding)
        except UnicodeDecodeError:
            content_str = file_content.decode('utf-8', errors='ignore')
        
        # Detect delimiter
        delimiter = cls.detect_csv_delimiter(content_str)
        
        # Create StringIO object for pandas
        string_io = io.StringIO(content_str)
        
        # Read CSV without header first to detect header row
        df_no_header = pd.read_csv(string_io, header=None, sep=delimiter)
        
        # Detect header row
        header_row = cls.detect_header_row(df_no_header)
        
        # Re-read with detected header
        string_io.seek(0)
        df_with_header = pd.read_csv(string_io, header=header_row, sep=delimiter)
        
        # Clean column names
        df_with_header.columns = [str(col).strip() for col in df_with_header.columns]
        
        return df_with_header, header_row
    
    @classmethod
    def process_excel_file(cls, uploaded_file) -> Tuple[pd.DataFrame, int]:
        """Process Excel file and detect header row."""
        # Read Excel file without assuming header position
        df = pd.read_excel(uploaded_file, header=None, dtype=str)
        
        # Detect header row
        header_row = cls.detect_header_row(df)
        
        # Re-read with detected header
        df_with_header = pd.read_excel(uploaded_file, header=header_row,dtype=str)
        
        # Clean column names
        df_with_header.columns = [str(col).strip() for col in df_with_header.columns]
        
        return df_with_header, header_row
    
    @staticmethod
    def detect_header_row(df: pd.DataFrame) -> int:
        """
        Detect the header row by finding the row with the most non-null string values
        that look like column headers.
        """
        max_score = 0
        header_row = 0
        
        # Check first 10 rows for potential headers
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            score = 0
            
            # Count non-null values
            non_null_count = row.notna().sum()
            
            # Check if values look like headers (strings, not too long, no special patterns)
            for val in row:
                if pd.notna(val) and isinstance(val, str):
                    # Prefer shorter strings that could be headers
                    if len(val) < 50 and not val.isdigit():
                        score += 1
            
            # Normalize score by total columns
            normalized_score = score / len(row) if len(row) > 0 else 0
            
            if normalized_score > max_score and non_null_count >= len(row) * 0.5:
                max_score = normalized_score
                header_row = i
        
        return header_row

class ColumnMapper:
    def __init__(self):
        """Initialize the Column Mapper with Azure OpenAI configuration."""
        self.client = self._initialize_azure_client()
    
    def _initialize_azure_client(self) -> Optional[AzureOpenAI]:
        """Initialize Azure OpenAI client with environment variables."""
        try:
            return AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            )
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            return None
    
    def prepare_sample_data(self, df: pd.DataFrame, max_samples: int = 3) -> Dict:
        """Prepare sample data for LLM analysis."""
        sample_data = {}
        
        for col in df.columns:
            # Get non-null sample values
            non_null_values = df[col].dropna().head(max_samples).tolist()
            sample_data[col] = non_null_values
        
        return sample_data
    
    def create_mapping_prompt(self, sample_data: Dict, required_fields: List[str], field_descriptions: Dict[str, str]) -> Tuple[str, str]:
        """Create system and user prompts for column mapping."""
        
        # Build field descriptions for the prompt
        field_descriptions_text = "\n".join([
            f"- {field}: {description}" for field, description in field_descriptions.items()
        ])
        
        # Build JSON structure example
        json_structure = {field: "mapped_column_name_or_null" for field in required_fields}
        
        system_prompt = f"""You are an expert data analyst specializing in mapping data columns to standardized fields. 
Your task is to analyze column names and sample data to map them to required fields.

Required fields to map:
{field_descriptions_text}

Return a JSON object with the mapping. If a required field cannot be mapped to any column, use null as the value.
The JSON should have this structure:
{json.dumps(json_structure, indent=2)}

Be flexible with naming variations and consider different languages, abbreviations, and conventions.
Analyze both column names and sample data values to make the best mapping decisions.
Consider common variations and synonyms for each field type."""

        user_prompt = f"""Please analyze the following data columns and their sample data, then map them to the required fields.

Required Fields to Map:
{json.dumps(required_fields, indent=2)}

Column Names and Sample Data:
{json.dumps(sample_data, indent=2, ensure_ascii=False)}

Map each required field to the most appropriate column name. Consider both the column name and the sample data values.
Look for patterns, data types, and content that would indicate which field each column represents."""

        return system_prompt, user_prompt
    
    def get_column_mapping(self, sample_data: Dict, required_fields: List[str], field_descriptions: Dict[str, str]) -> Optional[Dict]:
        """Get column mapping from Azure OpenAI."""
        if not self.client:
            st.error("Azure OpenAI client not initialized. Please check your environment variables.")
            return None
        
        try:
            system_prompt, user_prompt = self.create_mapping_prompt(sample_data, required_fields, field_descriptions)
            
            response = self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=2000,
                temperature=0.5
            )
            
            mapping_result = json.loads(response.choices[0].message.content)
            return mapping_result
            
        except Exception as e:
            logger.error(f"Error getting column mapping: {e}")
            st.error(f"Error calling Azure OpenAI: {e}")
            return None

class FieldManager:
    """Manages field definitions and presets."""
    
    @staticmethod
    def get_preset_fields() -> Dict[str, Dict[str, str]]:
        """Get predefined field presets."""
        return {
            "Customer Data": {
                "Customer Name": "Names of customers, clients, or companies",
                "Country": "Country names, codes, or abbreviations",
                "City": "City or town names",
                "Street": "Street addresses or full addresses",
                "VAT": "VAT numbers, tax IDs, or fiscal identifiers"
            },
            "Contact Information": {
                "Name": "Person or contact names",
                "Email": "Email addresses",
                "Phone": "Phone numbers or contact numbers",
                "Company": "Company or organization names",
                "Address": "Physical addresses"
            },
            "Product Data": {
                "Product Name": "Product titles or names",
                "SKU": "Stock keeping units or product codes",
                "Price": "Product prices or costs",
                "Category": "Product categories or classifications",
                "Description": "Product descriptions or details"
            },
            "Financial Data": {
                "Amount": "Monetary amounts or values",
                "Currency": "Currency codes or symbols",
                "Date": "Transaction or record dates",
                "Account": "Account numbers or identifiers",
                "Type": "Transaction or record types"
            },
            "Employee Data": {
                "Employee Name": "Employee full names",
                "Employee ID": "Employee identification numbers",
                "Department": "Department or division names",
                "Position": "Job titles or positions",
                "Salary": "Salary or compensation amounts"
            }
        }

def create_mapped_dataframe(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Create a new dataframe with only the mapped columns, renamed to required field names.
    
    Args:
        df: Original dataframe
        mapping: Dictionary mapping required fields to column names
        
    Returns:
        New dataframe with mapped columns
    """
    mapped_data = {}
    
    for required_field, column_name in mapping.items():
        if column_name and column_name in df.columns:
            mapped_data[required_field] = df[column_name]
        else:
            # Create empty column if mapping is missing
            mapped_data[required_field] = pd.Series([None] * len(df))
    
    return pd.DataFrame(mapped_data)

def create_excel_download(df: pd.DataFrame) -> bytes:
    """
    Create an Excel file in memory for download.
    
    Args:
        df: Dataframe to export
        
    Returns:
        Excel file as bytes
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Mapped Data', index=False)
        
        # Get the workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Mapped Data']
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            
            adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
            worksheet.column_dimensions[column_letter].width = adjusted_width
        
        # Style the header row
        from openpyxl.styles import Font, PatternFill
        
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        
        for cell in worksheet[1]:
            cell.font = header_font
            cell.fill = header_fill
    
    output.seek(0)
    return output.read()

def process_uploaded_file(uploaded_file) -> Tuple[pd.DataFrame, int, str]:
    """Process uploaded file based on its type."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'xlsx':
            df, header_row = FileProcessor.process_excel_file(uploaded_file)
            file_type = "Excel"
        elif file_extension == 'csv':
            df, header_row = FileProcessor.process_csv_file(uploaded_file)
            file_type = "CSV"
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        return df, header_row, file_type
    
    except Exception as e:
        logger.error(f"Error processing {file_extension} file: {e}")
        raise e

def main():
    st.set_page_config(
        page_title="Universal Column Mapper", 
        page_icon="🎯", 
        layout="wide"
    )
    
    st.title("🎯 AI-Powered Universal Column Mapper")
    st.markdown("Upload data files and let AI automatically map your columns to custom fields.")
    
    # Initialize session state
    if 'mapping_result' not in st.session_state:
        st.session_state.mapping_result = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'sample_data' not in st.session_state:
        st.session_state.sample_data = None
    if 'required_fields' not in st.session_state:
        st.session_state.required_fields = []
    if 'field_descriptions' not in st.session_state:
        st.session_state.field_descriptions = {}
    
    # Check environment variables
    required_env_vars = [
        "AZURE_OPENAI_API_KEY", 
        "AZURE_OPENAI_ENDPOINT", 
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"Missing environment variables: {', '.join(missing_vars)}")
        st.stop()
    
    mapper = ColumnMapper()
    field_manager = FieldManager()
    
    # Field Configuration Section
    st.header("1. Configure Fields to Map")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Field Presets")
        preset_options = ["Custom"] + list(field_manager.get_preset_fields().keys())
        selected_preset = st.selectbox(
            "Choose a preset or create custom fields:",
            preset_options,
            help="Select a predefined set of fields or choose 'Custom' to define your own"
        )
        
        if selected_preset != "Custom":
            preset_fields = field_manager.get_preset_fields()[selected_preset]
            st.session_state.required_fields = list(preset_fields.keys())
            st.session_state.field_descriptions = preset_fields
            
            st.success(f"✅ Loaded {selected_preset} preset with {len(preset_fields)} fields")
            
            # Display preset fields
            with st.expander("📝 Preset Fields", expanded=True):
                for field, description in preset_fields.items():
                    st.write(f"**{field}**: {description}")
    
    with col2:
        st.subheader("🛠️ Custom Fields")
        
        if selected_preset == "Custom":
            # Custom field creation
            st.write("Create your own fields to map:")
            
            # Initialize custom fields if not exists
            if 'custom_fields' not in st.session_state:
                st.session_state.custom_fields = {}
            
            # Add new field
            with st.expander("➕ Add New Field", expanded=len(st.session_state.custom_fields) == 0):
                new_field_name = st.text_input("Field Name:", placeholder="e.g., Customer Name")
                new_field_desc = st.text_area("Field Description:", placeholder="Describe what this field represents...")
                
                if st.button("Add Field", type="primary"):
                    if new_field_name and new_field_desc:
                        st.session_state.custom_fields[new_field_name] = new_field_desc
                        st.session_state.required_fields = list(st.session_state.custom_fields.keys())
                        st.session_state.field_descriptions = st.session_state.custom_fields.copy()
                        st.rerun()
                    else:
                        st.error("Please provide both field name and description")
            
            # Display and manage custom fields
            if st.session_state.custom_fields:
                st.write("**Custom Fields:**")
                for i, (field, description) in enumerate(st.session_state.custom_fields.items()):
                    col_field, col_remove = st.columns([4, 1])
                    with col_field:
                        st.write(f"**{field}**: {description}")
                    with col_remove:
                        if st.button("🗑️", key=f"remove_{i}", help=f"Remove {field}"):
                            del st.session_state.custom_fields[field]
                            st.session_state.required_fields = list(st.session_state.custom_fields.keys())
                            st.session_state.field_descriptions = st.session_state.custom_fields.copy()
                            st.rerun()
        else:
            st.info("Using preset fields. Select 'Custom' to create your own fields.")
    
    # Show current fields
    if st.session_state.required_fields:
        st.subheader("🎯 Current Fields to Map")
        field_cols = st.columns(len(st.session_state.required_fields))
        for i, field in enumerate(st.session_state.required_fields):
            with field_cols[i % len(field_cols)]:
                st.metric(
                    label=field,
                    value="📍",
                    help=st.session_state.field_descriptions.get(field, "No description")
                )
    else:
        st.warning("⚠️ Please configure fields to map before uploading a file.")
        st.stop()
    
    # File upload section
    st.header("2. Upload Data File")
    uploaded_file = st.file_uploader(
        "Choose a data file", 
        type=['xlsx', 'csv'],
        help="Upload an Excel (.xlsx) or CSV file with your data. The app will automatically detect headers and structure."
    )
    
    if uploaded_file is not None:
        try:
            # Process the uploaded file
            with st.spinner("Processing file..."):
                df, header_row, file_type = process_uploaded_file(uploaded_file)
                st.session_state.df = df
            
            st.success(f"✅ {file_type} file processed successfully! Header detected at row {header_row + 1}")
            
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Type", file_type)
            with col2:
                st.metric("Rows", f"{df.shape[0]:,}")
            with col3:
                st.metric("Columns", df.shape[1])
            
            # Display file preview
            st.header("3. File Preview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Column Names")
                st.write(list(df.columns))
            
            with col2:
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
            
            # AI Analysis section
            st.header("4. AI Column Mapping")
            
            if st.button("🤖 Analyze Columns with AI", type="primary"):
                with st.spinner("Analyzing columns with AI..."):
                    sample_data = mapper.prepare_sample_data(df)
                    st.session_state.sample_data = sample_data
                    
                    mapping_result = mapper.get_column_mapping(
                        sample_data, 
                        st.session_state.required_fields, 
                        st.session_state.field_descriptions
                    )
                    st.session_state.mapping_result = mapping_result
            
            # Display sample data for transparency
            if st.session_state.sample_data:
                with st.expander("📋 Sample Data Sent to AI"):
                    st.json(st.session_state.sample_data)
            
            # Display and validate mapping results
            if st.session_state.mapping_result:
                st.header("5. Mapping Results & Validation")
                
                st.subheader("AI Suggested Mapping")
                mapping_df = pd.DataFrame([
                    {
                        "Required Field": field, 
                        "Mapped Column": mapped_col or "❌ Not Found",
                        "Description": st.session_state.field_descriptions.get(field, "")
                    }
                    for field, mapped_col in st.session_state.mapping_result.items()
                ])
                st.dataframe(mapping_df, use_container_width=True)
                
                # Allow manual correction
                st.subheader("Manual Validation & Correction")
                st.write("Review and correct the mapping if needed:")
                
                corrected_mapping = {}
                available_columns = ["None"] + list(df.columns)
                
                # Create columns for better layout
                num_fields = len(st.session_state.required_fields)
                cols_per_row = 2
                
                for i in range(0, num_fields, cols_per_row):
                    row_cols = st.columns(cols_per_row)
                    
                    for j in range(cols_per_row):
                        if i + j < num_fields:
                            field = st.session_state.required_fields[i + j]
                            current_mapping = st.session_state.mapping_result.get(field)
                            
                            # Determine default index
                            if current_mapping and current_mapping in available_columns:
                                default_index = available_columns.index(current_mapping)
                            else:
                                default_index = 0
                            
                            with row_cols[j]:
                                selected = st.selectbox(
                                    f"**{field}**",
                                    available_columns,
                                    index=default_index,
                                    key=f"mapping_{field}",
                                    help=st.session_state.field_descriptions.get(field, "")
                                )
                                corrected_mapping[field] = None if selected == "None" else selected
                
                # Final mapping confirmation
                st.header("6. Final Mapping")
                
                final_df = pd.DataFrame([
                    {
                        "Required Field": field, 
                        "Mapped Column": mapped_col or "❌ Not Mapped",
                        "Status": "✅ Mapped" if mapped_col else "❌ Not Mapped"
                    }
                    for field, mapped_col in corrected_mapping.items()
                ])
                
                st.dataframe(final_df, use_container_width=True)
                
                # Export options
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📥 Export Mapping as JSON", type="secondary"):
                        mapping_json = json.dumps(corrected_mapping, indent=2)
                        st.download_button(
                            label="Download Mapping",
                            data=mapping_json,
                            file_name="column_mapping.json",
                            mime="application/json"
                        )
                
                with col2:
                    # Export field configuration
                    field_config = {
                        "fields": st.session_state.field_descriptions,
                        "mapping": corrected_mapping
                    }
                    config_json = json.dumps(field_config, indent=2)
                    st.download_button(
                        label="📋 Download Field Config",
                        data=config_json,
                        file_name="field_configuration.json",
                        mime="application/json"
                    )
                
                with col3:
                    # Export mapped dataframe
                    st.subheader("📊 Download Mapped Data")
                    
                    # Create mapped dataframe
                    mapped_df = create_mapped_dataframe(df, corrected_mapping)
                    
                    if not mapped_df.empty:
                        # Choose export format
                        export_format = st.selectbox(
                            "Export Format",
                            ["Excel (.xlsx)", "CSV (.csv)"],
                            key="export_format"
                        )
                        
                        if export_format == "Excel (.xlsx)":
                            # Create Excel file
                            excel_buffer = create_excel_download(mapped_df)
                            st.download_button(
                                label="📊 Download Excel",
                                data=excel_buffer,
                                file_name="mapped_data.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:
                            # Create CSV file
                            csv_data = mapped_df.to_csv(index=False)
                            st.download_button(
                                label="📊 Download CSV",
                                data=csv_data,
                                file_name="mapped_data.csv",
                                mime="text/csv"
                            )
                        
                        # Show download statistics
                        total_rows = len(mapped_df)
                        mapped_columns = len([col for col in corrected_mapping.values() if col is not None])
                        st.caption(f"📈 {total_rows:,} rows × {mapped_columns} mapped columns")
                    else:
                        st.warning("No data available for download")
                
                # Show mapped data preview
                st.header("7. Mapped Data Preview")
                
                # Create and display the mapped dataframe
                mapped_df = create_mapped_dataframe(df, corrected_mapping)
                
                if not mapped_df.empty:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.subheader("📊 Final Mapped Data")
                        st.dataframe(mapped_df.head(10), use_container_width=True)
                        
                        if len(mapped_df) > 10:
                            st.caption(f"Showing first 10 rows of {len(mapped_df):,} total rows")
                    
                    with col2:
                        st.subheader("📈 Data Summary")
                        
                        # Show summary statistics for each mapped column
                        for field, column in corrected_mapping.items():
                            if column and column in df.columns:
                                non_null_count = mapped_df[field].notna().sum()
                                total_count = len(mapped_df)
                                completeness = (non_null_count / total_count * 100) if total_count > 0 else 0
                                
                                st.metric(
                                    label=field,
                                    value=f"{completeness:.1f}%",
                                    help=f"{non_null_count:,} of {total_count:,} records have data"
                                )
                
                else:
                    st.warning("No mapped data available to preview")
                
                # Statistics and insights
                st.header("8. Data Quality Insights")
                
                col1, col2, col3, col4 = st.columns(4)
                
                mapped_count = sum(1 for v in corrected_mapping.values() if v is not None)
                total_count = len(corrected_mapping)
                mapping_percentage = (mapped_count / total_count) * 100 if total_count > 0 else 0
                
                with col1:
                    st.metric(
                        "Mapping Completion", 
                        f"{mapping_percentage:.1f}%", 
                        f"{mapped_count}/{total_count} fields"
                    )
                
                with col2:
                    total_records = len(mapped_df) if not mapped_df.empty else 0
                    st.metric("Total Records", f"{total_records:,}")
                
                with col3:
                    if not mapped_df.empty:
                        # Calculate overall data completeness
                        total_cells = mapped_df.size
                        non_null_cells = mapped_df.notna().sum().sum()
                        data_completeness = (non_null_cells / total_cells * 100) if total_cells > 0 else 0
                        st.metric("Data Completeness", f"{data_completeness:.1f}%")
                    else:
                        st.metric("Data Completeness", "0%")
                
                with col4:
                    if not mapped_df.empty:
                        # Count complete rows (rows with all fields filled)
                        complete_rows = mapped_df.dropna().shape[0]
                        complete_percentage = (complete_rows / len(mapped_df) * 100) if len(mapped_df) > 0 else 0
                        st.metric("Complete Records", f"{complete_percentage:.1f}%")
                    else:
                        st.metric("Complete Records", "0%")
                
                # Data quality warnings
                if not mapped_df.empty:
                    st.subheader("🔍 Data Quality Notes")
                    
                    warnings = []
                    
                    # Check for unmapped fields
                    unmapped_fields = [field for field, column in corrected_mapping.items() if column is None]
                    if unmapped_fields:
                        warnings.append(f"⚠️ **Unmapped fields**: {', '.join(unmapped_fields)}")
                    
                    # Check for low data completeness
                    for field, column in corrected_mapping.items():
                        if column and column in df.columns:
                            completeness = (mapped_df[field].notna().sum() / len(mapped_df) * 100)
                            if completeness < 50:
                                warnings.append(f"⚠️ **Low data quality**: {field} ({completeness:.1f}% complete)")
                    
                    # Check for potential duplicates
                    if len(mapped_df) > len(mapped_df.drop_duplicates()):
                        duplicate_count = len(mapped_df) - len(mapped_df.drop_duplicates())
                        warnings.append(f"ℹ️ **Potential duplicates**: {duplicate_count} duplicate records found")
                    
                    if warnings:
                        for warning in warnings:
                            st.markdown(warning)
                    else:
                        st.success("✅ No data quality issues detected")
                
        except Exception as e:
            st.error(f"Error processing file: {e}")
            logger.error(f"File processing error: {e}")

if __name__ == "__main__":
    main()