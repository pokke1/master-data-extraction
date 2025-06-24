# 🎯 AI-Powered Universal Column Mapper

An intelligent Streamlit application that uses Azure OpenAI to automatically map data columns to custom fields. Perfect for processing various data files with different layouts, formats, and naming conventions.

## 🎯 Features

- **Multiple File Formats**: Supports Excel (.xlsx) and CSV files with automatic format detection
- **Smart Header Detection**: Automatically detects header rows regardless of their position
- **Custom Field Configuration**: Define your own fields or use predefined presets
- **AI-Powered Mapping**: Uses Azure OpenAI to intelligently map columns to required fields
- **Flexible Input**: Handles various layouts, naming conventions, and column orders
- **Interactive Validation**: Allows users to review and correct AI suggestions
- **Data Preview**: Shows sample data and mapping results for transparency
- **Export Functionality**: Download mappings, field configurations, and final datasets
- **Data Quality Insights**: Completeness metrics and quality warnings
- **Preset Management**: Built-in presets for common use cases
- **Encoding Detection**: Automatic CSV encoding and delimiter detection
- **Dockerized**: Easy deployment with Docker and Docker Compose

## 🔍 Supported File Formats

- **Excel Files (.xlsx)**: Full support with automatic sheet detection
- **CSV Files (.csv)**: Automatic encoding detection (UTF-8, Latin-1, etc.)
- **Delimiter Detection**: Automatic detection of comma, semicolon, tab, and pipe delimiters

## 📋 Built-in Field Presets

Choose from predefined field sets or create custom ones:

### Customer Data
- Customer Name, Country, City, Street, VAT

### Contact Information  
- Name, Email, Phone, Company, Address

### Product Data
- Product Name, SKU, Price, Category, Description

### Financial Data
- Amount, Currency, Date, Account, Type

### Employee Data
- Employee Name, Employee ID, Department, Position, Salary

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Azure OpenAI service with a deployed model
- Environment variables configured

### 1. Clone and Setup

```bash
git clone <repository-url>
cd excel-column-mapper
```

### 2. Configure Environment

Copy the environment template and fill in your Azure OpenAI credentials:

```bash
cp .env.template .env
```

Edit `.env` with your Azure OpenAI configuration:

```bash
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name_here
```

### 3. Run with Docker Compose

```bash
docker-compose up -d
```

The application will be available at `http://localhost:8501`

## 🛠️ Development Setup

### Local Development

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set environment variables**:
```bash
export AZURE_OPENAI_API_KEY="your_key_here"
export AZURE_OPENAI_ENDPOINT="your_endpoint_here"
export AZURE_OPENAI_DEPLOYMENT_NAME="your_deployment_name"
```

3. **Run the application**:
```bash
streamlit run app.py
```

### Docker Development

Build and run the Docker container:

```bash
docker build -t universal-column-mapper .
docker run -p 8501:8501 --env-file .env universal-column-mapper
```

## 📋 How It Works

### 1. Field Configuration
- Choose from built-in presets or create custom fields
- Define field descriptions to guide AI mapping
- Flexible field management with add/remove capabilities

### 2. File Upload
- Upload Excel (.xlsx) or CSV files
- Automatic file format detection and processing
- Smart encoding and delimiter detection for CSVs

### 3. Header Detection
- Smart algorithm analyzes the first 10 rows to find the most likely header row
- Considers factors like non-null values, string patterns, and data types
- Works across different file formats

### 4. AI Analysis
- Extracts column names and sample data
- Sends structured prompt with custom field definitions to Azure OpenAI
- AI analyzes patterns and suggests mappings based on field descriptions

### 5. Validation & Correction
- Review AI suggestions in an intuitive interface
- Manually correct any incorrect mappings
- Real-time preview of mapped data

### 6. Export Results
- Download mapping configuration as JSON
- Export complete field configuration for reuse
- **Download final mapped dataset** in Excel or CSV format
- Data quality insights and completeness metrics Download the mapping configuration as JSON
- Use the mapping for data processing pipelines

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Excel/CSV Files │───▶│   Streamlit App  │───▶│  Azure OpenAI   │
│ (.xlsx/.csv)    │    │                  │    │     (GPT)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Custom Fields +  │
                       │ Column Mapping   │
                       │     (JSON)       │
                       └──────────────────┘
```

### Component Architecture

```
📁 File Processing Layer
├── 📄 FileProcessor (Excel/CSV handling)
├── 🔍 EncodingDetector (CSV encoding detection)
└── 📊 DelimiterDetector (CSV delimiter detection)

🧠 AI Mapping Layer  
├── 🤖 ColumnMapper (Azure OpenAI integration)
├── 📝 PromptBuilder (Dynamic prompt generation)
└── 🎯 FieldManager (Custom field management)

🖥️ User Interface Layer
├── 📋 FieldConfiguration (Custom/preset fields)
├── 📤 FileUpload (Multi-format support)
├── ✅ ValidationInterface (Manual corrections)
└── 📥 ExportManager (JSON downloads)
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_OPENAI_API_KEY` | Your Azure OpenAI API key | Yes |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI service endpoint | Yes |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Name of your deployed model | Yes |
| `AZURE_OPENAI_API_VERSION` | API version (default: 2024-02-15-preview) | No |

### Supported File Formats

#### Excel Files (.xlsx)
- Multi-sheet support (uses first sheet)
- Various layouts and structures
- Headers at different row positions
- Multiple languages and naming conventions

#### CSV Files (.csv)
- **Encoding Detection**: UTF-8, Latin-1, CP-1252, and more
- **Delimiter Detection**: Comma (,), Semicolon (;), Tab (\t), Pipe (|)
- **Robust Parsing**: Handles quotes, escaped characters, and edge cases
- **International Support**: Works with various regional CSV formats

### Field Management

#### Built-in Presets
- **Customer Data**: Standard customer information fields
- **Contact Information**: Personal and business contact details  
- **Product Data**: E-commerce and inventory fields
- **Financial Data**: Transaction and accounting fields
- **Employee Data**: HR and personnel information

#### Custom Fields
- **Dynamic Creation**: Add/remove fields on the fly
- **Rich Descriptions**: Detailed field descriptions guide AI mapping
- **Flexible Structure**: No limits on field types or count
- **Export/Import**: Save field configurations for reuse

## 🐳 Docker Configuration

### Dockerfile Features
- Multi-stage build for optimization
- Non-root user for security
- Health checks for monitoring
- Proper caching for faster builds

### Docker Compose Features
- Environment variable management
- Volume mounting for uploads
- Automatic restart policy
- Health check configuration

## 🔧 Advanced Configuration

### Custom Field Strategies

#### Field Description Best Practices
- **Be Specific**: "Customer company names" vs "Names"
- **Include Examples**: "Phone numbers (mobile, landline, international)"
- **Mention Formats**: "Dates in YYYY-MM-DD or DD/MM/YYYY format"
- **Language Hints**: "Address fields (may include Street, Avenue, etc.)"

#### Multi-language Support
The AI can handle various languages and naming conventions:
- **English**: Customer, Company, Address, Phone
- **Spanish**: Cliente, Empresa, Dirección, Teléfono  
- **French**: Client, Société, Adresse, Téléphone
- **German**: Kunde, Firma, Adresse, Telefon
- **Italian**: Cliente, Azienda, Indirizzo, Telefono

### File Processing Configuration

#### CSV Advanced Options
```python
# The app automatically detects these, but you can customize:
SUPPORTED_ENCODINGS = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
SUPPORTED_DELIMITERS = [',', ';', '\t', '|']
MAX_DETECTION_ROWS = 10
```

#### Excel Advanced Options  
```python
# Configurable parameters:
MAX_HEADER_DETECTION_ROWS = 10
MIN_HEADER_CONFIDENCE = 0.5
SAMPLE_DATA_ROWS = 3
```

### Docker Production Configuration

#### Environment Variables
```bash
# Performance tuning
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
STREAMLIT_SERVER_MAX_MESSAGE_SIZE=200

# Security
STREAMLIT_SERVER_ENABLE_CORS=false
STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
```

#### Resource Limits
```yaml
# In docker-compose.yml
deploy:
  resources:
    limits:
      memory: 1G
      cpus: '0.5'
    reservations:
      memory: 512M
      cpus: '0.25'
```

## 🚨 Troubleshooting

### Common Issues

1. **Missing Environment Variables**
   - Ensure all required Azure OpenAI variables are set
   - Check `.env` file formatting

2. **File Processing Errors**
   - **Excel**: Verify file is valid .xlsx format and contains data
   - **CSV**: Check for encoding issues or unusual delimiters
   - **Headers**: Ensure data has recognizable column headers

3. **CSV-Specific Issues**
   - **Encoding Problems**: App auto-detects, but manual encoding may be needed
   - **Delimiter Detection**: App tries common delimiters automatically
   - **Special Characters**: UTF-8 encoding handles most international characters

4. **AI Mapping Failures**
   - Verify Azure OpenAI deployment is active
   - Check API key permissions and quotas
   - Ensure field descriptions are clear and descriptive

5. **Container Health Check Failures**
   - Check if port 8501 is available
   - Verify container has sufficient resources
   - Review Docker logs for specific errors

### File Format Troubleshooting

#### Excel Files
```bash
# Check if file is corrupted
file your_file.xlsx
# Should show: Microsoft Excel 2007+
```

#### CSV Files  
```bash
# Check encoding
file -bi your_file.csv
# Example output: text/plain; charset=utf-8

# Check for unusual characters
hexdump -C your_file.csv | head
```

### Logs

View application logs:
```bash
docker-compose logs -f streamlit-app
```

## 📊 Sample Input/Output

### Input File Examples

#### Excel Structure
```
Row 1: [Company Data Export - Q1 2024]
Row 2: [Empty]
Row 3: [Customer, Country, Address, Tax ID, City, Contact]
Row 4: [ABC Corp, USA, 123 Main St, 12345, New York, john@abc.com]
```

#### CSV Structure  
```csv
"Client Name";"País";"Dirección";"NIF";"Ciudad"
"Empresa ABC";"España";"Calle Mayor 123";"B12345678";"Madrid"
"Firma XYZ";"Francia";"Rue de la Paix 45";"FR987654321";"París"
```

### Custom Field Configuration
```json
{
  "Company Name": "Name of the business or organization",
  "Country": "Country where the company is located",  
  "Full Address": "Complete physical address",
  "Tax Identifier": "Tax ID, VAT number, or fiscal identifier",
  "City": "City or municipality location"
}
```

### Output Mapping
```json
{
  "Company Name": "Customer",
  "Country": "País", 
  "City": "Ciudad",
  "Full Address": "Dirección",
  "Tax Identifier": "NIF"
}
```

### Field Configuration Export
```json
{
  "fields": {
    "Company Name": "Name of the business or organization",
    "Country": "Country where the company is located",
    "Full Address": "Complete physical address",
    "Tax Identifier": "Tax ID, VAT number, or fiscal identifier",
    "City": "City or municipality location"
  },
  "mapping": {
    "Company Name": "Customer",
    "Country": "País",
    "City": "Ciudad", 
    "Full Address": "Dirección",
    "Tax Identifier": "NIF"
  }
}
```

### Final Mapped Dataset Export

#### Excel Output (.xlsx)
```
| Company Name | Country | City   | Full Address      | Tax Identifier |
|--------------|---------|--------|-------------------|----------------|
| ABC Corp     | España  | Madrid | Calle Mayor 123   | B12345678      |
| XYZ Firma    | Francia | París  | Rue de la Paix 45 | FR987654321    |
```

#### CSV Output (.csv)
```csv
Company Name,Country,City,Full Address,Tax Identifier
ABC Corp,España,Madrid,Calle Mayor 123,B12345678
XYZ Firma,Francia,París,Rue de la Paix 45,FR987654321
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Docker logs
3. Open an issue on GitHub
4. Contact the development team

---

Built with ❤️ using Streamlit, Azure OpenAI, and modern data processing techniques

## 🔄 Recent Updates

### Version 2.1 Features
- ✅ **Dataset Export**: Download final mapped data in Excel/CSV format
- ✅ **Data Quality Insights**: Completeness metrics and quality warnings
- ✅ **Enhanced Preview**: Rich data preview with summary statistics
- ✅ **Auto-formatting**: Excel exports with styled headers and auto-sized columns

### Version 2.0 Features
- ✅ **Multi-format Support**: Added CSV file processing
- ✅ **Custom Fields**: Dynamic field configuration
- ✅ **Field Presets**: Built-in templates for common use cases
- ✅ **Encoding Detection**: Automatic CSV encoding detection
- ✅ **Delimiter Detection**: Smart CSV delimiter recognition
- ✅ **Enhanced UI**: Improved user experience and validation
- ✅ **Export Options**: Field configuration and mapping export

### Roadmap
- 🔄 **Additional Formats**: TSV, JSON, XML support
- 🔄 **Batch Processing**: Multiple file processing
- 🔄 **API Integration**: RESTful API for automated workflows  
- 🔄 **Advanced Validation**: Data quality checks and suggestions
- 🔄 **Template Library**: Community-shared field templates
- 🔄 **Data Transformation**: Built-in data cleaning and formatting