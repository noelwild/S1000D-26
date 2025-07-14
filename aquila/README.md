# Aquila S1000D-AI

A laboratory-scale pipeline that ingests legacy maintenance PDFs and automatically transforms them into fully linked S1000D Issue 5-0 data sets.

## Overview

Aquila S1000D-AI is a specialized application designed to modernize technical documentation by converting unstructured maintenance PDFs into structured, linked S1000D data modules. The application uses AI-powered text analysis to automatically classify content, rewrite it in Simplified Technical English (STE), and generate proper S1000D data module codes.

## Features

### Core Functionality

- **PDF Text Extraction**: Automatically extracts and processes text from uploaded PDF documents
- **AI-Powered Classification**: Uses OpenAI GPT-4o-mini to classify text chunks and determine data module types
- **STE Rewriting**: Converts legacy technical language to ASD-STE100 Simplified Technical English
- **DMC Generation**: Creates deterministic Data Module Codes following S1000D standards
- **Image Processing**: Extracts images from PDFs and generates captions with object detection
- **Real-time Progress**: WebSocket-based progress updates during document processing

### Technical Specifications

- **Framework**: FastAPI backend with vanilla JavaScript frontend
- **Database**: SQLite with SQLModel ORM
- **AI Integration**: OpenAI GPT-4o-mini for text classification and vision analysis
- **File Structure**: Exactly 5 runtime files as specified
- **Standards Compliance**: S1000D Issue 5-0 compliant output

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Required Python packages (see installation below)

### Installation

1. **Navigate to the application directory**:
   ```bash
   cd /app/aquila
   ```

2. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn sqlmodel pypdf pdfminer.six openai python-multipart websockets
   ```

3. **Configure API key**:
   - Ensure your OpenAI API key is in `keys.txt`:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Running the Application

1. **Start the server**:
   ```bash
   python server.py
   ```

2. **Access the application**:
   - Open your browser and navigate to: `http://127.0.0.1:8001/index.html`

## Application Structure

### Runtime Files

The application consists of exactly 5 runtime files:

| File | Purpose |
|------|---------|
| `server.py` | Complete FastAPI backend with SQLite database, OpenAI integration, and WebSocket support |
| `index.html` | Single-page user interface with upload, progress tracking, and module viewing |
| `app.css` | Pre-built Tailwind-flavored dark theme styling |
| `app.js` | Vanilla JavaScript with WebSocket handling and DOM manipulation |
| `keys.txt` | Plain-text API key storage (OpenAI API key required) |

### API Endpoints

- `GET /api/health` - Health check endpoint
- `POST /api/documents/upload` - Upload and process PDF documents
- `GET /api/documents` - Retrieve all processed documents
- `GET /api/data-modules` - Retrieve data modules (with optional document filtering)
- `GET /api/icns` - Retrieve Illustration Control Numbers
- `WebSocket /ws` - Real-time progress updates

### Database Schema

#### Documents Table
- `id`: Unique document identifier
- `filename`: Original filename
- `file_path`: Server file path
- `sha256`: File hash for deduplication
- `operational_context`: S1000D operational context (Water, Air, Land, Other)
- `status`: Processing status (processing, completed, failed)

#### Data Modules Table
- `id`: Unique module identifier
- `document_id`: Reference to parent document
- `dmc`: Generated Data Module Code
- `title`: Module title
- `info_code`: S1000D information code
- `item_location`: S1000D item location code
- `verbatim_content`: Original extracted text
- `ste_content`: Simplified Technical English version
- `type`: Module type (procedure, description, fault_isolation, etc.)

#### ICNs Table
- `id`: Unique ICN identifier
- `document_id`: Reference to parent document
- `data_module_id`: Reference to associated data module
- `icn`: Illustration Control Number (ICN-<hash>)
- `image_path`: Path to extracted image
- `caption`: AI-generated caption
- `objects`: JSON array of detected objects

## Usage Guide

### Document Upload Process

1. **Upload PDF**: Click "Upload PDF" and select your maintenance document
2. **Choose Context**: Select operational context (Water, Air, Land, Other)
3. **Monitor Progress**: Watch real-time progress updates via WebSocket
4. **View Results**: Browse generated data modules in the sidebar

### Processing Pipeline

1. **Text Extraction**: PDF text is extracted using pdfminer.six
2. **Content Chunking**: Text is divided into manageable chunks using heuristics
3. **AI Classification**: Each chunk is analyzed by OpenAI for:
   - Content type classification
   - Title generation
   - S1000D info code assignment
   - Item location determination
   - STE rewriting
4. **Module Generation**: Data modules are created with deterministic DMCs
5. **Image Processing**: Images are extracted and analyzed for captions/objects
6. **Database Storage**: All processed data is stored in SQLite

### DMC Generation

Data Module Codes are generated using the format:
```
{operational_context}-{type}-{info_code}-{item_location}-{sequence:02d}
```

Example: `Air-procedure-520-B-03`

### Viewing Content

- **Module Navigation**: Click on any data module in the sidebar to view its content
- **STE/Verbatim Toggle**: Switch between Simplified Technical English and original text
- **Module Information**: View DMC, type, and info code details

## Configuration

### OpenAI Settings

The application uses OpenAI GPT-4o-mini with the following configuration:
- **Model**: `gpt-4o-mini`
- **Temperature**: `0` (deterministic outputs)
- **Max Tokens**: Default (varies by endpoint)

### File Storage

- **Upload Directory**: `/tmp/aquila_uploads/`
- **Database**: `aquila.db` (SQLite)
- **Image Storage**: Temporary directory with SHA-256 naming

## Testing

### Backend Testing

All backend functionality has been comprehensively tested:

✅ Health check endpoint
✅ PDF document upload and processing
✅ Document retrieval API
✅ Data modules API with filtering
✅ ICNs API with filtering
✅ WebSocket real-time updates
✅ Database operations
✅ OpenAI integration
✅ Error handling
✅ S1000D compliance

### Frontend Testing

Complete frontend functionality verified:

✅ Main interface elements
✅ Upload modal functionality
✅ Document processing flow
✅ Module navigation
✅ STE/Verbatim toggle
✅ API integration
✅ WebSocket connectivity
✅ Responsive design
✅ Progress bar system

## Troubleshooting

### Common Issues

1. **OpenAI API Key Missing**
   - Error: "OPENAI_API_KEY not found in keys.txt"
   - Solution: Add your OpenAI API key to `keys.txt`

2. **Port Already in Use**
   - Error: "Address already in use"
   - Solution: Check if another instance is running or use a different port

3. **PDF Processing Fails**
   - Check if the PDF contains extractable text
   - Verify OpenAI API key is valid and has sufficient credits

4. **WebSocket Connection Issues**
   - Ensure server is running on correct port
   - Check browser console for connection errors

### Logs and Debugging

- Server logs are output to console when running `python server.py`
- Frontend errors appear in browser developer console
- Database issues can be debugged by examining `aquila.db`

## Development

### Architecture

The application follows a clean separation of concerns:

- **Backend**: FastAPI handles API endpoints, database operations, and AI integration
- **Frontend**: Vanilla JavaScript manages UI interactions and WebSocket communication
- **Database**: SQLite provides lightweight, file-based storage
- **AI**: OpenAI GPT-4o-mini handles text classification and image analysis

### Extension Points

The application can be extended by:

1. **Adding New Data Module Types**: Modify the AI classification prompts
2. **Supporting Additional File Formats**: Extend the upload handler
3. **Implementing Advanced S1000D Features**: Add more sophisticated DMC generation
4. **Integrating Additional AI Models**: Replace or supplement OpenAI integration

## Standards Compliance

### S1000D Issue 5-0

The application generates S1000D-compliant data modules:

- **DMC Structure**: Follows S1000D Data Module Code format
- **Info Codes**: Uses appropriate S1000D information codes
- **Item Locations**: Implements S1000D item location coding
- **Cross-References**: Maintains linkage between related modules

### ASD-STE100 Simplified Technical English

- **Automated Rewriting**: Converts legacy text to STE standards
- **Dual View**: Maintains both original and STE versions
- **Consistency**: Ensures consistent technical language across modules

## Performance

### Optimizations

- **Chunked Processing**: Text is processed in manageable chunks to stay within AI token limits
- **Asynchronous Operations**: Document processing runs asynchronously with progress updates
- **Efficient Storage**: SHA-256 hashing prevents duplicate file storage
- **Lightweight Database**: SQLite provides fast, local data access

### Scalability Considerations

For production use, consider:

- **Database Migration**: Move from SQLite to PostgreSQL for concurrent users
- **File Storage**: Implement cloud storage for uploaded documents
- **API Rate Limiting**: Add rate limiting for OpenAI API calls
- **Caching**: Implement caching for frequently accessed data

## License

This is a laboratory-scale prototype implementation. Use in accordance with your organization's policies and applicable regulations.

## Support

For issues or questions, check the troubleshooting section above or examine the application logs for detailed error information.