import os
import json
import hashlib
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
import tempfile
import uuid
from datetime import datetime
import re

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Field, create_engine, Session, select
from pydantic import BaseModel
import pypdf
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import openai
import uvicorn
from contextlib import asynccontextmanager

# Load API key from keys.txt
try:
    keys_path = Path("keys.txt")
    if keys_path.exists():
        for line in keys_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                openai.api_key = line.split("=", 1)[1].strip()
                break
        else:
            raise SystemExit("OPENAI_API_KEY not found in keys.txt")
    else:
        raise SystemExit("keys.txt file not found")
except Exception as e:
    print(f"Warning: {e}")
    openai.api_key = "test-key"  # Fallback for testing

# Database Models
class Document(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    filename: str
    file_path: str
    sha256: str
    uploaded_at: datetime = Field(default_factory=datetime.now)
    operational_context: str = "Water"  # Water, Air, Land, Other
    status: str = "processing"  # processing, completed, failed

class DataModule(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    document_id: str = Field(foreign_key="document.id")
    dmc: str  # Data Module Code
    title: str
    info_code: str
    item_location: str
    sequence: int
    verbatim_content: str
    ste_content: str
    type: str  # procedure, description, etc.
    created_at: datetime = Field(default_factory=datetime.now)

class ICN(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    document_id: str = Field(foreign_key="document.id")
    data_module_id: str = Field(foreign_key="datamodule.id")
    icn: str  # ICN-<sha-prefix>
    image_path: str
    caption: str
    objects: str  # JSON string of detected objects
    created_at: datetime = Field(default_factory=datetime.now)

# Response Models
class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str
    uploaded_at: datetime
    operational_context: str

class DataModuleResponse(BaseModel):
    id: str
    dmc: str
    title: str
    verbatim_content: str
    ste_content: str
    type: str

class ICNResponse(BaseModel):
    id: str
    icn: str
    caption: str
    objects: List[str]

# Database setup
DATABASE_URL = "sqlite:///aquila.db"
engine = create_engine(DATABASE_URL)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                pass

manager = ConnectionManager()

# OpenAI helper functions
async def classify_extract(text: str) -> Dict[str, Any]:
    """Classify text chunk and extract STE version"""
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert in S1000D technical documentation and ASD-STE100 Simplified Technical English.
                    
                    Analyze the given text chunk and return a JSON response with:
                    - type: The data module type (procedure, description, fault_isolation, etc.)
                    - title: A clear title for this section
                    - info_code: S1000D info code (e.g., 040, 520, 730)
                    - item_location: Item location code (e.g., A, B, C, D)
                    - ste: The text rewritten in ASD-STE100 Simplified Technical English
                    - should_start_new_module: boolean indicating if this should start a new data module
                    
                    Keep titles concise and descriptive. Use appropriate S1000D info codes for the content type."""
                },
                {
                    "role": "user",
                    "content": f"Process this text chunk:\n\n{text}"
                }
            ]
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"OpenAI classify_extract error: {e}")
        # Fallback response
        return {
            "type": "description",
            "title": "Unknown Section",
            "info_code": "040",
            "item_location": "A",
            "ste": text,
            "should_start_new_module": True
        }

async def caption_objects(image_path: str) -> Dict[str, Any]:
    """Generate caption and detect objects in image"""
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        
        with open(image_path, "rb") as image_file:
            import base64
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": """Analyze this technical diagram/image and return a JSON response with:
                    - caption: A clear, concise caption describing what the image shows
                    - objects: A list of technical objects, components, or parts visible in the image
                    
                    Focus on technical accuracy and use appropriate maintenance terminology."""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this technical image:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"OpenAI caption_objects error: {e}")
        # Fallback response
        return {
            "caption": "Technical diagram",
            "objects": ["component", "system"]
        }

# Helper functions
def calculate_sha256(file_path: str) -> str:
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def chunk_text(text: str) -> List[str]:
    """Simple text chunking based on page breaks and headings"""
    chunks = []
    
    # Split by double newlines first
    sections = text.split('\n\n')
    
    current_chunk = ""
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        # Check if this looks like a heading (short line, all caps, etc.)
        is_heading = (
            len(section) < 100 and 
            (section.isupper() or 
             re.match(r'^\d+\.', section) or
             re.match(r'^[A-Z][A-Z\s]{5,50}$', section))
        )
        
        # If adding this section would exceed 4k tokens (~3200 chars), start new chunk
        if len(current_chunk) + len(section) > 3200:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = section
        else:
            if current_chunk and is_heading:
                # Start new chunk on headings
                chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                current_chunk += "\n\n" + section if current_chunk else section
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def generate_dmc(context: str, type_info: str, info_code: str, item_loc: str, sequence: int) -> str:
    """Generate DMC according to S1000D standards"""
    return f"{context}-{type_info}-{info_code}-{item_loc}-{sequence:02d}"

def extract_images_from_pdf(pdf_path: str) -> List[str]:
    """Extract images from PDF and save them"""
    images = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            
            for page_num, page in enumerate(reader.pages):
                if '/XObject' in page['/Resources']:
                    xObjects = page['/Resources']['/XObject'].get_object()
                    
                    for obj in xObjects:
                        if xObjects[obj]['/Subtype'] == '/Image':
                            try:
                                size = (xObjects[obj]['/Width'], xObjects[obj]['/Height'])
                                data = xObjects[obj].get_data()
                                
                                if data:
                                    # Generate unique filename
                                    image_hash = hashlib.sha256(data).hexdigest()[:8]
                                    filename = f"image_{page_num}_{image_hash}.jpg"
                                    
                                    # Create uploads directory
                                    upload_dir = Path("/tmp/aquila_uploads")
                                    upload_dir.mkdir(exist_ok=True)
                                    
                                    image_path = upload_dir / filename
                                    
                                    # Save image
                                    with open(image_path, 'wb') as img_file:
                                        img_file.write(data)
                                    
                                    images.append(str(image_path))
                            except Exception as e:
                                print(f"Error extracting image: {e}")
                                continue
    except Exception as e:
        print(f"Error in image extraction: {e}")
    
    return images

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database
    create_db_and_tables()
    
    # Create upload directory
    upload_dir = Path("/tmp/aquila_uploads")
    upload_dir.mkdir(exist_ok=True)
    
    print("Aquila S1000D-AI initialized successfully")
    yield
    
    print("Shutting down Aquila S1000D-AI")

# FastAPI app
app = FastAPI(title="Aquila S1000D-AI", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "Aquila S1000D-AI"}

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...), operational_context: str = "Water"):
    """Upload and process a PDF document"""
    try:
        # Create upload directory
        upload_dir = Path("/tmp/aquila_uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Calculate SHA-256 and save file
        temp_path = upload_dir / f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        file_hash = calculate_sha256(str(temp_path))
        final_path = upload_dir / f"{file_hash}.pdf"
        temp_path.rename(final_path)
        
        # Create document record
        with Session(engine) as session:
            document = Document(
                filename=file.filename,
                file_path=str(final_path),
                sha256=file_hash,
                operational_context=operational_context,
                status="processing"
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            doc_id = document.id
        
        # Broadcast upload complete
        await manager.broadcast({
            "type": "progress",
            "phase": "upload_complete",
            "doc_id": doc_id,
            "detail": f"File {file.filename} uploaded successfully"
        })
        
        # Process document asynchronously
        asyncio.create_task(process_document(doc_id, str(final_path), operational_context))
        
        return {"document_id": doc_id, "status": "processing"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_document(doc_id: str, file_path: str, operational_context: str):
    """Process the uploaded PDF document"""
    try:
        # Extract text
        await manager.broadcast({
            "type": "progress",
            "phase": "text_extracted",
            "doc_id": doc_id,
            "detail": "Extracting text from PDF..."
        })
        
        text = extract_text(file_path, laparams=LAParams())
        chunks = chunk_text(text)
        
        # Process chunks and create data modules
        await manager.broadcast({
            "type": "progress",
            "phase": "modules_created",
            "doc_id": doc_id,
            "detail": f"Processing {len(chunks)} text chunks..."
        })
        
        with Session(engine) as session:
            sequence = 1
            current_module = None
            
            for chunk in chunks:
                # Classify and extract
                result = await classify_extract(chunk)
                
                if result.get("should_start_new_module", True) or current_module is None:
                    # Create new data module
                    dmc = generate_dmc(
                        operational_context,
                        result["type"],
                        result["info_code"],
                        result["item_location"],
                        sequence
                    )
                    
                    current_module = DataModule(
                        document_id=doc_id,
                        dmc=dmc,
                        title=result["title"],
                        info_code=result["info_code"],
                        item_location=result["item_location"],
                        sequence=sequence,
                        verbatim_content=chunk,
                        ste_content=result["ste"],
                        type=result["type"]
                    )
                    session.add(current_module)
                    sequence += 1
                else:
                    # Append to current module
                    current_module.verbatim_content += "\n\n" + chunk
                    current_module.ste_content += "\n\n" + result["ste"]
            
            session.commit()
        
        # Extract and process images
        await manager.broadcast({
            "type": "progress",
            "phase": "images_processing",
            "doc_id": doc_id,
            "detail": "Extracting images from PDF..."
        })
        
        images = extract_images_from_pdf(file_path)
        
        with Session(engine) as session:
            for image_path in images:
                try:
                    # Generate ICN
                    with open(image_path, 'rb') as img_file:
                        image_hash = hashlib.sha256(img_file.read()).hexdigest()[:8]
                    icn = f"ICN-{image_hash}"
                    
                    # Get caption and objects
                    vision_result = await caption_objects(image_path)
                    
                    # Find appropriate data module (for now, use first one)
                    data_module = session.exec(
                        select(DataModule).where(DataModule.document_id == doc_id)
                    ).first()
                    
                    if data_module:
                        icn_record = ICN(
                            document_id=doc_id,
                            data_module_id=data_module.id,
                            icn=icn,
                            image_path=image_path,
                            caption=vision_result["caption"],
                            objects=json.dumps(vision_result["objects"])
                        )
                        session.add(icn_record)
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    continue
            
            session.commit()
        
        # Update document status
        with Session(engine) as session:
            document = session.get(Document, doc_id)
            document.status = "completed"
            session.commit()
        
        await manager.broadcast({
            "type": "progress",
            "phase": "finished",
            "doc_id": doc_id,
            "detail": "Document processing completed successfully"
        })
        
    except Exception as e:
        print(f"Error processing document: {e}")
        with Session(engine) as session:
            document = session.get(Document, doc_id)
            if document:
                document.status = "failed"
                session.commit()
        
        await manager.broadcast({
            "type": "progress",
            "phase": "error",
            "doc_id": doc_id,
            "detail": f"Processing failed: {str(e)}"
        })

@app.get("/api/documents", response_model=List[DocumentResponse])
async def get_documents():
    """Get all documents"""
    with Session(engine) as session:
        documents = session.exec(select(Document)).all()
        return documents

@app.get("/api/data-modules", response_model=List[DataModuleResponse])
async def get_data_modules(document_id: Optional[str] = None):
    """Get data modules, optionally filtered by document"""
    with Session(engine) as session:
        query = select(DataModule)
        if document_id:
            query = query.where(DataModule.document_id == document_id)
        
        modules = session.exec(query).all()
        return modules

@app.get("/api/icns", response_model=List[ICNResponse])
async def get_icns(document_id: Optional[str] = None):
    """Get ICNs, optionally filtered by document"""
    with Session(engine) as session:
        query = select(ICN)
        if document_id:
            query = query.where(ICN.document_id == document_id)
        
        icns = session.exec(query).all()
        response = []
        for icn in icns:
            response.append(ICNResponse(
                id=icn.id,
                icn=icn.icn,
                caption=icn.caption,
                objects=json.loads(icn.objects)
            ))
        return response

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time progress updates"""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# Mount static files for serving the frontend
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)