import os
import json
import hashlib
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
from PIL.Image import Image
import tempfile
import uuid
from datetime import datetime
import re
import shutil

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Request, Form, Form
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

# Project Management
PROJECTS_DIR = Path("projects")
PROJECTS_CONFIG_FILE = "projects.json"

class ProjectManager:
    def __init__(self):
        self.projects_dir = PROJECTS_DIR
        self.projects_config = PROJECTS_CONFIG_FILE
        self.current_project = None
        self.current_engine = None
        self.ensure_projects_directory()
        
    def ensure_projects_directory(self):
        """Ensure projects directory exists"""
        self.projects_dir.mkdir(exist_ok=True)
        
    def load_projects_config(self) -> Dict[str, Any]:
        """Load projects configuration"""
        if Path(self.projects_config).exists():
            try:
                with open(self.projects_config, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"projects": [], "current_project": None}
        return {"projects": [], "current_project": None}
        
    def save_projects_config(self, config: Dict[str, Any]):
        """Save projects configuration"""
        with open(self.projects_config, 'w') as f:
            json.dump(config, f, indent=2)
            
    def create_project(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new project"""
        project_id = str(uuid.uuid4())
        project_dir = self.projects_dir / project_id
        project_dir.mkdir(exist_ok=True)
        
        # Create project uploads directory
        uploads_dir = project_dir / "uploads"
        uploads_dir.mkdir(exist_ok=True)
        
        project_data = {
            "id": project_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "database_path": str(project_dir / "aquila.db"),
            "uploads_path": str(uploads_dir)
        }
        
        # Load and update projects config
        config = self.load_projects_config()
        config["projects"].append(project_data)
        self.save_projects_config(config)
        
        return project_data
        
    def get_projects(self) -> List[Dict[str, Any]]:
        """Get all projects"""
        config = self.load_projects_config()
        return config.get("projects", [])
        
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific project"""
        projects = self.get_projects()
        return next((p for p in projects if p["id"] == project_id), None)
        
    def delete_project(self, project_id: str) -> bool:
        """Delete a project"""
        project = self.get_project(project_id)
        if not project:
            return False
            
        # Remove project directory
        project_dir = self.projects_dir / project_id
        if project_dir.exists():
            shutil.rmtree(project_dir)
            
        # Update projects config
        config = self.load_projects_config()
        config["projects"] = [p for p in config["projects"] if p["id"] != project_id]
        if config.get("current_project") == project_id:
            config["current_project"] = None
            
        self.save_projects_config(config)
        return True
        
    def set_current_project(self, project_id: str) -> bool:
        """Set current project and initialize database"""
        project = self.get_project(project_id)
        if not project:
            return False
            
        self.current_project = project
        
        # Initialize database engine for this project
        database_url = f"sqlite:///{project['database_path']}"
        self.current_engine = create_engine(database_url)
        
        # Create tables if they don't exist
        SQLModel.metadata.create_all(self.current_engine)
        
        # Update current project in config
        config = self.load_projects_config()
        config["current_project"] = project_id
        self.save_projects_config(config)
        
        return True
        
    def get_current_project(self) -> Optional[Dict[str, Any]]:
        """Get current project"""
        return self.current_project
        
    def get_current_engine(self):
        """Get current database engine"""
        return self.current_engine
        
    def get_uploads_path(self) -> str:
        """Get uploads path for current project"""
        if self.current_project:
            return self.current_project["uploads_path"]
        return "/tmp/aquila_uploads"

# Initialize project manager
project_manager = ProjectManager()

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
    type: str  # procedure, description, fault_isolation, etc.
    
    # S1000D specific structured content
    prerequisites: Optional[str] = Field(default="")  # Prerequisites and conditions
    tools_equipment: Optional[str] = Field(default="")  # Required tools and equipment
    warnings: Optional[str] = Field(default="")  # Safety warnings
    cautions: Optional[str] = Field(default="")  # Cautions and notes
    procedural_steps: Optional[str] = Field(default="")  # JSON string of structured steps
    expected_results: Optional[str] = Field(default="")  # Expected outcomes
    
    # Technical data
    specifications: Optional[str] = Field(default="")  # Technical specifications
    references: Optional[str] = Field(default="")  # Reference materials
    
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
class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: str

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
    prerequisites: str
    tools_equipment: str
    warnings: str
    cautions: str
    procedural_steps: str
    expected_results: str
    specifications: str
    references: str

class ICNResponse(BaseModel):
    id: str
    icn: str
    caption: str
    objects: List[str]

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
    """Classify text chunk according to S1000D standards and extract structured STE content"""
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert in S1000D technical documentation and ASD-STE100 Simplified Technical English.

Analyze the given text and return a JSON response with the following structure:

{
  "type": "procedure|description|fault_isolation|theory_of_operation|maintenance_planning|support_equipment",
  "title": "Clear, descriptive title following S1000D naming conventions",
  "info_code": "S1000D info code (040=description, 520=procedure, 730=fault_isolation, 710=theory, 320=maintenance_planning, 920=support_equipment)",
  "item_location": "S1000D item location code (A, B, C, etc.)",
  "ste": "Text rewritten in ASD-STE100 Simplified Technical English with controlled vocabulary",
  "should_start_new_module": true,
  "prerequisites": "Prerequisites and initial conditions required",
  "tools_equipment": "Required tools, equipment, and consumables",
  "warnings": "Safety warnings and critical information",
  "cautions": "Important cautions and notes",
  "procedural_steps": [],
  "expected_results": "Expected outcomes and verification steps",
  "specifications": "Technical specifications and tolerances",
  "references": "Reference materials and related documents"
}

IMPORTANT S1000D RULES:
1. Use proper S1000D info codes: 040 (description), 520 (procedure), 730 (fault isolation)
2. Procedures should have clear step-by-step instructions
3. Include all safety information (warnings, cautions)
4. STE should use controlled vocabulary, simple sentences, active voice
5. Group related content logically - don't split coherent procedures
6. Identify prerequisites, tools, and expected results for procedures
7. Use proper technical terminology but simplified grammar for STE

STE RULES:
- Use active voice: "Remove the plug" not "The plug should be removed"
- Use simple sentences with one main action
- Use approved vocabulary only
- Use present tense for procedures
- Use specific nouns, not pronouns
- Maximum 25 words per sentence
- Use parallel structure for similar actions"""
                },
                {
                    "role": "user",
                    "content": f"Process this text according to S1000D standards and convert to STE:\n\n{text}"
                }
            ]
        )
        
        # Debug: Print the raw response
        raw_response = response.choices[0].message.content
        print(f"DEBUG: Raw OpenAI response: {raw_response}")
        
        # Check if response is empty or None
        if not raw_response or raw_response.strip() == "":
            print("DEBUG: Empty response from OpenAI")
            raise ValueError("Empty response from OpenAI")
        
        # Clean up the response - remove markdown code blocks if present
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]  # Remove ```json
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]  # Remove ```
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]  # Remove ending ```
        cleaned_response = cleaned_response.strip()
        
        print(f"DEBUG: Cleaned response: {cleaned_response}")
        
        # Try to parse JSON
        result = json.loads(cleaned_response)
        
        # Validate required fields and set defaults
        required_fields = ["type", "title", "info_code", "item_location", "ste", "should_start_new_module"]
        for field in required_fields:
            if field not in result:
                result[field] = get_default_value(field)
        
        # Ensure all text fields are strings
        for field in ["prerequisites", "tools_equipment", "warnings", "cautions", "expected_results", "specifications", "references"]:
            if field not in result:
                result[field] = ""
            elif not isinstance(result[field], str):
                result[field] = str(result[field])
        
        # Ensure procedural_steps is a JSON string
        if isinstance(result.get("procedural_steps"), list):
            result["procedural_steps"] = json.dumps(result["procedural_steps"])
        elif not result.get("procedural_steps"):
            result["procedural_steps"] = json.dumps([])
        elif not isinstance(result.get("procedural_steps"), str):
            result["procedural_steps"] = json.dumps([])
            
        return result
        
    except json.JSONDecodeError as e:
        print(f"DEBUG: JSON decode error: {e}")
        print(f"DEBUG: Raw response that failed to parse: {raw_response if 'raw_response' in locals() else 'No response captured'}")
        print(f"DEBUG: Cleaned response that failed to parse: {cleaned_response if 'cleaned_response' in locals() else 'No cleaned response'}")
        # Enhanced fallback response
        return {
            "type": "description",
            "title": "Content Section",
            "info_code": "040",
            "item_location": "A",
            "ste": text,
            "should_start_new_module": True,
            "prerequisites": "",
            "tools_equipment": "",
            "warnings": "",
            "cautions": "",
            "procedural_steps": json.dumps([]),
            "expected_results": "",
            "specifications": "",
            "references": ""
        }
    except Exception as e:
        print(f"OpenAI classify_extract error: {e}")
        # Enhanced fallback response
        return {
            "type": "description",
            "title": "Content Section",
            "info_code": "040",
            "item_location": "A",
            "ste": text,
            "should_start_new_module": True,
            "prerequisites": "",
            "tools_equipment": "",
            "warnings": "",
            "cautions": "",
            "procedural_steps": json.dumps([]),
            "expected_results": "",
            "specifications": "",
            "references": ""
        }

def get_default_value(field: str) -> Any:
    """Get default value for missing fields"""
    defaults = {
        "type": "description",
        "title": "Content Section",
        "info_code": "040",
        "item_location": "A",
        "ste": "",
        "should_start_new_module": True
    }
    return defaults.get(field, "")

async def caption_objects(image_path: str) -> Dict[str, Any]:
    """Generate caption and detect objects in image"""
    from PIL import Image
    import io
    
    try:
        # Verify and potentially convert the image before sending to OpenAI
        try:
            # Open and verify the image
            with Image.open(image_path) as img:
                img.verify()
            
            # Re-open for processing (verify() closes the image)
            with Image.open(image_path) as img:
                # Ensure the image is in a format OpenAI supports
                if img.format not in ['JPEG', 'PNG', 'GIF', 'WEBP']:
                    print(f"Converting image from {img.format} to JPEG for OpenAI compatibility")
                    
                    # Convert to RGB if necessary
                    if img.mode in ('RGBA', 'LA', 'P'):
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                        img = rgb_img
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save as JPEG
                    temp_path = image_path.rsplit('.', 1)[0] + '_converted.jpg'
                    img.save(temp_path, 'JPEG', quality=85, optimize=True)
                    image_path = temp_path
        
        except Exception as img_error:
            print(f"Image verification/conversion error: {img_error}")
            # Continue with original path and hope for the best
        
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
    """Extract images from PDF and save them in a format supported by OpenAI"""
    from PIL import Image
    import io
    
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
                                    
                                    # Use project-specific upload directory
                                    upload_dir = Path(project_manager.get_uploads_path())
                                    upload_dir.mkdir(exist_ok=True)
                                    
                                    image_path = upload_dir / filename
                                    
                                    # Convert image to JPEG format using PIL
                                    try:
                                        # Try to open the image data with PIL
                                        image = Image.open(io.BytesIO(data))
                                        
                                        # Convert to RGB if necessary (for JPEG compatibility)
                                        if image.mode in ('RGBA', 'LA', 'P'):
                                            # Convert to RGB for JPEG
                                            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                                            if image.mode == 'P':
                                                image = image.convert('RGBA')
                                            rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                                            image = rgb_image
                                        elif image.mode != 'RGB':
                                            image = image.convert('RGB')
                                        
                                        # Save as JPEG with good quality
                                        image.save(image_path, 'JPEG', quality=85, optimize=True)
                                        images.append(str(image_path))
                                        
                                    except Exception as pil_error:
                                        print(f"PIL conversion failed for image {image_hash}: {pil_error}")
                                        # Fallback: try to save raw data and hope it works
                                        try:
                                            with open(image_path, 'wb') as img_file:
                                                img_file.write(data)
                                            # Verify it's a valid image by trying to open it
                                            test_image = Image.open(image_path)
                                            test_image.verify()
                                            images.append(str(image_path))
                                        except Exception as fallback_error:
                                            print(f"Fallback save failed for image {image_hash}: {fallback_error}")
                                            # Clean up failed file
                                            if image_path.exists():
                                                image_path.unlink()
                                            continue
                                            
                            except Exception as e:
                                print(f"Error extracting image: {e}")
                                continue
    except Exception as e:
        print(f"Error in image extraction: {e}")
    
    return images

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Aquila S1000D-AI with Project Management initialized successfully")
    yield
    print("Shutting down Aquila S1000D-AI")

# FastAPI app
app = FastAPI(title="Aquila S1000D-AI with Project Management", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Project Management API Routes
@app.get("/api/projects", response_model=List[ProjectResponse])
async def get_projects():
    """Get all projects"""
    projects = project_manager.get_projects()
    return [ProjectResponse(
        id=p["id"],
        name=p["name"],
        description=p.get("description", ""),
        created_at=p["created_at"]
    ) for p in projects]

@app.post("/api/projects")
async def create_project(name: str = Form(...), description: str = Form("")):
    """Create a new project"""
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="Project name is required")
    
    project = project_manager.create_project(name.strip(), description.strip())
    return ProjectResponse(
        id=project["id"],
        name=project["name"],
        description=project.get("description", ""),
        created_at=project["created_at"]
    )

@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project"""
    success = project_manager.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"message": "Project deleted successfully"}

@app.post("/api/projects/{project_id}/select")
async def select_project(project_id: str):
    """Select a project as current"""
    success = project_manager.set_current_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"message": "Project selected successfully"}

@app.get("/api/projects/current")
async def get_current_project():
    """Get current project"""
    project = project_manager.get_current_project()
    if not project:
        return {"current_project": None}
    
    return {
        "current_project": ProjectResponse(
            id=project["id"],
            name=project["name"],
            description=project.get("description", ""),
            created_at=project["created_at"]
        )
    }

# Application API Routes
@app.get("/api/health")
async def health_check():
    current_project = project_manager.get_current_project()
    return {
        "status": "healthy", 
        "service": "Aquila S1000D-AI",
        "current_project": current_project["name"] if current_project else None
    }

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...), operational_context: str = "Water"):
    """Upload and process a PDF document"""
    # Check if project is selected
    if not project_manager.get_current_project():
        raise HTTPException(status_code=400, detail="No project selected. Please select a project first.")
        
    try:
        # Create project-specific upload directory
        upload_dir = Path(project_manager.get_uploads_path())
        upload_dir.mkdir(exist_ok=True)
        
        # Calculate SHA-256 and save file
        temp_path = upload_dir / f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        file_hash = calculate_sha256(str(temp_path))
        final_path = upload_dir / f"{file_hash}.pdf"
        temp_path.rename(final_path)
        
        # Create document record in current project's database
        engine = project_manager.get_current_engine()
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

def find_best_module_for_image(session: Session, doc_id: str, caption: str) -> Optional[DataModule]:
    """Find the most appropriate data module for an image based on its caption"""
    modules = session.exec(select(DataModule).where(DataModule.document_id == doc_id)).all()
    
    if not modules:
        return None
    
    # Simple scoring based on caption content matching module titles and content
    best_module = None
    best_score = 0
    
    caption_lower = caption.lower()
    
    for module in modules:
        score = 0
        
        # Check title match
        if any(word in caption_lower for word in module.title.lower().split()):
            score += 3
        
        # Check content match
        if any(word in module.verbatim_content.lower() for word in caption_lower.split()):
            score += 2
        
        # Prefer procedure modules for images showing steps
        if module.type == "procedure" and any(word in caption_lower for word in ["step", "procedure", "process", "install", "remove", "check"]):
            score += 1
        
        if score > best_score:
            best_score = score
            best_module = module
    
    # If no good match found, return first module
    return best_module or modules[0]

async def process_document(doc_id: str, file_path: str, operational_context: str):
    """Process the uploaded PDF document with improved S1000D compliance"""
    try:
        # Get current project's database engine
        engine = project_manager.get_current_engine()
        
        # Extract text
        await manager.broadcast({
            "type": "progress",
            "phase": "text_extraction",
            "doc_id": doc_id,
            "detail": "Extracting text from PDF...",
            "processing_type": "Text Extraction",
            "current_text": "Reading PDF file structure..."
        })
        
        text = extract_text(file_path, laparams=LAParams())
        chunks = chunk_text(text)  # Using existing chunk_text function
        
        # Process chunks and create data modules
        await manager.broadcast({
            "type": "progress",
            "phase": "text_extracted",
            "doc_id": doc_id,
            "detail": f"Processing {len(chunks)} logical sections...",
            "processing_type": "Document Analysis",
            "current_text": f"Divided document into {len(chunks)} logical sections for analysis"
        })
        
        with Session(engine) as session:
            sequence = 1
            current_module = None
            
            for i, chunk in enumerate(chunks):
                # Truncate text for display (first 150 characters)
                truncated_text = chunk[:150] + "..." if len(chunk) > 150 else chunk
                
                # Send progress update for classification
                await manager.broadcast({
                    "type": "progress",
                    "phase": "classification",
                    "doc_id": doc_id,
                    "detail": f"Classifying section {i+1} of {len(chunks)}",
                    "processing_type": "AI Classification",
                    "current_text": truncated_text,
                    "progress_section": f"{i+1}/{len(chunks)}"
                })
                
                # Classify and extract structured content
                result = await classify_extract(chunk)
                
                # Send progress update for STE conversion
                await manager.broadcast({
                    "type": "progress",
                    "phase": "ste_conversion",
                    "doc_id": doc_id,
                    "detail": f"Converting section {i+1} to Simplified Technical English",
                    "processing_type": "STE Conversion",
                    "current_text": result.get("ste", "")[:150] + "..." if len(result.get("ste", "")) > 150 else result.get("ste", ""),
                    "progress_section": f"{i+1}/{len(chunks)}"
                })
                
                if result.get("should_start_new_module", True) or current_module is None:
                    # Create new data module with all S1000D fields
                    dmc = generate_dmc(
                        operational_context,
                        result["type"],
                        result["info_code"],
                        result["item_location"],
                        sequence
                    )
                    
                    # Send progress update for module creation
                    await manager.broadcast({
                        "type": "progress",
                        "phase": "module_creation",
                        "doc_id": doc_id,
                        "detail": f"Creating data module: {result['title']}",
                        "processing_type": "Module Creation",
                        "current_text": f"DMC: {dmc} | Title: {result['title']} | Type: {result['type']}",
                        "progress_section": f"{i+1}/{len(chunks)}"
                    })
                    
                    current_module = DataModule(
                        document_id=doc_id,
                        dmc=dmc,
                        title=result["title"],
                        info_code=result["info_code"],
                        item_location=result["item_location"],
                        sequence=sequence,
                        verbatim_content=chunk,
                        ste_content=result["ste"],
                        type=result["type"],
                        prerequisites=result.get("prerequisites", ""),
                        tools_equipment=result.get("tools_equipment", ""),
                        warnings=result.get("warnings", ""),
                        cautions=result.get("cautions", ""),
                        procedural_steps=result.get("procedural_steps", "[]"),
                        expected_results=result.get("expected_results", ""),
                        specifications=result.get("specifications", ""),
                        references=result.get("references", "")
                    )
                    session.add(current_module)
                    sequence += 1
                else:
                    # Append to current module (only for closely related content)
                    current_module.verbatim_content += "\n\n" + chunk
                    current_module.ste_content += "\n\n" + result["ste"]
                    
                    # Merge structured content intelligently
                    if result.get("prerequisites") and isinstance(result["prerequisites"], str):
                        current_module.prerequisites += "\n" + result["prerequisites"]
                    if result.get("tools_equipment") and isinstance(result["tools_equipment"], str):
                        current_module.tools_equipment += "\n" + result["tools_equipment"]
                    if result.get("warnings") and isinstance(result["warnings"], str):
                        current_module.warnings += "\n" + result["warnings"]
                    if result.get("cautions") and isinstance(result["cautions"], str):
                        current_module.cautions += "\n" + result["cautions"]
                    if result.get("expected_results") and isinstance(result["expected_results"], str):
                        current_module.expected_results += "\n" + result["expected_results"]
                    if result.get("specifications") and isinstance(result["specifications"], str):
                        current_module.specifications += "\n" + result["specifications"]
                    if result.get("references") and isinstance(result["references"], str):
                        current_module.references += "\n" + result["references"]
                    
                    # Merge procedural steps
                    try:
                        existing_steps = json.loads(current_module.procedural_steps or "[]")
                        new_steps = json.loads(result.get("procedural_steps", "[]"))
                        if new_steps:
                            existing_steps.extend(new_steps)
                            current_module.procedural_steps = json.dumps(existing_steps)
                    except json.JSONDecodeError:
                        pass
            
            session.commit()
            
            # Send completion update for modules
            await manager.broadcast({
                "type": "progress",
                "phase": "modules_created",
                "doc_id": doc_id,
                "detail": f"Created {sequence-1} data modules",
                "processing_type": "Module Creation Complete",
                "current_text": f"Successfully created {sequence-1} S1000D data modules"
            })
        
        # Extract and process images
        await manager.broadcast({
            "type": "progress",
            "phase": "images_processing",
            "doc_id": doc_id,
            "detail": "Extracting images from PDF...",
            "processing_type": "Image Extraction",
            "current_text": "Scanning PDF for embedded images..."
        })
        
        images = extract_images_from_pdf(file_path)
        
        if images:
            await manager.broadcast({
                "type": "progress",
                "phase": "images_processing",
                "doc_id": doc_id,
                "detail": f"Found {len(images)} images, generating captions...",
                "processing_type": "Image Analysis",
                "current_text": f"Processing {len(images)} images with AI vision analysis"
            })
        
        with Session(engine) as session:
            for i, image_path in enumerate(images):
                try:
                    # Generate ICN
                    with open(image_path, 'rb') as img_file:
                        image_hash = hashlib.sha256(img_file.read()).hexdigest()[:8]
                    icn = f"ICN-{image_hash}"
                    
                    # Send progress update for image processing
                    await manager.broadcast({
                        "type": "progress",
                        "phase": "image_analysis",
                        "doc_id": doc_id,
                        "detail": f"Analyzing image {i+1} of {len(images)}",
                        "processing_type": "AI Vision Analysis",
                        "current_text": f"Generating caption for image {icn}...",
                        "progress_section": f"{i+1}/{len(images)}"
                    })
                    
                    # Get caption and objects
                    vision_result = await caption_objects(image_path)
                    
                    # Find most appropriate data module for this image
                    data_module = find_best_module_for_image(session, doc_id, vision_result["caption"])
                    
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
                        
                        # Send progress update with caption
                        await manager.broadcast({
                            "type": "progress",
                            "phase": "image_analysis",
                            "doc_id": doc_id,
                            "detail": f"Generated caption for image {i+1}",
                            "processing_type": "Image Caption Generated",
                            "current_text": vision_result["caption"][:150] + "..." if len(vision_result["caption"]) > 150 else vision_result["caption"],
                            "progress_section": f"{i+1}/{len(images)}"
                        })
                        
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
            "detail": "Document processing completed successfully",
            "processing_type": "Complete",
            "current_text": "All processing stages completed. Document is ready for viewing."
        })
        
    except Exception as e:
        print(f"Error processing document {doc_id}: {e}")
        
        # Update document status to failed
        engine = project_manager.get_current_engine()
        with Session(engine) as session:
            document = session.get(Document, doc_id)
            document.status = "failed"
            session.commit()
        
        await manager.broadcast({
            "type": "error",
            "doc_id": doc_id,
            "detail": f"Document processing failed: {str(e)}",
            "processing_type": "Error",
            "current_text": f"Processing failed: {str(e)}"
        })

@app.get("/api/documents", response_model=List[DocumentResponse])
async def get_documents():
    """Get all documents for current project"""
    engine = project_manager.get_current_engine()
    if not engine:
        raise HTTPException(status_code=400, detail="No project selected")
        
    with Session(engine) as session:
        documents = session.exec(select(Document)).all()
        return documents

@app.get("/api/data-modules", response_model=List[DataModuleResponse])
async def get_data_modules(document_id: Optional[str] = None):
    """Get data modules for current project, optionally filtered by document"""
    engine = project_manager.get_current_engine()
    if not engine:
        raise HTTPException(status_code=400, detail="No project selected")
        
    with Session(engine) as session:
        query = select(DataModule)
        if document_id:
            query = query.where(DataModule.document_id == document_id)
        
        modules = session.exec(query).all()
        return modules

@app.get("/api/icns", response_model=List[ICNResponse])
async def get_icns(document_id: Optional[str] = None):
    """Get ICNs for current project, optionally filtered by document"""
    engine = project_manager.get_current_engine()
    if not engine:
        raise HTTPException(status_code=400, detail="No project selected")
        
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
    uvicorn.run(app, host="0.0.0.0", port=8001)