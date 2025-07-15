import os
import json
import hashlib
import asyncio
import re
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import tempfile
import uuid
from datetime import datetime
import shutil
import tiktoken

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Request, Form, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
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

class DocumentPlan(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    document_id: str = Field(foreign_key="document.id")
    plan_data: str  # JSON string containing the planning data
    planning_confidence: float = Field(default=0.0)
    total_chunks_analyzed: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = Field(default="planned")  # planned, populating, completed, failed

class DataModule(SQLModel, table=True):
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    document_id: str = Field(foreign_key="document.id")
    plan_id: Optional[str] = Field(foreign_key="documentplan.id", default=None)
    module_id: str  # From planning phase
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
    
    # Enhanced fields for new system
    content_sources: Optional[str] = Field(default="")  # JSON string of source chunks
    completeness_score: Optional[float] = Field(default=0.0)  # 0.0-1.0 completeness
    relevant_chunks_found: Optional[int] = Field(default=0)
    total_chunks_analyzed: Optional[int] = Field(default=0)
    population_status: Optional[str] = Field(default="pending")  # pending, complete, partial, error
    
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

# Enhanced OpenAI helper functions with detailed logging
async def classify_extract(text: str) -> Dict[str, Any]:
    """Classify text chunk according to S1000D standards and extract structured STE content"""
    print(f"\n{'='*60}")
    print(f"CLASSIFY_EXTRACT - Starting AI call")
    print(f"Input text length: {len(text)} characters")
    print(f"Input text preview: {text[:200]}...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        
        system_prompt = """You are an expert in S1000D technical documentation and ASD-STE100 Simplified Technical English.

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

        user_prompt = f"Process this text according to S1000D standards and convert to STE:\n\n{text}"
        
        print(f"AI REQUEST:")
        print(f"Model: gpt-4o-mini")
        print(f"Temperature: 0")
        print(f"System prompt length: {len(system_prompt)} characters")
        print(f"User prompt length: {len(user_prompt)} characters")
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Debug: Print the raw response
        raw_response = response.choices[0].message.content
        elapsed_time = time.time() - start_time
        
        print(f"\nAI RESPONSE:")
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print(f"Raw response length: {len(raw_response)} characters")
        print(f"Raw response: {raw_response}")
        print(f"{'='*60}")
        
        # Check if response is empty or None
        if not raw_response or raw_response.strip() == "":
            print("ERROR: Empty response from OpenAI")
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
        
        print(f"Cleaned response: {cleaned_response}")
        
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
            
        print(f"SUCCESS: Parsed and validated result")
        return result
        
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON decode error: {e}")
        print(f"Raw response that failed to parse: {raw_response if 'raw_response' in locals() else 'No response captured'}")
        print(f"Cleaned response that failed to parse: {cleaned_response if 'cleaned_response' in locals() else 'No cleaned response'}")
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
        print(f"ERROR: OpenAI classify_extract error: {e}")
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
    """Generate caption and detect objects in image with robust error handling"""
    from PIL import Image, ImageFile
    import io
    import base64
    
    print(f"\n{'='*60}")
    print(f"CAPTION_OBJECTS - Starting AI vision call")
    print(f"Image path: {image_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Enable loading of truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    try:
        # Verify and potentially convert the image before sending to OpenAI
        processed_image_path = image_path
        
        try:
            # Open and verify the image
            with Image.open(image_path) as img:
                print(f"Image loaded: format={img.format}, mode={img.mode}, size={img.size}")
                
                # Check if image has valid dimensions
                if img.size[0] == 0 or img.size[1] == 0:
                    print(f"Image has invalid dimensions: {img.size}")
                    raise ValueError("Invalid image dimensions")
                
                # Always ensure we have a supported format for OpenAI
                needs_conversion = (
                    img.format not in ['JPEG', 'PNG', 'GIF', 'WEBP'] or
                    img.mode not in ['RGB', 'L', 'P'] or
                    img.size[0] > 2048 or img.size[1] > 2048  # OpenAI size limits
                )
                
                if needs_conversion:
                    print(f"Converting image from {img.format}/{img.mode} to JPEG for OpenAI compatibility")
                    
                    # Convert to RGB if necessary
                    if img.mode in ('RGBA', 'LA', 'P'):
                        rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            try:
                                img = img.convert('RGBA')
                            except Exception:
                                img = img.convert('RGB')
                        
                        if img.mode in ('RGBA', 'LA'):
                            try:
                                # Use alpha channel as mask if available
                                alpha = img.split()[-1]
                                rgb_img.paste(img, mask=alpha)
                            except Exception:
                                # If alpha handling fails, just paste RGB channels
                                rgb_img.paste(img.convert('RGB'))
                        else:
                            rgb_img.paste(img)
                        
                        img = rgb_img
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Resize if too large
                    if img.size[0] > 2048 or img.size[1] > 2048:
                        img.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                    
                    # Save as JPEG
                    temp_path = str(Path(image_path).with_suffix('_converted.jpg'))
                    img.save(temp_path, 'JPEG', quality=85, optimize=True)
                    processed_image_path = temp_path
                    print(f"Image converted and saved to: {processed_image_path}")
        
        except Exception as img_error:
            print(f"Image verification/conversion error: {img_error}")
            # Try to create a minimal valid image as fallback
            try:
                fallback_img = Image.new('RGB', (100, 100), color=(128, 128, 128))
                fallback_path = str(Path(image_path).with_suffix('_fallback.jpg'))
                fallback_img.save(fallback_path, 'JPEG', quality=85)
                processed_image_path = fallback_path
                print(f"Created fallback image: {processed_image_path}")
            except Exception as fallback_error:
                print(f"Fallback image creation failed: {fallback_error}")
                # Return early with fallback response
                return {
                    "caption": "Technical diagram (processing error)",
                    "objects": ["component", "system"]
                }
        
        # Attempt to call OpenAI API
        client = openai.OpenAI(api_key=openai.api_key)
        
        with open(processed_image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        print(f"AI VISION REQUEST:")
        print(f"Model: gpt-4o-mini")
        print(f"Temperature: 0")
        print(f"Base64 image size: {len(base64_image)} characters")
        
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
        
        # Parse the response
        raw_response = response.choices[0].message.content
        elapsed_time = time.time() - start_time
        
        print(f"\nAI VISION RESPONSE:")
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print(f"Raw response: {raw_response}")
        print(f"{'='*60}")
        
        # Clean up the response - remove markdown code blocks if present
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
        cleaned_response = cleaned_response.strip()
        
        result = json.loads(cleaned_response)
        
        # Clean up temporary files
        if processed_image_path != image_path and Path(processed_image_path).exists():
            try:
                Path(processed_image_path).unlink()
            except Exception:
                pass
        
        print(f"SUCCESS: Vision analysis complete")
        return result
        
    except Exception as e:
        print(f"ERROR: OpenAI caption_objects error: {e}")
        
        # Clean up temporary files
        if 'processed_image_path' in locals() and processed_image_path != image_path:
            try:
                if Path(processed_image_path).exists():
                    Path(processed_image_path).unlink()
            except Exception:
                pass
        
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

class TextCleaner:
    """Advanced text cleaning to remove headers, footers, page numbers"""
    
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
    def clean_extracted_text(self, raw_text: str) -> Dict[str, str]:
        """
        Clean text by removing:
        - Headers and footers (repeated content)
        - Page numbers
        - Navigation elements
        - Metadata stamps
        """
        lines = raw_text.split('\n')
        cleaned_lines = []
        removed_elements = []
        
        # Patterns for common non-content elements
        page_number_patterns = [
            r'^\s*\d+\s*$',  # Just a number
            r'^\s*Page\s+\d+\s*$',  # "Page X"
            r'^\s*\d+\s*of\s*\d+\s*$',  # "X of Y"
            r'^\s*-\s*\d+\s*-\s*$',  # "- X -"
        ]
        
        header_footer_patterns = [
            r'^[A-Z\s]{10,}$',  # All caps headers
            r'^\s*[A-Z]+\s+\d+\s*$',  # Manual reference codes
            r'^\s*TM\s+\d+',  # Technical manual references
            r'^\s*Figure\s+\d+',  # Figure references at start of line
            r'^\s*Table\s+\d+',  # Table references
            r'^\s*WARNING\s*$',  # Standalone WARNING
            r'^\s*CAUTION\s*$',  # Standalone CAUTION
            r'^\s*NOTE\s*$',  # Standalone NOTE
        ]
        
        # Track repeated content (potential headers/footers)
        line_frequency = {}
        for line in lines:
            stripped = line.strip()
            if stripped and len(stripped) > 3:
                line_frequency[stripped] = line_frequency.get(stripped, 0) + 1
        
        # Find lines that appear multiple times (likely headers/footers)
        repeated_lines = {line for line, count in line_frequency.items() if count > 2 and len(line) < 100}
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
                
            # Check if it's a page number
            is_page_number = any(re.match(pattern, stripped, re.IGNORECASE) for pattern in page_number_patterns)
            if is_page_number:
                removed_elements.append(f"Page number: {stripped}")
                continue
                
            # Check if it's a header/footer pattern
            is_header_footer = any(re.match(pattern, stripped, re.IGNORECASE) for pattern in header_footer_patterns)
            if is_header_footer:
                removed_elements.append(f"Header/Footer: {stripped}")
                continue
                
            # Check if it's repeated content
            if stripped in repeated_lines:
                removed_elements.append(f"Repeated content: {stripped}")
                continue
                
            # Keep the line
            cleaned_lines.append(line)
        
        clean_text = '\n'.join(cleaned_lines)
        
        # Additional cleaning
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)  # Multiple newlines to double
        clean_text = re.sub(r'[ \t]+', ' ', clean_text)  # Multiple spaces to single
        clean_text = clean_text.strip()
        
        return {
            "clean_text": clean_text,
            "removed_elements": removed_elements,
            "cleaning_report": f"Removed {len(removed_elements)} non-content elements"
        }

class ChunkingStrategy:
    """Dual chunking approach for planning and population phases"""
    
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def create_planning_chunks(self, text: str) -> List[str]:
        """
        Create large chunks for planning phase:
        - 2000 tokens per chunk
        - 200 token overlap between chunks
        - Maintain context for comprehensive planning
        """
        return self._create_chunks(text, target_tokens=2000, overlap_tokens=200)
    
    def create_population_chunks(self, text: str) -> List[str]:
        """
        Create smaller chunks for population phase:
        - 400 tokens per chunk
        - 50 token overlap between chunks
        - Optimized for detailed content extraction
        """
        return self._create_chunks(text, target_tokens=400, overlap_tokens=50)
    
    def extract_keyword_chunks(self, text: str, keywords: List[str], chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Extract chunks around keyword matches with specified size
        """
        chunks = []
        text_lower = text.lower()
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Find all occurrences of the keyword
            start = 0
            while True:
                pos = text_lower.find(keyword_lower, start)
                if pos == -1:
                    break
                
                # Calculate chunk boundaries
                chunk_start = max(0, pos - chunk_size // 2)
                chunk_end = min(len(text), pos + len(keyword) + chunk_size // 2)
                
                # Extract the chunk
                chunk_text = text[chunk_start:chunk_end]
                
                # Count tokens to ensure we're within limits
                token_count = self.count_tokens(chunk_text)
                if token_count > chunk_size:
                    # Trim to fit token limit
                    encoded = self.encoding.encode(chunk_text)
                    trimmed = self.encoding.decode(encoded[:chunk_size])
                    chunk_text = trimmed
                
                chunks.append({
                    "keyword": keyword,
                    "position": pos,
                    "content": chunk_text,
                    "token_count": self.count_tokens(chunk_text)
                })
                
                # Move to next potential match
                start = pos + len(keyword)
        
        # Remove duplicates based on position overlap
        unique_chunks = []
        for chunk in chunks:
            is_duplicate = False
            for existing in unique_chunks:
                # Check if chunks overlap significantly
                if abs(chunk["position"] - existing["position"]) < chunk_size // 4:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _create_chunks(self, text: str, target_tokens: int, overlap_tokens: int) -> List[str]:
        """Create chunks with specified token limits and overlap"""
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If adding this sentence would exceed target, start new chunk
            if current_tokens + sentence_tokens > target_tokens and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Create overlap for next chunk
                overlap_text = self._create_overlap(current_chunk, overlap_tokens)
                current_chunk = overlap_text + sentence
                current_tokens = self.count_tokens(current_chunk)
            else:
                current_chunk += sentence
                current_tokens += sentence_tokens
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving structure"""
        # Split by periods, but be careful about abbreviations
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            
            # Check for sentence endings
            if char in '.!?':
                # Look ahead to see if this is really a sentence end
                # (not just an abbreviation)
                if len(current_sentence.strip()) > 10:  # Minimum sentence length
                    sentences.append(current_sentence)
                    current_sentence = ""
            elif char == '\n':
                # Paragraph breaks
                if current_sentence.strip():
                    sentences.append(current_sentence)
                    current_sentence = ""
        
        # Add any remaining text
        if current_sentence.strip():
            sentences.append(current_sentence)
        
        return sentences
    
    def _create_overlap(self, text: str, overlap_tokens: int) -> str:
        """Create overlap text from the end of current chunk"""
        tokens = self.encoding.encode(text)
        if len(tokens) <= overlap_tokens:
            return text
        
        # Get the last overlap_tokens tokens
        overlap_token_ids = tokens[-overlap_tokens:]
        overlap_text = self.encoding.decode(overlap_token_ids)
        
        # Try to start at a sentence boundary
        sentences = overlap_text.split('.')
        if len(sentences) > 1:
            # Start from the second sentence to avoid partial sentences
            return '.'.join(sentences[1:]).strip()
        
        return overlap_text

class EnhancedDocumentPlanner:
    """Enhanced AI-powered planning system with keyword extraction for optimized population"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=openai.api_key)
        self.chunker = ChunkingStrategy()
        
    async def analyze_and_plan(self, clean_text: str, operational_context: str) -> Dict[str, Any]:
        """
        Analyze document using large chunks and create planning JSON with keyword extraction
        """
        print(f"\n{'='*60}")
        print(f"ENHANCED DOCUMENT PLANNER - Starting analysis")
        print(f"Document length: {len(clean_text)} characters")
        print(f"Operational context: {operational_context}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Create planning chunks
        planning_chunks = self.chunker.create_planning_chunks(clean_text)
        
        print(f"Created {len(planning_chunks)} planning chunks")
        
        # Initialize planning data
        planning_data = {
            "planned_modules": [],
            "document_summary": "",
            "total_planning_chunks": len(planning_chunks),
            "planning_confidence": 0.0,
            "operational_context": operational_context
        }
        
        # Process each chunk to build comprehensive plan
        for i, chunk in enumerate(planning_chunks):
            print(f"\nProcessing planning chunk {i+1}/{len(planning_chunks)}")
            chunk_plan = await self._analyze_chunk_for_planning(chunk, planning_data, i + 1, len(planning_chunks))
            
            # Merge chunk plan with overall plan
            if i == 0:
                # First chunk - establish initial plan
                planning_data["planned_modules"] = chunk_plan.get("planned_modules", [])
                planning_data["document_summary"] = chunk_plan.get("document_summary", "")
                planning_data["planning_confidence"] = chunk_plan.get("planning_confidence", 0.0)
            else:
                # Subsequent chunks - refine and expand plan
                planning_data = await self._merge_chunk_plan(planning_data, chunk_plan)
        
        elapsed_time = time.time() - start_time
        print(f"\nPlanning complete in {elapsed_time:.2f} seconds")
        print(f"Total modules planned: {len(planning_data['planned_modules'])}")
        print(f"Planning confidence: {planning_data['planning_confidence']:.2f}")
        
        return planning_data
    
    async def _analyze_chunk_for_planning(self, chunk: str, existing_plan: Dict[str, Any], 
                                        chunk_num: int, total_chunks: int) -> Dict[str, Any]:
        """Analyze a single chunk for planning purposes with enhanced keyword extraction"""
        
        # Prepare context for AI
        context_prompt = ""
        if chunk_num > 1:
            context_prompt = f"""
            EXISTING PLAN CONTEXT:
            This is chunk {chunk_num} of {total_chunks}. Here's what we've planned so far:
            
            Existing modules: {json.dumps([{"title": m["title"], "description": m["description"]} for m in existing_plan.get("planned_modules", [])], indent=2)}
            
            Document summary so far: {existing_plan.get("document_summary", "")}
            
            Please analyze this new chunk and either:
            1. Suggest NEW modules if this chunk contains distinctly different content
            2. Suggest REFINEMENTS to existing modules if this chunk provides additional context
            3. Indicate if this chunk should be MERGED with existing modules
            
            """
        
        print(f"\nAI PLANNING REQUEST - Chunk {chunk_num}")
        print(f"Chunk length: {len(chunk)} characters")
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an expert in S1000D technical documentation planning. 
                        
                        Analyze the given text chunk and create a comprehensive plan for data modules.
                        
                        {context_prompt}
                        
                        Return a JSON response with this structure:
                        {{
                            "planned_modules": [
                                {{
                                    "module_id": "unique_id_string",
                                    "title": "Clear, descriptive title following S1000D naming conventions",
                                    "description": "Detailed description of what this module should contain",
                                    "type": "procedure|description|fault_isolation|theory_of_operation|maintenance_planning|support_equipment",
                                    "info_code": "S1000D info code (040=description, 520=procedure, 730=fault_isolation, 710=theory, 320=maintenance_planning, 920=support_equipment)",
                                    "item_location": "S1000D item location code (A, B, C, etc.)",
                                    "estimated_content_sections": ["list of expected content sections"],
                                    "priority": "high|medium|low",
                                    "chunk_source": {chunk_num},
                                    "keywords": ["comma", "separated", "list", "of", "key", "words", "and", "phrases", "from", "document", "text", "relevant", "to", "this", "module"]
                                }}
                            ],
                            "document_summary": "Overall summary of what this document covers",
                            "planning_confidence": 0.95,
                            "content_analysis": "Analysis of what type of content this chunk contains"
                        }}
                        
                        CRITICAL: For each planned module, include a "keywords" array with 10-20 specific words and phrases 
                        extracted directly from the document text that are relevant to that module. These keywords will be 
                        used to efficiently locate relevant content during population. Include:
                        - Technical terms and component names
                        - Procedure action words
                        - Specific part numbers or identifiers
                        - Important concept words
                        - Measurement units and values
                        
                        IMPORTANT S1000D PLANNING RULES:
                        1. Create logical, coherent modules that make sense as standalone units
                        2. Don't create too many small modules - combine related content
                        3. Procedures should be complete workflows, not fragments
                        4. Descriptions should cover complete systems or components
                        5. Consider the operational context: {existing_plan.get("operational_context", "Unknown")}
                        6. Each module should have clear, actionable content
                        7. Avoid duplicate or overlapping modules
                        """
                    },
                    {
                        "role": "user",
                        "content": f"Analyze this text chunk for S1000D data module planning:\n\n{chunk}"
                    }
                ]
            )
            
            # Parse response
            raw_response = response.choices[0].message.content
            
            print(f"AI PLANNING RESPONSE - Chunk {chunk_num}")
            print(f"Raw response: {raw_response}")
            
            cleaned_response = self._clean_json_response(raw_response)
            result = json.loads(cleaned_response)
            
            # Validate keywords for each module
            for module in result.get("planned_modules", []):
                if "keywords" not in module or not isinstance(module["keywords"], list):
                    module["keywords"] = []
                    print(f"WARNING: Module '{module.get('title', 'Unknown')}' missing keywords")
            
            print(f"SUCCESS: Parsed planning result with {len(result.get('planned_modules', []))} modules")
            return result
            
        except Exception as e:
            print(f"ERROR: Planning analysis failed: {e}")
            return {
                "planned_modules": [],
                "document_summary": f"Error analyzing chunk {chunk_num}",
                "planning_confidence": 0.0,
                "content_analysis": f"Error: {str(e)}"
            }
    
    async def _merge_chunk_plan(self, existing_plan: Dict[str, Any], chunk_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Merge planning data from multiple chunks with keyword consolidation"""
        
        # Merge document summary
        if chunk_plan.get("document_summary"):
            existing_plan["document_summary"] = (
                existing_plan["document_summary"] + " " + chunk_plan["document_summary"]
            ).strip()
        
        # Merge modules intelligently
        new_modules = chunk_plan.get("planned_modules", [])
        existing_modules = existing_plan.get("planned_modules", [])
        
        for new_module in new_modules:
            # Check if this module is similar to an existing one
            merged = False
            for existing_module in existing_modules:
                if self._should_merge_modules(existing_module, new_module):
                    # Merge the modules
                    existing_module["description"] = (
                        existing_module["description"] + " " + new_module["description"]
                    ).strip()
                    existing_module["estimated_content_sections"] = list(set(
                        existing_module.get("estimated_content_sections", []) + 
                        new_module.get("estimated_content_sections", [])
                    ))
                    
                    # Merge keywords
                    existing_keywords = existing_module.get("keywords", [])
                    new_keywords = new_module.get("keywords", [])
                    merged_keywords = list(set(existing_keywords + new_keywords))
                    existing_module["keywords"] = merged_keywords
                    
                    merged = True
                    break
            
            if not merged:
                # Add as new module
                existing_modules.append(new_module)
        
        existing_plan["planned_modules"] = existing_modules
        
        # Update confidence (average of all chunks processed)
        chunk_confidence = chunk_plan.get("planning_confidence", 0.0)
        existing_confidence = existing_plan.get("planning_confidence", 0.0)
        existing_plan["planning_confidence"] = (existing_confidence + chunk_confidence) / 2
        
        return existing_plan
    
    def _should_merge_modules(self, existing_module: Dict[str, Any], new_module: Dict[str, Any]) -> bool:
        """Determine if two modules should be merged"""
        # Check if titles are similar
        existing_title = existing_module.get("title", "").lower()
        new_title = new_module.get("title", "").lower()
        
        # Simple similarity check
        if existing_title in new_title or new_title in existing_title:
            return True
        
        # Check if same type and location
        if (existing_module.get("type") == new_module.get("type") and
            existing_module.get("item_location") == new_module.get("item_location")):
            return True
        
        return False
    
    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response from AI"""
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

class EnhancedContentPopulator:
    """Enhanced AI-powered content population using keyword-based chunk extraction"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=openai.api_key)
        self.chunker = ChunkingStrategy()
        
    async def populate_modules_concurrently(self, planned_modules: List[Dict[str, Any]], 
                                          clean_text: str, operational_context: str,
                                          max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """
        Populate multiple modules concurrently with rate limiting
        """
        print(f"\n{'='*60}")
        print(f"ENHANCED CONTENT POPULATOR - Starting concurrent population")
        print(f"Total modules to populate: {len(planned_modules)}")
        print(f"Max concurrent requests: {max_concurrent}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Create tasks for all modules
        tasks = []
        for i, planned_module in enumerate(planned_modules):
            task = self._populate_module_with_semaphore(
                semaphore, planned_module, clean_text, operational_context, i + 1, len(planned_modules)
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        populated_modules = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        successful_modules = []
        for i, result in enumerate(populated_modules):
            if isinstance(result, Exception):
                print(f"ERROR: Module {i+1} failed: {result}")
                # Create a fallback module
                planned_module = planned_modules[i]
                fallback_module = {
                    **planned_module,
                    "status": "error",
                    "verbatim_content": f"Error processing module: {result}",
                    "ste_content": f"Error processing module: {result}",
                    "completeness_score": 0.0,
                    "error": str(result)
                }
                successful_modules.append(fallback_module)
            else:
                successful_modules.append(result)
        
        elapsed_time = time.time() - start_time
        print(f"\nConcurrent population complete in {elapsed_time:.2f} seconds")
        print(f"Successfully populated {len(successful_modules)} modules")
        
        return successful_modules
    
    async def _populate_module_with_semaphore(self, semaphore: asyncio.Semaphore, 
                                            planned_module: Dict[str, Any], clean_text: str,
                                            operational_context: str, module_num: int, total_modules: int) -> Dict[str, Any]:
        """Populate a single module with semaphore control"""
        async with semaphore:
            return await self.populate_module_with_keywords(
                planned_module, clean_text, operational_context, module_num, total_modules
            )
    
    async def populate_module_with_keywords(self, planned_module: Dict[str, Any], clean_text: str,
                                          operational_context: str, module_num: int, total_modules: int) -> Dict[str, Any]:
        """
        Populate a specific module using keyword-based chunk extraction
        """
        module_title = planned_module.get("title", "Unknown")
        keywords = planned_module.get("keywords", [])
        
        print(f"\n{'='*40}")
        print(f"POPULATING MODULE {module_num}/{total_modules}: {module_title}")
        print(f"Keywords: {keywords}")
        print(f"{'='*40}")
        
        start_time = time.time()
        
        if not keywords:
            print("WARNING: No keywords found for module, using fallback method")
            # Fallback to original method if no keywords
            return await self._populate_module_fallback(planned_module, clean_text, operational_context)
        
        # Extract relevant chunks using keywords
        relevant_chunks = self.chunker.extract_keyword_chunks(clean_text, keywords, chunk_size=1000)
        
        print(f"Found {len(relevant_chunks)} relevant chunks using keyword search")
        
        if not relevant_chunks:
            print("WARNING: No relevant chunks found using keywords, using fallback")
            return await self._populate_module_fallback(planned_module, clean_text, operational_context)
        
        # Combine all relevant content
        combined_content = "\n\n".join([chunk["content"] for chunk in relevant_chunks])
        content_sources = [f"Keyword '{chunk['keyword']}' at position {chunk['position']}" for chunk in relevant_chunks]
        
        print(f"Combined content length: {len(combined_content)} characters")
        print(f"Content sources: {content_sources}")
        
        # Populate the module with collected content
        populated_module = await self._populate_module_content(
            planned_module, combined_content, operational_context, module_num, total_modules
        )
        
        populated_module["content_sources"] = content_sources
        populated_module["total_chunks_analyzed"] = len(relevant_chunks)
        populated_module["relevant_chunks_found"] = len(relevant_chunks)
        populated_module["keywords_used"] = keywords
        
        elapsed_time = time.time() - start_time
        print(f"Module {module_num} populated in {elapsed_time:.2f} seconds")
        
        return populated_module
    
    async def _populate_module_fallback(self, planned_module: Dict[str, Any], clean_text: str,
                                       operational_context: str) -> Dict[str, Any]:
        """Fallback population method when keywords are not available"""
        # Use first 2000 characters as fallback
        fallback_content = clean_text[:2000]
        
        populated_module = await self._populate_module_content(
            planned_module, fallback_content, operational_context, 0, 1
        )
        
        populated_module["content_sources"] = ["Fallback method - first 2000 characters"]
        populated_module["total_chunks_analyzed"] = 1
        populated_module["relevant_chunks_found"] = 1
        populated_module["keywords_used"] = []
        
        return populated_module
    
    async def _populate_module_content(self, planned_module: Dict[str, Any], 
                                     combined_content: str, operational_context: str,
                                     module_num: int, total_modules: int) -> Dict[str, Any]:
        """Populate module content using relevant chunks"""
        
        module_title = planned_module.get("title", "Unknown")
        
        if not combined_content:
            print("WARNING: No content found for module")
            # No relevant content found
            return {
                **planned_module,
                "status": "no_content_found",
                "verbatim_content": "",
                "ste_content": "",
                "prerequisites": "",
                "tools_equipment": "",
                "warnings": "",
                "cautions": "",
                "procedural_steps": json.dumps([]),
                "expected_results": "",
                "specifications": "",
                "references": "",
                "completeness_score": 0.0
            }
        
        print(f"\nAI POPULATION REQUEST - Module {module_num}: {module_title}")
        print(f"Content length: {len(combined_content)} characters")
        
        try:
            system_prompt = f"""You are an expert in S1000D technical documentation and ASD-STE100 Simplified Technical English.

            You are populating a specific data module with content from relevant text chunks.
            
            Create a comprehensive data module with the following structure:
            {{
                "verbatim_content": "Original text content formatted for this module",
                "ste_content": "Text rewritten in ASD-STE100 Simplified Technical English",
                "prerequisites": "Prerequisites and initial conditions required",
                "tools_equipment": "Required tools, equipment, and consumables",
                "warnings": "Safety warnings and critical information",
                "cautions": "Important cautions and notes",
                "procedural_steps": [
                    {{
                        "step_number": 1,
                        "action": "Clear, actionable step description",
                        "details": "Additional details or sub-steps"
                    }}
                ],
                "expected_results": "Expected outcomes and verification steps",
                "specifications": "Technical specifications and tolerances",
                "references": "Reference materials and related documents",
                "completeness_score": 0.0-1.0,
                "status": "complete|partial|insufficient_data"
            }}
            
            IMPORTANT S1000D RULES:
            1. Use proper S1000D structure and terminology
            2. Procedures should have clear step-by-step instructions
            3. Include all safety information (warnings, cautions)
            4. STE should use controlled vocabulary, simple sentences, active voice
            5. Use operational context: {operational_context}
            6. Be thorough but only include information that belongs in this specific module
            
            STE RULES:
            - Use active voice: "Remove the plug" not "The plug should be removed"
            - Use simple sentences with one main action
            - Use approved vocabulary only
            - Use present tense for procedures
            - Use specific nouns, not pronouns
            - Maximum 25 words per sentence
            - Use parallel structure for similar actions
            """
            
            user_prompt = f"""
            PLANNED MODULE TO POPULATE:
            Title: {planned_module.get('title', '')}
            Description: {planned_module.get('description', '')}
            Type: {planned_module.get('type', '')}
            Expected sections: {planned_module.get('estimated_content_sections', [])}
            
            RELEVANT CONTENT TO USE:
            {combined_content}
            
            Please populate this module with the relevant content, ensuring completeness and S1000D compliance.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            raw_response = response.choices[0].message.content
            print(f"\nAI POPULATION RESPONSE - Module {module_num}: {module_title}")
            print(f"Raw response: {raw_response}")
            
            cleaned_response = self._clean_json_response(raw_response)
            result = json.loads(cleaned_response)
            
            # Generate DMC - fix the sequence parsing
            try:
                sequence = int(planned_module.get("module_id", "1").split("_")[-1]) if "_" in str(planned_module.get("module_id", "1")) else module_num
            except (ValueError, AttributeError):
                sequence = module_num
            
            dmc = generate_dmc(
                operational_context,
                planned_module.get("type", "description"),
                planned_module.get("info_code", "040"),
                planned_module.get("item_location", "A"),
                sequence
            )
            
            # Combine planned module with populated content
            populated_module = {
                **planned_module,
                "dmc": dmc,
                **result
            }
            
            print(f"SUCCESS: Module {module_num} populated successfully")
            return populated_module
            
        except Exception as e:
            print(f"ERROR: Module {module_num} population failed: {e}")
            return {
                **planned_module,
                "status": "error",
                "verbatim_content": combined_content,
                "ste_content": combined_content,
                "prerequisites": "",
                "tools_equipment": "",
                "warnings": "",
                "cautions": "",
                "procedural_steps": json.dumps([]),
                "expected_results": "",
                "specifications": "",
                "references": "",
                "completeness_score": 0.0,
                "error": str(e)
            }
    
    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response from AI"""
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

def chunk_text(text: str) -> List[str]:
    """Legacy function maintained for backward compatibility"""
    chunker = ChunkingStrategy()
    return chunker.create_planning_chunks(text)

def generate_dmc(context: str, type_info: str, info_code: str, item_loc: str, sequence: int) -> str:
    """Generate DMC according to S1000D standards"""
    return f"{context}-{type_info}-{info_code}-{item_loc}-{sequence:02d}"

def extract_images_from_pdf(pdf_path: str) -> List[str]:
    """Extract images from PDF and save them in a format supported by OpenAI with robust error handling"""
    from PIL import Image, ImageFile
    import io
    import struct
    
    print(f"\n{'='*60}")
    print(f"EXTRACT_IMAGES_FROM_PDF - Starting image extraction")
    print(f"PDF path: {pdf_path}")
    print(f"{'='*60}")
    
    # Enable loading of truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    images = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            print(f"PDF has {len(reader.pages)} pages")
            
            for page_num, page in enumerate(reader.pages):
                print(f"Processing page {page_num + 1}")
                
                if '/XObject' in page['/Resources']:
                    xObjects = page['/Resources']['/XObject'].get_object()
                    
                    for obj in xObjects:
                        if xObjects[obj]['/Subtype'] == '/Image':
                            try:
                                size = (xObjects[obj]['/Width'], xObjects[obj]['/Height'])
                                data = xObjects[obj].get_data()
                                
                                if xObjects[obj]['/ColorSpace'] == '/DeviceRGB':
                                    mode = "RGB"
                                else:
                                    mode = "P"
                                
                                # Create a unique filename
                                temp_dir = Path("/tmp/aquila_images")
                                temp_dir.mkdir(exist_ok=True)
                                
                                image_hash = hashlib.sha256(data).hexdigest()[:8]
                                image_path = temp_dir / f"image_{page_num}_{obj}_{image_hash}.jpg"
                                
                                try:
                                    if '/Filter' in xObjects[obj]:
                                        # Handle different compression formats
                                        if xObjects[obj]['/Filter'] == '/FlateDecode':
                                            img = Image.frombytes(mode, size, data)
                                        elif xObjects[obj]['/Filter'] == '/DCTDecode':
                                            img = Image.open(io.BytesIO(data))
                                        elif xObjects[obj]['/Filter'] == '/JPXDecode':
                                            img = Image.open(io.BytesIO(data))
                                        else:
                                            img = Image.frombytes(mode, size, data)
                                    else:
                                        img = Image.frombytes(mode, size, data)
                                    
                                    # Convert to RGB if necessary
                                    if img.mode != 'RGB':
                                        img = img.convert('RGB')
                                    
                                    # Save as JPEG
                                    img.save(image_path, 'JPEG', quality=85)
                                    images.append(str(image_path))
                                    print(f"Extracted image: {image_path}")
                                    
                                except Exception as img_error:
                                    print(f"Error processing image on page {page_num}: {img_error}")
                                    continue
                                    
                            except Exception as e:
                                print(f"Error extracting image from page {page_num}: {e}")
                                continue
                                
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []
    
    print(f"Total images extracted: {len(images)}")
    return images

# Initialize FastAPI app
app = FastAPI(title="Aquila S1000D-AI", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# API Routes
@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.get("/api")
async def api_root():
    return {"message": "Aquila S1000D-AI API"}

@app.get("/index.html")
async def serve_index():
    return FileResponse("index.html")

@app.get("/app.js")
async def serve_app_js():
    return FileResponse("app.js")

@app.get("/app.css")
async def serve_app_css():
    return FileResponse("app.css")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Project Management Routes
@app.post("/api/projects", response_model=ProjectResponse)
async def create_project(name: str = Form(...), description: str = Form("")):
    """Create a new project"""
    try:
        project = project_manager.create_project(name, description)
        return ProjectResponse(**project)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects", response_model=List[ProjectResponse])
async def get_projects():
    """Get all projects"""
    try:
        projects = project_manager.get_projects()
        return [ProjectResponse(**project) for project in projects]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/projects/{project_id}/select")
async def select_project(project_id: str):
    """Select a project as current"""
    try:
        success = project_manager.set_current_project(project_id)
        if success:
            return {"status": "success", "project_id": project_id}
        else:
            raise HTTPException(status_code=404, detail="Project not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects/current")
async def get_current_project():
    """Get current project"""
    try:
        project = project_manager.get_current_project()
        if project:
            return ProjectResponse(**project)
        else:
            return {"status": "no_project_selected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """Delete a project"""
    try:
        success = project_manager.delete_project(project_id)
        if success:
            return {"status": "success", "message": "Project deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Project not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/upload", response_model=Dict[str, Any])
async def upload_document(file: UploadFile = File(...), operational_context: str = "Water"):
    """Upload and process a PDF document with enhanced processing"""
    print(f"\n{'='*80}")
    print(f"DOCUMENT UPLOAD - Starting upload process")
    print(f"Filename: {file.filename}")
    print(f"Operational context: {operational_context}")
    print(f"{'='*80}")
    
    # Check if project is selected
    if not project_manager.get_current_project():
        raise HTTPException(status_code=400, detail="No project selected")
    
    engine = project_manager.get_current_engine()
    if not engine:
        raise HTTPException(status_code=400, detail="No project selected")
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file
    upload_dir = Path(project_manager.get_uploads_path())
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    print(f"File saved to: {file_path}")
    
    # Calculate file hash for deduplication
    file_hash = calculate_sha256(str(file_path))
    print(f"File hash: {file_hash}")
    
    # Check for duplicates
    with Session(engine) as session:
        existing_doc = session.exec(
            select(Document).where(Document.sha256 == file_hash)
        ).first()
        
        if existing_doc:
            print(f"Duplicate file detected: {existing_doc.id}")
            return {
                "status": "duplicate",
                "message": "Document already exists",
                "document_id": existing_doc.id
            }
        
        # Create document record
        document = Document(
            filename=file.filename,
            file_path=str(file_path),
            sha256=file_hash,
            operational_context=operational_context,
            status="processing"
        )
        session.add(document)
        session.commit()
        session.refresh(document)
        doc_id = document.id
    
    print(f"Document created with ID: {doc_id}")
    
    # Start asynchronous processing with enhanced system
    asyncio.create_task(process_document_enhanced(doc_id, str(file_path), operational_context))
    
    return {
        "status": "upload_successful",
        "document_id": doc_id,
        "message": "Document uploaded successfully. Enhanced processing will begin shortly."
    }

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

async def process_document_enhanced(doc_id: str, file_path: str, operational_context: str):
    """Enhanced document processing with all 4 optimization phases"""
    print(f"\n{'='*80}")
    print(f"PROCESS_DOCUMENT_ENHANCED - Starting enhanced processing")
    print(f"Document ID: {doc_id}")
    print(f"File path: {file_path}")
    print(f"Operational context: {operational_context}")
    print(f"{'='*80}")
    
    overall_start_time = time.time()
    engine = project_manager.get_current_engine()
    
    try:
        # Phase 1: Upload Complete
        await manager.broadcast({
            "type": "progress",
            "phase": "upload_complete",
            "doc_id": doc_id,
            "detail": "Document uploaded successfully",
            "processing_type": "Upload Complete",
            "current_text": "Starting enhanced processing with concurrent module population..."
        })
        
        # Phase 2: Text Extraction and Cleaning
        print(f"\n{'='*60}")
        print(f"PHASE 2: TEXT EXTRACTION AND CLEANING")
        print(f"{'='*60}")
        
        await manager.broadcast({
            "type": "progress",
            "phase": "text_extraction",
            "doc_id": doc_id,
            "detail": "Extracting and cleaning text from PDF...",
            "processing_type": "Text Extraction",
            "current_text": "Removing headers, footers, and page numbers..."
        })
        
        # Extract and clean text
        text_cleaner = TextCleaner()
        raw_text = extract_text(file_path, laparams=LAParams())
        cleaning_result = text_cleaner.clean_extracted_text(raw_text)
        clean_text = cleaning_result["clean_text"]
        
        print(f"Raw text length: {len(raw_text)} characters")
        print(f"Clean text length: {len(clean_text)} characters")
        print(f"Cleaning report: {cleaning_result['cleaning_report']}")
        
        await manager.broadcast({
            "type": "progress",
            "phase": "text_extracted",
            "doc_id": doc_id,
            "detail": f"Text cleaned successfully. {cleaning_result['cleaning_report']}",
            "processing_type": "Text Cleaning Complete",
            "current_text": f"Removed: {len(cleaning_result['removed_elements'])} non-content elements"
        })
        
        # Phase 3: Enhanced Document Planning with Keywords
        print(f"\n{'='*60}")
        print(f"PHASE 3: ENHANCED DOCUMENT PLANNING")
        print(f"{'='*60}")
        
        await manager.broadcast({
            "type": "progress",
            "phase": "planning",
            "doc_id": doc_id,
            "detail": "Analyzing document structure and extracting keywords for optimal data modules...",
            "processing_type": "Enhanced Document Planning",
            "current_text": "AI is analyzing content to plan data modules with keyword extraction..."
        })
        
        planner = EnhancedDocumentPlanner()
        planning_data = await planner.analyze_and_plan(clean_text, operational_context)
        
        # Save planning data
        with Session(engine) as session:
            plan_record = DocumentPlan(
                document_id=doc_id,
                plan_data=json.dumps(planning_data),
                planning_confidence=planning_data.get("planning_confidence", 0.0),
                total_chunks_analyzed=planning_data.get("total_planning_chunks", 0),
                status="planned"
            )
            session.add(plan_record)
            session.commit()
            plan_id = plan_record.id
        
        await manager.broadcast({
            "type": "progress",
            "phase": "planning_complete",
            "doc_id": doc_id,
            "detail": f"Enhanced planning complete. {len(planning_data.get('planned_modules', []))} modules planned with keywords",
            "processing_type": "Planning Complete",
            "current_text": f"Confidence: {planning_data.get('planning_confidence', 0.0):.2f}, Modules: {len(planning_data.get('planned_modules', []))} with keyword extraction"
        })
        
        # Phase 4: Concurrent Module Population with Keyword-Based Extraction
        print(f"\n{'='*60}")
        print(f"PHASE 4: CONCURRENT MODULE POPULATION")
        print(f"{'='*60}")
        
        await manager.broadcast({
            "type": "progress",
            "phase": "population",
            "doc_id": doc_id,
            "detail": "Populating planned modules concurrently with keyword-based content extraction...",
            "processing_type": "Concurrent Module Population",
            "current_text": "AI is using keywords to efficiently extract and populate module content..."
        })
        
        planned_modules = planning_data.get("planned_modules", [])
        populator = EnhancedContentPopulator()
        
        # Populate modules concurrently
        populated_modules = await populator.populate_modules_concurrently(
            planned_modules, clean_text, operational_context, max_concurrent=3
        )
        
        # Save populated modules to database
        with Session(engine) as session:
            for i, populated_module in enumerate(populated_modules):
                
                # Send progress update
                await manager.broadcast({
                    "type": "progress",
                    "phase": "population",
                    "doc_id": doc_id,
                    "detail": f"Saving module {i+1} of {len(populated_modules)}: {populated_module.get('title', 'Unknown')}",
                    "processing_type": "Saving Populated Modules",
                    "current_text": f"Saved: {populated_module.get('title', 'Unknown')}",
                    "progress_section": f"{i+1}/{len(populated_modules)}"
                })
                
                # Save to database - ensure all data is properly converted for SQLite
                data_module = DataModule(
                    document_id=doc_id,
                    plan_id=plan_id,
                    module_id=str(populated_module.get("module_id", f"module_{i+1}")),
                    dmc=str(populated_module.get("dmc", "")),
                    title=str(populated_module.get("title", "")),
                    info_code=str(populated_module.get("info_code", "040")),
                    item_location=str(populated_module.get("item_location", "A")),
                    sequence=i + 1,
                    verbatim_content=str(populated_module.get("verbatim_content", "")),
                    ste_content=str(populated_module.get("ste_content", "")),
                    type=str(populated_module.get("type", "description")),
                    prerequisites=str(populated_module.get("prerequisites", "")),
                    tools_equipment=str(populated_module.get("tools_equipment", "")),
                    warnings=str(populated_module.get("warnings", "")),
                    cautions=str(populated_module.get("cautions", "")),
                    procedural_steps=json.dumps(populated_module.get("procedural_steps", [])) if isinstance(populated_module.get("procedural_steps"), list) else str(populated_module.get("procedural_steps", "[]")),
                    expected_results=str(populated_module.get("expected_results", "")),
                    specifications=str(populated_module.get("specifications", "")),
                    references=str(populated_module.get("references", "")),
                    content_sources=json.dumps(populated_module.get("content_sources", [])) if isinstance(populated_module.get("content_sources"), list) else str(populated_module.get("content_sources", "[]")),
                    completeness_score=float(populated_module.get("completeness_score", 0.0)),
                    relevant_chunks_found=int(populated_module.get("relevant_chunks_found", 0)),
                    total_chunks_analyzed=int(populated_module.get("total_chunks_analyzed", 0)),
                    population_status=str(populated_module.get("status", "complete"))
                )
                
                session.add(data_module)
            
            # Update plan status
            plan_record = session.get(DocumentPlan, plan_id)
            if plan_record:
                plan_record.status = "completed"
            
            session.commit()
        
        # Phase 5: Image Processing (keep existing functionality)
        print(f"\n{'='*60}")
        print(f"PHASE 5: IMAGE PROCESSING")
        print(f"{'='*60}")
        
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
            if document:
                document.status = "completed"
                session.commit()
        
        overall_elapsed_time = time.time() - overall_start_time
        
        print(f"\n{'='*80}")
        print(f"PROCESSING COMPLETE")
        print(f"Total time: {overall_elapsed_time:.2f} seconds")
        print(f"Modules created: {len(populated_modules)}")
        print(f"Images processed: {len(images) if images else 0}")
        print(f"{'='*80}")
        
        await manager.broadcast({
            "type": "progress",
            "phase": "finished",
            "doc_id": doc_id,
            "detail": f"Enhanced processing completed in {overall_elapsed_time:.2f} seconds",
            "processing_type": "Complete",
            "current_text": f"Created {len(populated_modules)} data modules with concurrent processing and keyword-based extraction"
        })
        
    except Exception as e:
        print(f"ERROR: Processing document {doc_id} failed: {e}")
        
        # Update document status to failed
        with Session(engine) as session:
            document = session.get(Document, doc_id)
            if document:
                document.status = "failed"
                session.commit()
        
        await manager.broadcast({
            "type": "error",
            "doc_id": doc_id,
            "detail": f"Enhanced document processing failed: {str(e)}",
            "processing_type": "Error",
            "current_text": f"Processing failed: {str(e)}"
        })

@app.get("/api/documents", response_model=List[DocumentResponse])
async def get_documents():
    """Get all documents for the current project"""
    if not project_manager.get_current_project():
        raise HTTPException(status_code=400, detail="No project selected")
    
    engine = project_manager.get_current_engine()
    if not engine:
        raise HTTPException(status_code=400, detail="No project selected")
    
    try:
        with Session(engine) as session:
            documents = session.exec(select(Document)).all()
            return [DocumentResponse(
                id=doc.id,
                filename=doc.filename,
                status=doc.status,
                uploaded_at=doc.uploaded_at,
                operational_context=doc.operational_context
            ) for doc in documents]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{document_id}/plan")
async def get_document_plan(document_id: str):
    """Get the plan for a specific document"""
    if not project_manager.get_current_project():
        raise HTTPException(status_code=400, detail="No project selected")
    
    engine = project_manager.get_current_engine()
    if not engine:
        raise HTTPException(status_code=400, detail="No project selected")
    
    try:
        with Session(engine) as session:
            plan = session.exec(
                select(DocumentPlan).where(DocumentPlan.document_id == document_id)
            ).first()
            
            if not plan:
                raise HTTPException(status_code=404, detail="Document plan not found")
            
            return {
                "id": plan.id,
                "document_id": plan.document_id,
                "plan_data": json.loads(plan.plan_data),
                "planning_confidence": plan.planning_confidence,
                "total_chunks_analyzed": plan.total_chunks_analyzed,
                "status": plan.status,
                "created_at": plan.created_at.isoformat()
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data-modules", response_model=List[DataModuleResponse])
async def get_data_modules(document_id: Optional[str] = None):
    """Get all data modules, optionally filtered by document"""
    if not project_manager.get_current_project():
        raise HTTPException(status_code=400, detail="No project selected")
    
    engine = project_manager.get_current_engine()
    if not engine:
        raise HTTPException(status_code=400, detail="No project selected")
    
    try:
        with Session(engine) as session:
            query = select(DataModule)
            if document_id:
                query = query.where(DataModule.document_id == document_id)
            
            modules = session.exec(query).all()
            return [DataModuleResponse(
                id=module.id,
                dmc=module.dmc,
                title=module.title,
                verbatim_content=module.verbatim_content,
                ste_content=module.ste_content,
                type=module.type,
                prerequisites=module.prerequisites or "",
                tools_equipment=module.tools_equipment or "",
                warnings=module.warnings or "",
                cautions=module.cautions or "",
                procedural_steps=module.procedural_steps or "[]",
                expected_results=module.expected_results or "",
                specifications=module.specifications or "",
                references=module.references or ""
            ) for module in modules]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/icns", response_model=List[ICNResponse])
async def get_icns(document_id: Optional[str] = None):
    """Get all ICNs, optionally filtered by document"""
    if not project_manager.get_current_project():
        raise HTTPException(status_code=400, detail="No project selected")
    
    engine = project_manager.get_current_engine()
    if not engine:
        raise HTTPException(status_code=400, detail="No project selected")
    
    try:
        with Session(engine) as session:
            query = select(ICN)
            if document_id:
                query = query.where(ICN.document_id == document_id)
            
            icns = session.exec(query).all()
            return [ICNResponse(
                id=icn.id,
                icn=icn.icn,
                caption=icn.caption,
                objects=json.loads(icn.objects) if icn.objects else []
            ) for icn in icns]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle any incoming messages if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)