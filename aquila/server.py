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
import tiktoken

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
    """Generate caption and detect objects in image with robust error handling"""
    from PIL import Image, ImageFile
    import io
    import base64
    
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
        print(f"OpenAI raw response: {raw_response}")
        
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
        
        return result
        
    except Exception as e:
        print(f"OpenAI caption_objects error: {e}")
        
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

class DocumentPlanner:
    """AI-powered planning system for data module structure"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=openai.api_key)
        self.chunker = ChunkingStrategy()
        
    async def analyze_and_plan(self, clean_text: str, operational_context: str) -> Dict[str, Any]:
        """
        Analyze document using large chunks (2000 tokens/200 overlap) and create planning JSON
        """
        # Create planning chunks
        planning_chunks = self.chunker.create_planning_chunks(clean_text)
        
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
        
        return planning_data
    
    async def _analyze_chunk_for_planning(self, chunk: str, existing_plan: Dict[str, Any], 
                                        chunk_num: int, total_chunks: int) -> Dict[str, Any]:
        """Analyze a single chunk for planning purposes"""
        
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
                                    "chunk_source": {chunk_num}
                                }}
                            ],
                            "document_summary": "Overall summary of what this document covers",
                            "planning_confidence": 0.95,
                            "content_analysis": "Analysis of what type of content this chunk contains"
                        }}
                        
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
            cleaned_response = self._clean_json_response(raw_response)
            result = json.loads(cleaned_response)
            
            return result
            
        except Exception as e:
            print(f"Error in planning analysis: {e}")
            return {
                "planned_modules": [],
                "document_summary": f"Error analyzing chunk {chunk_num}",
                "planning_confidence": 0.0,
                "content_analysis": f"Error: {str(e)}"
            }
    
    async def _merge_chunk_plan(self, existing_plan: Dict[str, Any], chunk_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Merge planning data from multiple chunks"""
        
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

class ContentPopulator:
    """AI-powered content population for planned modules"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=openai.api_key)
        self.chunker = ChunkingStrategy()
        
    async def populate_module(self, planned_module: Dict[str, Any], clean_text: str, 
                            operational_context: str) -> Dict[str, Any]:
        """
        Populate a specific module using small chunks (400 tokens/50 overlap):
        - Searches all population chunks for relevant content
        - Ensures completeness of information
        - Maintains S1000D compliance
        """
        # Create population chunks
        population_chunks = self.chunker.create_population_chunks(clean_text)
        
        # Collect all relevant content for this module
        relevant_content = []
        content_sources = []
        
        for i, chunk in enumerate(population_chunks):
            relevance = await self._assess_chunk_relevance(chunk, planned_module)
            
            if relevance["is_relevant"]:
                relevant_content.append({
                    "chunk_index": i,
                    "content": chunk,
                    "relevance_score": relevance["relevance_score"],
                    "relevant_sections": relevance["relevant_sections"]
                })
                content_sources.append(f"Chunk {i+1}")
        
        # Sort by relevance score
        relevant_content.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Populate the module with collected content
        populated_module = await self._populate_module_content(
            planned_module, relevant_content, operational_context
        )
        
        populated_module["content_sources"] = content_sources
        populated_module["total_chunks_analyzed"] = len(population_chunks)
        populated_module["relevant_chunks_found"] = len(relevant_content)
        
        return populated_module
    
    async def _assess_chunk_relevance(self, chunk: str, planned_module: Dict[str, Any]) -> Dict[str, Any]:
        """Assess how relevant a chunk is to a planned module"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert in S1000D technical documentation. 
                        
                        Assess how relevant the given text chunk is to the planned data module.
                        
                        Return a JSON response with this structure:
                        {
                            "is_relevant": true/false,
                            "relevance_score": 0.0-1.0,
                            "relevant_sections": ["list of specific sections that are relevant"],
                            "reasoning": "explanation of why this chunk is or isn't relevant"
                        }
                        
                        Be thorough but selective - only mark as relevant if the chunk contains 
                        information that would actually belong in this specific module."""
                    },
                    {
                        "role": "user",
                        "content": f"""
                        PLANNED MODULE:
                        Title: {planned_module.get('title', '')}
                        Description: {planned_module.get('description', '')}
                        Type: {planned_module.get('type', '')}
                        Expected sections: {planned_module.get('estimated_content_sections', [])}
                        
                        TEXT CHUNK TO ASSESS:
                        {chunk}
                        
                        Is this chunk relevant to the planned module?
                        """
                    }
                ]
            )
            
            raw_response = response.choices[0].message.content
            cleaned_response = self._clean_json_response(raw_response)
            result = json.loads(cleaned_response)
            
            return result
            
        except Exception as e:
            print(f"Error assessing chunk relevance: {e}")
            return {
                "is_relevant": False,
                "relevance_score": 0.0,
                "relevant_sections": [],
                "reasoning": f"Error: {str(e)}"
            }
    
    async def _populate_module_content(self, planned_module: Dict[str, Any], 
                                     relevant_content: List[Dict[str, Any]], 
                                     operational_context: str) -> Dict[str, Any]:
        """Populate module content using relevant chunks"""
        
        # Combine all relevant content
        combined_content = "\n\n".join([item["content"] for item in relevant_content])
        
        if not combined_content:
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
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an expert in S1000D technical documentation and ASD-STE100 Simplified Technical English.

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
                    },
                    {
                        "role": "user",
                        "content": f"""
                        PLANNED MODULE TO POPULATE:
                        Title: {planned_module.get('title', '')}
                        Description: {planned_module.get('description', '')}
                        Type: {planned_module.get('type', '')}
                        Expected sections: {planned_module.get('estimated_content_sections', [])}
                        
                        RELEVANT CONTENT TO USE:
                        {combined_content}
                        
                        Please populate this module with the relevant content, ensuring completeness and S1000D compliance.
                        """
                    }
                ]
            )
            
            raw_response = response.choices[0].message.content
            cleaned_response = self._clean_json_response(raw_response)
            result = json.loads(cleaned_response)
            
            # Generate DMC
            dmc = generate_dmc(
                operational_context,
                planned_module.get("type", "description"),
                planned_module.get("info_code", "040"),
                planned_module.get("item_location", "A"),
                int(planned_module.get("module_id", "1"))
            )
            
            # Combine planned module with populated content
            populated_module = {
                **planned_module,
                "dmc": dmc,
                **result
            }
            
            return populated_module
            
        except Exception as e:
            print(f"Error populating module content: {e}")
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
    
    # Enable loading of truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
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
                                
                                if data and len(data) > 0:
                                    # Generate unique filename
                                    image_hash = hashlib.sha256(data).hexdigest()[:8]
                                    filename = f"image_{page_num}_{image_hash}.jpg"
                                    
                                    # Use project-specific upload directory
                                    upload_dir = Path(project_manager.get_uploads_path())
                                    upload_dir.mkdir(exist_ok=True)
                                    
                                    image_path = upload_dir / filename
                                    
                                    # Multiple strategies for image processing
                                    image_processed = False
                                    
                                    # Strategy 1: Direct PIL processing
                                    try:
                                        image = Image.open(io.BytesIO(data))
                                        image_processed = _process_and_save_image(image, image_path, image_hash)
                                        if image_processed:
                                            images.append(str(image_path))
                                            continue
                                    except Exception as pil_error:
                                        print(f"PIL direct processing failed for image {image_hash}: {pil_error}")
                                    
                                    # Strategy 2: Try with different modes and error handling
                                    try:
                                        # Try to detect and fix common image format issues
                                        fixed_data = _fix_image_data(data)
                                        if fixed_data:
                                            image = Image.open(io.BytesIO(fixed_data))
                                            image_processed = _process_and_save_image(image, image_path, image_hash)
                                            if image_processed:
                                                images.append(str(image_path))
                                                continue
                                    except Exception as fix_error:
                                        print(f"Fixed data processing failed for image {image_hash}: {fix_error}")
                                    
                                    # Strategy 3: Create a placeholder image with size info
                                    try:
                                        if not image_processed and size[0] > 0 and size[1] > 0:
                                            # Create a placeholder image with the correct dimensions
                                            placeholder = Image.new('RGB', size, color=(200, 200, 200))
                                            placeholder.save(image_path, 'JPEG', quality=85, optimize=True)
                                            images.append(str(image_path))
                                            print(f"Created placeholder image for {image_hash} with size {size}")
                                            continue
                                    except Exception as placeholder_error:
                                        print(f"Placeholder creation failed for image {image_hash}: {placeholder_error}")
                                    
                                    # Strategy 4: Raw data fallback (last resort)
                                    try:
                                        if not image_processed:
                                            # Try to save raw data with different extensions
                                            for ext in ['.jpg', '.png', '.bmp', '.tiff']:
                                                try:
                                                    raw_path = image_path.with_suffix(ext)
                                                    with open(raw_path, 'wb') as img_file:
                                                        img_file.write(data)
                                                    
                                                    # Test if it's a valid image
                                                    test_image = Image.open(raw_path)
                                                    test_image.verify()
                                                    
                                                    # If verification passes, convert to JPEG
                                                    test_image = Image.open(raw_path)
                                                    if _process_and_save_image(test_image, image_path, image_hash):
                                                        images.append(str(image_path))
                                                        raw_path.unlink()  # Clean up raw file
                                                        image_processed = True
                                                        break
                                                except Exception:
                                                    if raw_path.exists():
                                                        raw_path.unlink()
                                                    continue
                                    except Exception as raw_error:
                                        print(f"Raw data fallback failed for image {image_hash}: {raw_error}")
                                    
                                    # Clean up failed attempts
                                    if not image_processed and image_path.exists():
                                        image_path.unlink()
                                        
                            except Exception as e:
                                print(f"Error extracting image: {e}")
                                continue
    except Exception as e:
        print(f"Error in image extraction: {e}")
    
    return images

def _process_and_save_image(image, image_path: Path, image_hash: str) -> bool:
    """Process and save an image with proper format conversion"""
    from PIL import Image
    
    try:
        # Ensure we have a valid image
        if not image or image.size[0] == 0 or image.size[1] == 0:
            return False
        
        # Convert to RGB if necessary (for JPEG compatibility)
        if image.mode in ('RGBA', 'LA', 'P'):
            # Handle transparency properly
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'P':
                try:
                    image = image.convert('RGBA')
                except Exception:
                    image = image.convert('RGB')
            
            if image.mode in ('RGBA', 'LA'):
                try:
                    # Use alpha channel as mask if available
                    alpha = image.split()[-1]
                    rgb_image.paste(image, mask=alpha)
                except Exception:
                    # If alpha handling fails, just paste without mask
                    rgb_image.paste(image.convert('RGB'))
            else:
                rgb_image.paste(image)
            
            image = rgb_image
        elif image.mode not in ('RGB', 'L'):
            # Convert other modes to RGB
            image = image.convert('RGB')
        
        # Save as JPEG with good quality
        image.save(image_path, 'JPEG', quality=85, optimize=True)
        print(f"Successfully processed image {image_hash} (format: {image.format}, mode: {image.mode}, size: {image.size})")
        return True
        
    except Exception as e:
        print(f"Error processing image {image_hash}: {e}")
        return False

def _fix_image_data(data: bytes) -> bytes:
    """Attempt to fix common image data issues"""
    try:
        # Check for minimum data size
        if len(data) < 10:
            return None
        
        # Try to detect and fix JPEG headers
        if data[:3] == b'\xff\xd8\xff':
            # This looks like a JPEG, but might be truncated
            # Ensure it ends with JPEG end marker
            if not data.endswith(b'\xff\xd9'):
                data = data + b'\xff\xd9'
        
        # Try to detect PNG headers and fix if needed
        elif data[:8] == b'\x89PNG\r\n\x1a\n':
            # PNG header looks good, no fixing needed
            pass
        
        # Try to detect BMP headers
        elif data[:2] == b'BM':
            # BMP header detected, no fixing needed
            pass
        
        # For other formats, try to add a minimal header if missing
        else:
            # Check if this might be raw image data
            # If the data size matches expected dimensions, create a minimal header
            pass
        
        return data
        
    except Exception:
        return None

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

@app.post("/api/documents/plan")
async def plan_document_modules(doc_id: str) -> Dict[str, Any]:
    """Generate planning JSON for document using large chunks"""
    engine = project_manager.get_current_engine()
    if not engine:
        raise HTTPException(status_code=400, detail="No project selected")
    
    with Session(engine) as session:
        # Get document
        document = session.get(Document, doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if plan already exists
        existing_plan = session.exec(
            select(DocumentPlan).where(DocumentPlan.document_id == doc_id)
        ).first()
        
        if existing_plan:
            return {
                "status": "already_planned",
                "plan_id": existing_plan.id,
                "plan_data": json.loads(existing_plan.plan_data)
            }
        
        # Extract and clean text
        text_cleaner = TextCleaner()
        raw_text = extract_text(document.file_path, laparams=LAParams())
        cleaning_result = text_cleaner.clean_extracted_text(raw_text)
        clean_text = cleaning_result["clean_text"]
        
        # Create planning
        planner = DocumentPlanner()
        planning_data = await planner.analyze_and_plan(clean_text, document.operational_context)
        
        # Save planning data
        plan_record = DocumentPlan(
            document_id=doc_id,
            plan_data=json.dumps(planning_data),
            planning_confidence=planning_data.get("planning_confidence", 0.0),
            total_chunks_analyzed=planning_data.get("total_planning_chunks", 0),
            status="planned"
        )
        session.add(plan_record)
        session.commit()
        
        return {
            "status": "planned",
            "plan_id": plan_record.id,
            "plan_data": planning_data
        }

@app.post("/api/documents/populate")
async def populate_planned_modules(doc_id: str) -> Dict[str, Any]:
    """Populate planned modules with content using small chunks"""
    engine = project_manager.get_current_engine()
    if not engine:
        raise HTTPException(status_code=400, detail="No project selected")
    
    with Session(engine) as session:
        # Get document and plan
        document = session.get(Document, doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        plan = session.exec(
            select(DocumentPlan).where(DocumentPlan.document_id == doc_id)
        ).first()
        
        if not plan:
            raise HTTPException(status_code=404, detail="No plan found. Please create a plan first.")
        
        # Get clean text
        text_cleaner = TextCleaner()
        raw_text = extract_text(document.file_path, laparams=LAParams())
        cleaning_result = text_cleaner.clean_extracted_text(raw_text)
        clean_text = cleaning_result["clean_text"]
        
        # Load plan data
        planning_data = json.loads(plan.plan_data)
        planned_modules = planning_data.get("planned_modules", [])
        
        # Populate each module
        populator = ContentPopulator()
        populated_modules = []
        
        for i, planned_module in enumerate(planned_modules):
            # Send progress update
            await manager.broadcast({
                "type": "progress",
                "phase": "population",
                "doc_id": doc_id,
                "detail": f"Populating module {i+1} of {len(planned_modules)}: {planned_module.get('title', 'Unknown')}",
                "processing_type": "Module Population",
                "current_text": f"Analyzing content for: {planned_module.get('title', 'Unknown')}",
                "progress_section": f"{i+1}/{len(planned_modules)}"
            })
            
            populated_module = await populator.populate_module(
                planned_module, clean_text, document.operational_context
            )
            
            # Save to database
            data_module = DataModule(
                document_id=doc_id,
                plan_id=plan.id,
                module_id=populated_module.get("module_id", f"module_{i+1}"),
                dmc=populated_module.get("dmc", ""),
                title=populated_module.get("title", ""),
                info_code=populated_module.get("info_code", "040"),
                item_location=populated_module.get("item_location", "A"),
                sequence=i + 1,
                verbatim_content=populated_module.get("verbatim_content", ""),
                ste_content=populated_module.get("ste_content", ""),
                type=populated_module.get("type", "description"),
                prerequisites=populated_module.get("prerequisites", ""),
                tools_equipment=populated_module.get("tools_equipment", ""),
                warnings=populated_module.get("warnings", ""),
                cautions=populated_module.get("cautions", ""),
                procedural_steps=populated_module.get("procedural_steps", "[]"),
                expected_results=populated_module.get("expected_results", ""),
                specifications=populated_module.get("specifications", ""),
                references=populated_module.get("references", ""),
                content_sources=json.dumps(populated_module.get("content_sources", [])),
                completeness_score=populated_module.get("completeness_score", 0.0),
                relevant_chunks_found=populated_module.get("relevant_chunks_found", 0),
                total_chunks_analyzed=populated_module.get("total_chunks_analyzed", 0),
                population_status=populated_module.get("status", "complete")
            )
            
            session.add(data_module)
            populated_modules.append(populated_module)
        
        # Update plan status
        plan.status = "completed"
        session.commit()
        
        return {
            "status": "populated",
            "modules_created": len(populated_modules),
            "populated_modules": populated_modules
        }

@app.get("/api/documents/{doc_id}/plan")
async def get_document_plan(doc_id: str) -> Dict[str, Any]:
    """Get planning information for document"""
    engine = project_manager.get_current_engine()
    if not engine:
        raise HTTPException(status_code=400, detail="No project selected")
    
    with Session(engine) as session:
        plan = session.exec(
            select(DocumentPlan).where(DocumentPlan.document_id == doc_id)
        ).first()
        
        if not plan:
            raise HTTPException(status_code=404, detail="No plan found")
        
        return {
            "plan_id": plan.id,
            "document_id": plan.document_id,
            "plan_data": json.loads(plan.plan_data),
            "planning_confidence": plan.planning_confidence,
            "total_chunks_analyzed": plan.total_chunks_analyzed,
            "status": plan.status,
            "created_at": plan.created_at
        }

@app.post("/api/documents/upload", response_model=Dict[str, Any])
async def upload_document(file: UploadFile = File(...), operational_context: str = "Water"):
    """Upload and process a PDF document with enhanced two-phase processing"""
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
    
    # Calculate file hash for deduplication
    file_hash = calculate_sha256(str(file_path))
    
    # Check for duplicates
    with Session(engine) as session:
        existing_doc = session.exec(
            select(Document).where(Document.sha256 == file_hash)
        ).first()
        
        if existing_doc:
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
    
    # Start asynchronous processing with new two-phase system
    asyncio.create_task(process_document_enhanced(doc_id, str(file_path), operational_context))
    
    return {
        "status": "upload_successful",
        "document_id": doc_id,
        "message": "Document uploaded successfully. Processing will begin shortly."
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
    """Enhanced document processing with two-phase approach"""
    engine = project_manager.get_current_engine()
    
    try:
        # Phase 1: Upload Complete
        await manager.broadcast({
            "type": "progress",
            "phase": "upload_complete",
            "doc_id": doc_id,
            "detail": "Document uploaded successfully",
            "processing_type": "Upload Complete",
            "current_text": "Starting enhanced processing..."
        })
        
        # Phase 2: Text Extraction and Cleaning
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
        
        await manager.broadcast({
            "type": "progress",
            "phase": "text_extracted",
            "doc_id": doc_id,
            "detail": f"Text cleaned successfully. {cleaning_result['cleaning_report']}",
            "processing_type": "Text Cleaning Complete",
            "current_text": f"Removed: {len(cleaning_result['removed_elements'])} non-content elements"
        })
        
        # Phase 3: Document Planning
        await manager.broadcast({
            "type": "progress",
            "phase": "planning",
            "doc_id": doc_id,
            "detail": "Analyzing document structure for optimal data modules...",
            "processing_type": "Document Planning",
            "current_text": "AI is analyzing content to plan data modules..."
        })
        
        planner = DocumentPlanner()
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
            "detail": f"Planning complete. {len(planning_data.get('planned_modules', []))} modules planned",
            "processing_type": "Planning Complete",
            "current_text": f"Confidence: {planning_data.get('planning_confidence', 0.0):.2f}, Modules: {len(planning_data.get('planned_modules', []))}"
        })
        
        # Phase 4: Module Population
        await manager.broadcast({
            "type": "progress",
            "phase": "population",
            "doc_id": doc_id,
            "detail": "Populating planned modules with content...",
            "processing_type": "Module Population",
            "current_text": "AI is reading through text to populate each module..."
        })
        
        planned_modules = planning_data.get("planned_modules", [])
        populator = ContentPopulator()
        populated_modules = []
        
        with Session(engine) as session:
            for i, planned_module in enumerate(planned_modules):
                # Send progress update
                await manager.broadcast({
                    "type": "progress",
                    "phase": "population",
                    "doc_id": doc_id,
                    "detail": f"Populating module {i+1} of {len(planned_modules)}: {planned_module.get('title', 'Unknown')}",
                    "processing_type": "Module Population",
                    "current_text": f"Analyzing content for: {planned_module.get('title', 'Unknown')}",
                    "progress_section": f"{i+1}/{len(planned_modules)}"
                })
                
                populated_module = await populator.populate_module(
                    planned_module, clean_text, operational_context
                )
                
                # Save to database
                data_module = DataModule(
                    document_id=doc_id,
                    plan_id=plan_id,
                    module_id=populated_module.get("module_id", f"module_{i+1}"),
                    dmc=populated_module.get("dmc", ""),
                    title=populated_module.get("title", ""),
                    info_code=populated_module.get("info_code", "040"),
                    item_location=populated_module.get("item_location", "A"),
                    sequence=i + 1,
                    verbatim_content=populated_module.get("verbatim_content", ""),
                    ste_content=populated_module.get("ste_content", ""),
                    type=populated_module.get("type", "description"),
                    prerequisites=populated_module.get("prerequisites", ""),
                    tools_equipment=populated_module.get("tools_equipment", ""),
                    warnings=populated_module.get("warnings", ""),
                    cautions=populated_module.get("cautions", ""),
                    procedural_steps=populated_module.get("procedural_steps", "[]"),
                    expected_results=populated_module.get("expected_results", ""),
                    specifications=populated_module.get("specifications", ""),
                    references=populated_module.get("references", ""),
                    content_sources=json.dumps(populated_module.get("content_sources", [])),
                    completeness_score=populated_module.get("completeness_score", 0.0),
                    relevant_chunks_found=populated_module.get("relevant_chunks_found", 0),
                    total_chunks_analyzed=populated_module.get("total_chunks_analyzed", 0),
                    population_status=populated_module.get("status", "complete")
                )
                
                session.add(data_module)
                populated_modules.append(populated_module)
            
            # Update plan status
            plan_record = session.get(DocumentPlan, plan_id)
            if plan_record:
                plan_record.status = "completed"
            
            session.commit()
        
        # Phase 5: Image Processing (keep existing functionality)
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
        
        await manager.broadcast({
            "type": "progress",
            "phase": "finished",
            "doc_id": doc_id,
            "detail": "Enhanced document processing completed successfully",
            "processing_type": "Complete",
            "current_text": f"Created {len(populated_modules)} data modules with enhanced AI planning and population"
        })
        
    except Exception as e:
        print(f"Error processing document {doc_id}: {e}")
        
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