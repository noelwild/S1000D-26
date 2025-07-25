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
import xml.etree.ElementTree as ET
import openai
import uvicorn
import pypdf

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Field, create_engine, Session, select
from pydantic import BaseModel
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
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
PLANNER_DIR_NAME = "planner_json"

class ProjectManager:
    def __init__(self):
        self.projects_dir = PROJECTS_DIR
        self.projects_config = PROJECTS_CONFIG_FILE
        self.current_project = None
        self.current_engine = None
        self.ensure_projects_directory()
        
        # Load current project from config on startup
        self.load_current_project_from_config()
        
    def load_current_project_from_config(self):
        """Load current project from config on startup"""
        try:
            config = self.load_projects_config()
            current_project_id = config.get("current_project")
            if current_project_id:
                self.set_current_project(current_project_id)
        except Exception as e:
            print(f"Error loading current project from config: {e}")
    
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
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create project uploads directory
        uploads_dir = project_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
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

    plan_id: Optional[str] = Field(default=None, foreign_key="documentplan.id")
    module_id: Optional[str] = Field(default=None, index=True)
    content_sources: Optional[str] = Field(default="")
    completeness_score: float = 0.0
    relevant_chunks_found: int = 0
    total_chunks_analyzed: int = 0
    population_status: str = "complete"

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
    """
    Two-phase S1000D planner:

    • Phase-A: discover module skeletons (module_id only).
    • Phase-B: for each skeleton, repass the chunks and populate every JSON field
      in isolation so the model never sees more than one module at a time.
    """

    # A. skeleton discovery  (single function)
    _PASS_A = ["_plan_module_skeleton"]

    # B. all other planners, executed per module
    _PASS_B = [
        "_plan_titles",
        "_plan_types",
        "_plan_info_codes",
        "_plan_item_locations",
        "_plan_estimated_sections",
        "_plan_priorities",
        "_plan_chunk_sources",
        "_plan_keywords",
        "_plan_document_summary",
        "_plan_planning_confidence",
        "_plan_content_analysis",
    ]

    # ───────────── lifecycle ─────────────
    def __init__(self, plan_dir: Path):
        self.client   = openai.OpenAI(api_key=openai.api_key)
        self.chunker  = ChunkingStrategy()
        self.plan_dir = plan_dir
        self.plan_dir.mkdir(parents=True, exist_ok=True)

    # ───────────── public orchestrator ─────────────
    async def analyze_and_plan(         # ← signature unchanged
        self,
        full_text: str,
        operational_context: str = "",
    ) -> dict:

        # ─── PASS‑A – discover skeletons ──────────────────────────
        planning_chunks = self.chunker.create_planning_chunks(full_text)
        skeleton_state  = {"planned_modules": []}

        for idx, chunk in enumerate(planning_chunks, 1):
            partial = await self._plan_module_skeleton(
                chunk        = chunk,
                existing_plan= skeleton_state,
                state        = skeleton_state,
                chunk_num    = idx,
                total_chunks = len(planning_chunks),
            )
            self._deep_merge(skeleton_state, partial)

        # Persist each discovered skeleton immediately
        for mod in skeleton_state["planned_modules"]:
            _write_module(self.plan_dir, mod)

        # ─── PASS‑B – populate one module at a time ──────────────
        population_chunks = self.chunker.create_population_chunks(full_text)

        full_plan = skeleton_state        # needed by _plan_keywords
        for mod_path in sorted(self.plan_dir.glob("*.json")):
            per_mod_state = {
                "planned_modules": [_read_module(self.plan_dir, mod_path.stem)]
            }

            for fn in self._PASS_B:
                for c_idx, chunk in enumerate(population_chunks, 1):
                    partial = await getattr(self, fn)(
                        chunk         = chunk,
                        existing_plan = full_plan if fn == "_plan_keywords" else {},
                        state         = per_mod_state,
                        chunk_num     = c_idx,
                        total_chunks  = len(population_chunks),
                    )
                    self._deep_merge(per_mod_state, partial)

                # write after every field‑pass
                _write_module(self.plan_dir, per_mod_state["planned_modules"][0])

        # ─── Rebuild final plan from disk & aggregate confidence ─
        modules = [json.load(fp.open()) for fp in sorted(self.plan_dir.glob("*.json"))]

        # average of module‑level planning_confidence values (0 if none)
        conf_vals = [
            m.get("planning_confidence") for m in modules
            if isinstance(m.get("planning_confidence"), (int, float))
        ]
        planning_conf = round(sum(conf_vals) / len(conf_vals), 3) if conf_vals else 0.0

        return {
            "planned_modules"      : modules,
            "operational_context"  : operational_context,
            "total_planning_chunks": len(planning_chunks),
            "planning_confidence"  : planning_conf,           # ← NEW
        }

    # ───────────── per-field planners ─────────────
    # Each returns {field: value}  –or–  {"planned_modules":[{field: …}]}

    async def _plan_module_skeleton(
        self, *, chunk, existing_plan, state, chunk_num, total_chunks
    ):
        """
        Runs once per planning‑sized chunk.

        Returns:
            { "planned_modules":[ {module_id, description}, … ] }

        • If the current chunk yields no *new* modules, returns
          { "planned_modules":[] } so the merge helper can ignore it.
        """
        # -----------------------------------------------------------------
        # Collect module_ids already known (existing_plan + state so far)
        # -----------------------------------------------------------------
        known_ids = {
            m.get("module_id")
            for container in (
                existing_plan.get("planned_modules", []),
                state.get("planned_modules", []),
            )
            for m in (container if isinstance(container, list) else [])
            if m.get("module_id")
        }

        # -----------------------------------------------------------------
        # Build system prompt
        # -----------------------------------------------------------------
        system_prompt = f"""
1. ROLE & EXPERTISE DECLARATION
You are a senior S1000D information‑architect with 15 years of tri‑service
ILS experience. You routinely decompose legacy manuals into optimally sized
Issue 5.0 data‑module plans.

2. TASK OBJECTIVE (MISSION STATEMENT)
From **# Chunk {chunk_num}/{total_chunks}** identify every *new* logical
data‑module candidate **not** already listed below and return a JSON
skeleton containing exactly two keys:
    • "module_id"   – globally unique, snake_case, ≤ 60 ASCII characters
    • "description" – one concise ≤ 35‑word STE scope sentence

ALREADY IDENTIFIED MODULES (do NOT repeat):
{sorted(known_ids) if known_ids else "[ none ]"}                 

3. DETAILED CONTEXT & BACKGROUND
Guidelines (internal use only):
• One module per distinct system, procedure, description, fault‑isolation
  topic, etc.
• Reuse established naming patterns to keep DMC lineage coherent.

4. SCOPE & BOUNDARIES
IN‑SCOPE   : Creating new skeleton objects (module_id + description only).
OUT‑OF‑SCOPE: Editing existing modules; populating other fields.

5. OUTPUT FORMAT  (STRICT JSON – NO MARKDOWN)
{{
  "planned_modules":[
    {{
      "module_id":"snake_case_unique_id",
      "description":"One‑sentence STE scope statement"
    }}
  ]
}}

If no new modules are found in this chunk, return:
{{ "planned_modules":[] }}

6. STYLE & TONE GUIDELINES
• `module_id` snake_case; avoid redundant words; ensure uniqueness.
• Description follows ASD‑STE100 principles: active voice, simple present
  tense, ≤ 35 words, one sentence.

7. INTERNAL REASONING (DO NOT OUTPUT)
analyse → list candidates → merge/split sensibly → draft JSON → self‑check:
  [✓] module_id not in ALREADY IDENTIFIED list
  [✓] snake_case, ≤ 60 chars
  [✓] description ≤ 35 words
  [✓] JSON parses; no extra keys or commas

8. CONSTRAINTS & GUARD‑RAILS
• Deterministic: temperature 0, top_p 1.
• ≤ 600 tokens output.
• Absolutely no duplicate module_id values.
• No markdown fences, no commentary outside JSON.
"""

        # -----------------------------------------------------------------
        # Call the LLM
        # -----------------------------------------------------------------
        payload = await self._call_llm_generic(
            system_prompt=system_prompt,
            chunk=chunk,
            existing_plan=existing_plan,
            state=state,
            chunk_num=chunk_num,
            total_chunks=total_chunks,
        )

        if "planned_modules" not in payload:
            raise ValueError("Planner response missing 'planned_modules'")

        return {"planned_modules": payload["planned_modules"]}

    async def _plan_titles(self, **ctx):
        chunk_num    = ctx["chunk_num"]
        total_chunks = ctx["total_chunks"]

        # You run one module at a time, so grab it directly
        module      = ctx["state"]["planned_modules"][0]
        module_id   = module.get("module_id", "")
        description = module.get("description", "")

        prompt = f"""
1. ROLE & EXPERTISE DECLARATION
You are a senior S1000D titling specialist with 15 years of defence ILS authorship experience. You craft precise data-module titles that meet Issue 5.0 naming and search-retrieval standards.
Your titles MUST align with each module’s **module_id** and **description**.

2. MODULE CONTEXT  (must drive the title)
module_id   : {module_id}
description : {description}

3. TASK OBJECTIVE (MISSION STATEMENT)
Populate ONLY the “title” field for the planned module with a unique, S1000D-compliant Title-Case string ≤ 60 characters.

4. STRICT CONSISTENCY RULES
For every module:
• The object / system named in the module_id (e.g. “front fender”) **must**
  appear in the title (singular or plural acceptable).
• Any orientation or location token in the module_id (front, rear, LH, RH,
  no. 1, etc.) **must** also appear.
• The action word in the description (“installation”, “removal”, “inspection”,
  “description”, etc.) must be reflected exactly once.
• If you cannot produce a compliant title for a module, output an **empty
  object** {{}} for that module; the merge will leave its existing value.

5. DETAILED CONTEXT & BACKGROUND
INPUTS supplied in the user message:  
• # Chunk {chunk_num}/{total_chunks} – raw text slice (may include headings and step captions).  
Titling rules (internal):  
• Begin with the noun (object/system).  
• If procedural, append an en dash (–) and the action (“Removal”, “Installation”).  
• Avoid duplicating adjacent modules; include disambiguators (“LH”, “RH”) when needed.  
• Do not include DMCs, page numbers, MIL-STD codes, or punctuation beyond the dash.

6. SCOPE & BOUNDARY CONDITIONS
IN-SCOPE: Filling the “title” key within each module dict.  
OUT-OF-SCOPE: Modifying other keys, adding commentary, exceeding length cap.

7. OUTPUT FORMAT SPECIFICATION
Return strict JSON only:  
{{
  "planned_modules":[
    {{"title":"Hydraulic Pump – Removal"}}
  ]
}}
Return one object per module, preserving array order.

8. STYLE & TONE GUIDELINES
• Title Case, no trailing period.  
• ≤ 60 characters (count spaces).  
• Single en dash (–) separates noun and action for procedures; descriptions omit dash if clear (“Crew Heater System Description”).  
• Use Australian/British spelling (“Lubrication” not “Lubrication”).  
• Abbreviations allowed only if defined elsewhere in planning state.

9. REASONING INSTRUCTIONS / THINKING SCAFFOLD
Internal 5-phase loop: analyse nouns/actions → draft titles → check uniqueness/length → self-critique → finalise.  
Do **not** expose chain-of-thought.

10. CONSTRAINTS & GUARD-RAILS
• Temperature 0 (deterministic).  
• ≤ 400 tokens output.  
• JSON must parse; no markdown fences, no extra keys.  
• No classified terms or ITAR-restricted data.

11. EVALUATION & SELF-VERIFICATION CRITERIA
Silently confirm before responding:  
[✓] Each title unique within document.  
[✓] Title Case, ≤ 60 characters.  
[✓] Starts with noun, action follows dash if procedural.  
[✓] JSON parses; array length matches planning state.

12. FEW-SHOT EXAMPLES
Example A – Procedure chunk  
Input signals: “remove roadwheel arm bolts…”  
→ "Roadwheel Arm – Removal"

Example B – Description chunk  
Input signals: “overview of crew heater system…”  
→ "Crew Heater System Description"

Example C – Theory of operation chunk  
→ "Electric Drive – Theory of Operation"

13. LLM PARAMETER DIRECTIVES
Caller enforces model="gpt-4o-mini", temperature 0, top_p 1; max_tokens managed externally.

14. POST-PROCESSING HOOKS / FOLLOW-UP ACTIONS
Downstream planners depend on stable array alignment; altering order or adding keys will break the pipeline—return exactly the specified JSON.
-- End instructions --
"""
        return await self._call_llm_modules("title", prompt, **ctx)

    async def _plan_types(self, **ctx):
        chunk_num = ctx["chunk_num"]
        total_chunks = ctx["total_chunks"]
        prompt = f"""
1. ROLE & EXPERTISE DECLARATION
You are an S1000D categorisation expert with 15 years of tri-service ILS experience, routinely mapping source content to correct data-module types for Issue 5.0 technical publications.

2. TASK OBJECTIVE (MISSION)
For every module listed in the current planning state, assign exactly **one** “type” value from the authorised S1000D set.

3. DETAILED CONTEXT & BACKGROUND
INPUTS (in the user message):
• # Chunk {chunk_num}/{total_chunks}: The text segment under analysis.
Permitted S1000D Issue 5.0 data-module types for this planner phase:
  • procedure  
  • description  
  • fault_isolation  
  • theory_of_operation  
  • maintenance_planning  
  • support_equipment  
Use titles, descriptions, and chunk content cues to infer intent. Edge cases:  
• If chunk mixes narrative and steps, prioritise “procedure” unless clearly overall descriptive.  
• If equipment/tooling description solely supports maintenance, choose “support_equipment”.

4. SCOPE & BOUNDARY CONDITIONS
IN-SCOPE: Purely selecting the correct **type** token.  
OUT-OF-SCOPE: Creating new module IDs, altering other fields, inventing new categories, returning multiple types.

5. OUTPUT FORMAT SPECIFICATION
Return **strict JSON** with only the populated “type” key, preserving array index alignment:
{{
  "planned_modules":[
    {{ "type":"procedure" }}
  ]
}}
Return the same number of objects as modules already present; do not add or remove elements.

6. RELEVANCE FILTER & SAFE‑WRITE RULES
• For **each** module now in scope, first decide if the current chunk
  # {chunk_num}/{total_chunks} is *relevant* to that module.  
  – Use the module’s **module_id**, **description**, and existing field
    value(s) to judge topical match.  
• If the chunk is **not relevant**, emit an **empty object** for that module
  so the merge leaves its existing value unchanged  
  (e.g. `{{}}` or `{{ "procedure":[] }}` depending on this planner’s field).  
• If the chunk **is relevant** but offers **no improvement** over the module’s
  current value, also emit an empty object—do **not** overwrite with equal or
  lower‑quality content.  
• **Only** when the chunk is relevant **and** provides a *better* or *missing*
  value should you populate the field.

↳ *“Better” means*: clearer title, richer description, more specific type,
  additional unique keywords, more precise item_location, etc.

All other prompt instructions still apply (array length alignment, JSON only,
deterministic output, no duplicate tokens, etc.).

7. STYLE & TONE GUIDELINES
Values must be lowercase ASCII, match the enum exactly, no extra whitespace.

8. REASONING INSTRUCTIONS / THINKING SCAFFOLD
Internally follow the five-phase loop: Analyse → Compare against enum definitions → Draft assignment list → Self-critique for edge cases → Finalise.  
Do **not** expose chain-of-thought.

9. CONSTRAINTS & GUARD-RAILS
• Temperature=0 (deterministic).  
• Must not hallucinate categories outside the six allowed.  
• JSON must parse.  
• ≤ 300 tokens budget.  
• Follow Australian Defence unclassified disclosure rules—no sensitive markings added or removed.

10. EVALUATION & SELF-VERIFICATION CRITERIA
Before responding, silently verify:
[✓] Each module has exactly one enum value.  
[✓] No enum value duplicated within a single module entry.  
[✓] Array length matches planning state length.  
[✓] JSON parses without correction.

11. FEW-SHOT EXAMPLES
Example 1  
Title: “Hydraulic Pump – Removal” → **procedure**

Example 2  
Title: “Crew Heater System – Description” → **description**

Example 3  
Title: “Fire Suppression Fault Codes” → **fault_isolation**

Example 4  
Title: “Electric Drive – Theory of Operation” → **theory_of_operation**

Example 5  
Title: “Quarterly Servicing Schedule” → **maintenance_planning**

Example 6  
Title: “Jack, Vehicle Lifting – Technical Description” → **support_equipment**

12. LLM PARAMETER DIRECTIVES
Caller enforces: model="gpt-4o-mini", temperature=0, top_p=1, no penalties, max_tokens handled by caller.

13. POST-PROCESSING HOOKS / FOLLOW-UP ACTIONS
Downstream merge logic relies on index alignment; ensure ordering intact. No further actions required.
-- End instructions --
"""
        return await self._call_llm_modules("type", prompt, **ctx)


    async def _plan_info_codes(self, **ctx):
        chunk_num = ctx["chunk_num"]
        total_chunks = ctx["total_chunks"]
        prompt = f"""
1. ROLE & EXPERTISE DECLARATION
You are an S1000D information-code allocator with 15 years of defence ILS and data-module taxonomy experience. You routinely validate technical publications for Issue 5.0 compliance.

2. TASK OBJECTIVE (MISSION)
Populate exactly one numeric “info_code” value for every planned module in the current planning state.

3. DETAILED CONTEXT & BACKGROUND
Inputs per user message:  
• # Chunk {chunk_num}/{total_chunks} – current text slice (may provide additional cues).  

Authorised S1000D Issue 5.0 mapping (do **not** use any other codes):  
| type                | info_code |  
|---------------------|-----------|  
| description         | 040 |  
| procedure           | 520 |  
| fault_isolation     | 730 |  
| theory_of_operation | 710 |  
| maintenance_planning| 320 |  
| support_equipment   | 920 |  

Edge cases:  
• If the “type” field is missing for a module, infer from title/description but **never** leave info_code blank.  
• If conflicting signals occur, default to the mapping driven by **type**; otherwise prioritise explicit S1000D wording in title.

4. SCOPE & BOUNDARY CONDITIONS
IN-SCOPE: Selecting the correct three-digit info_code from the table above.  
OUT-OF-SCOPE: Creating new codes, editing other fields, or adding explanatory text.

5. OUTPUT FORMAT SPECIFICATION
Return strict JSON only:  
{{
  "planned_modules":[
    {{"info_code":"520"}}
  ]
}}
Return one object per module, preserving array order alignment.

6. RELEVANCE FILTER & SAFE‑WRITE RULES
• For **each** module now in scope, first decide if the current chunk
  # {chunk_num}/{total_chunks} is *relevant* to that module.  
  – Use the module’s **module_id**, **description**, and existing field
    value(s) to judge topical match.  
• If the chunk is **not relevant**, emit an **empty object** for that module
  so the merge leaves its existing value unchanged  
  (e.g. `{{}}` or `{{ "info_code":[] }}` depending on this planner’s field).  
• If the chunk **is relevant** but offers **no improvement** over the module’s
  current value, also emit an empty object—do **not** overwrite with equal or
  lower‑quality content.  
• **Only** when the chunk is relevant **and** provides a *better* or *missing*
  value should you populate the field.

↳ *“Better” means*: clearer title, richer description, more specific type,
  additional unique keywords, more precise item_location, etc.

All other prompt instructions still apply (array length alignment, JSON only,
deterministic output, no duplicate tokens, etc.).

7. STYLE & TONE GUIDELINES
• Value must be a three-digit ASCII string (e.g., "520").  
• No extra whitespace, no comments, no markdown.

8. REASONING INSTRUCTIONS / THINKING SCAFFOLD
Internal loop: Analyse module data → Map to table → Draft list → Self-critique for mapping accuracy & array length → Finalise.  
Do **not** expose chain-of-thought.

9. CONSTRAINTS & GUARD-RAILS
• Temperature = 0 (deterministic).  
• Must choose only from the six authorised codes.  
• ≤ 300 tokens output.  
• JSON must parse; no leading/trailing markdown fences.

10. EVALUATION & SELF-VERIFICATION CRITERIA
Silently check before responding:  
[✓] One info_code per module, matches enum.  
[✓] Array length equals planning state length.  
[✓] JSON parses.

11. FEW-SHOT EXAMPLES
Example A – type: "procedure"  → info_code "520"  
Example B – title: "Engine Cooling System – Description" (type=description) → "040"  
Example C – title: "Brake Fault Codes" (type=fault_isolation) → "730"

12. LLM PARAMETER DIRECTIVES
Caller enforces: model="gpt-4o-mini", temperature=0, top_p=1, no penalties, max_tokens governed by caller.

13. POST-PROCESSING HOOKS / FOLLOW-UP ACTIONS
Downstream merge relies on array index consistency; do not alter ordering or add keys.
-- End instructions --
"""
        return await self._call_llm_modules("info_code", prompt, **ctx)


    async def _plan_item_locations(self, **ctx):
        chunk_num = ctx["chunk_num"]
        total_chunks = ctx["total_chunks"]
        prompt = f"""
1. ROLE & EXPERTISE DECLARATION
You are a senior S1000D logistics analyst with 15 years of Australian Army and Defence aviation ILS experience. You specialise in mapping tasks and equipment to the correct “item_location” code for Issue 5.0 data modules.

2. TASK OBJECTIVE (MISSION)
Populate exactly one uppercase letter in the “item_location” field for every planned module to indicate where the task or information is performed or primarily applies.

3. DETAILED CONTEXT & BACKGROUND
INPUTS delivered in the user message:  
• # Chunk {chunk_num}/{total_chunks} – the raw text currently under review.  

Authorised **item_location codes** (use only these):  
| Code | Meaning (mnemonic)                                | Frequent cues                          |  
|------|---------------------------------------------------|----------------------------------------|  
| A    | On-equipment / in-situ (vehicle, aircraft, ship)  | “on-vehicle”, “installed”, “field”     |  
| B    | Off-equipment but organisational-level bench/shop | “bench test”, “shop check”, “remove”   |  
| C    | Intermediate workshop / specialised facility      | “intermediate”, “repair shop”, “cal”   |  
| D    | Depot / factory overhaul level                    | “overhaul”, “depot”, “factory”         |  
| E    | Test & evaluation / lab                           | “laboratory test”, “qualification”     |  
| F    | Packaging, storage, or provisioning               | “packaging”, “preservation”, “store”   |  

Edge cases & heuristics:  
• If a module is purely descriptive with no action location clues, align with the dominant location of its related procedure modules.  
• If contradictory cues, select the **highest** level (A < B < C < D < E < F) indicated.

4. SCOPE & BOUNDARY CONDITIONS
IN-SCOPE: Selecting a single code (A-F) per module.  
EXCLUDE: Free-text explanations, new keys, multiple letters, lowercase letters.

5. OUTPUT FORMAT SPECIFICATION
Return strict JSON *only*:
{{
  "planned_modules":[
    {{"item_location":"A"}}
  ]
}}
The `planned_modules` array length **must equal** the length implied by the planning state received.

6. RELEVANCE FILTER & SAFE‑WRITE RULES
• For **each** module now in scope, first decide if the current chunk
  # {chunk_num}/{total_chunks} is *relevant* to that module.  
  – Use the module’s **module_id**, **description**, and existing field
    value(s) to judge topical match.  
• If the chunk is **not relevant**, emit an **empty object** for that module
  so the merge leaves its existing value unchanged  
  (e.g. `{{}}` or `{{ "item_location":[] }}` depending on this planner’s field).  
• If the chunk **is relevant** but offers **no improvement** over the module’s
  current value, also emit an empty object—do **not** overwrite with equal or
  lower‑quality content.  
• **Only** when the chunk is relevant **and** provides a *better* or *missing*
  value should you populate the field.

↳ *“Better” means*: clearer title, richer description, more specific type,
  additional unique keywords, more precise item_location, etc.

All other prompt instructions still apply (array length alignment, JSON only,
deterministic output, no duplicate tokens, etc.).

7. STYLE & TONE GUIDELINES
Value must be a solitary uppercase ASCII letter A–F; no whitespace or comments.

8. REASONING INSTRUCTIONS / THINKING SCAFFOLD
Follow internally: Analyse cues → Map to table → Draft list → Self-critique for array alignment & code validity → Finalise.  
Do **not** expose chain-of-thought in output.

9. CONSTRAINTS & GUARD-RAILS
• Temperature 0 (deterministic).  
• Only codes A–F allowed; default to “A” if genuinely indeterminate.  
• ≤ 300 tokens output.  
• JSON must parse; no markdown fences.

10. EVALUATION & SELF-VERIFICATION CRITERIA
Before sending, silently confirm:  
[✓] One uppercase letter A–F per module.  
[✓] Array length matches planning state.  
[✓] JSON parses.

11. FEW-SHOT EXAMPLES
Example 1  
Title: “Roadwheel Arm – Removal (On-Vehicle)” → **A**

Example 2  
Title: “Fuel Pump – Bench Test” → **B**

Example 3  
Title: “Main Gearbox Overhaul Procedure” → **D**

12. LLM PARAMETER DIRECTIVES
Model: gpt-4o-mini, temperature=0, top_p=1, no frequency/presence penalties, max_tokens handled by caller.

13. POST-PROCESSING HOOKS / FOLLOW-UP ACTIONS
Downstream merge logic relies on array-index order; do not change ordering or add keys.
-- End instructions --
"""
        return await self._call_llm_modules("item_location", prompt, **ctx)


    async def _plan_estimated_sections(self, **ctx):
        chunk_num = ctx["chunk_num"]
        total_chunks = ctx["total_chunks"]
        prompt = f"""
1. ROLE & EXPERTISE DECLARATION
You are an S1000D content-structure architect with 15 years of tri-service ILS experience. You routinely outline data-module skeletons and know typical section patterns for Issue 5.0 procedure, description, fault-isolation, theory-of-operation, maintenance-planning, and support-equipment modules.

2. TASK OBJECTIVE (MISSION)
For each planned module, predict **2–6 Title-Case section headings** that will most likely appear in the fully drafted data module. Populate only the “estimated_content_sections” array.

3. DETAILED CONTEXT & BACKGROUND
INPUTS (in the user message):  
• # Chunk {chunk_num}/{total_chunks} – raw text slice.  

Typical section patterns by module **type**:  
• procedure → [Scope, Applicable Data, Tools & Consumables, Procedure, Functional Test]  
• description → [Scope, System Overview, Functional Description, Components, Applicable Data]  
• fault_isolation → [Scope, Symptom Table, Fault Tree, Test Procedure]  
• theory_of_operation → [Scope, System Overview, Operating Principle, Data Flow]  
• maintenance_planning → [Scope, Maintenance Levels, Task Interval, Resource Requirements]  
• support_equipment → [Scope, Equipment Description, Assembly / Disassembly, Operating Instructions]  
Edge cases: If chunk text explicitly references unique headings, reuse them verbatim in Title Case.

4. SCOPE & BOUNDARY CONDITIONS
IN-SCOPE: Listing high-level section headings only.  
OUT-OF-SCOPE: Detailed steps, paragraph numbers, S1000D XML tags, explanatory sentences.

5. OUTPUT FORMAT SPECIFICATION
Return strict JSON only – one object per module, array order intact:
{{
  "planned_modules":[
    {{"estimated_content_sections":["Scope","Procedure","Functional Test"]}}
  ]
}}

6. RELEVANCE FILTER & SAFE‑WRITE RULES
• For **each** module now in scope, first decide if the current chunk
  # {chunk_num}/{total_chunks} is *relevant* to that module.  
  – Use the module’s **module_id**, **description**, and existing field
    value(s) to judge topical match.  
• If the chunk is **not relevant**, emit an **empty object** for that module
  so the merge leaves its existing value unchanged  
  (e.g. `{{}}` or `{{ "estimated_content_sections":[] }}` depending on this planner’s field).  
• If the chunk **is relevant** but offers **no improvement** over the module’s
  current value, also emit an empty object—do **not** overwrite with equal or
  lower‑quality content.  
• **Only** when the chunk is relevant **and** provides a *better* or *missing*
  value should you populate the field.

↳ *“Better” means*: clearer title, richer description, more specific type,
  additional unique keywords, more precise item_location, etc.

All other prompt instructions still apply (array length alignment, JSON only,
deterministic output, no duplicate tokens, etc.).

7. STYLE & TONE GUIDELINES
• Each heading Title Case (capitalize major words).  
• Singular nouns preferred (“Tool List” not “Tools Lists”).  
• No punctuation except internal hyphens if part of formal heading (e.g., “Pre-Flight Check”).

8. REASONING INSTRUCTIONS / THINKING SCAFFOLD
Internal five-phase plan: Analyse → Map type & cues → Draft heading list → Self-critique (count & relevance) → Finalise.  
Do **not** expose chain-of-thought.

9. CONSTRAINTS & GUARD-RAILS
• 2–6 unique headings per module.  
• JSON must parse; no markdown fences.  
• Temperature 0 (deterministic).  
• ≤ 400 tokens output.  
• Avoid classified or ITAR-restricted wording unless present in source chunk.

10. EVALUATION & SELF-VERIFICATION CRITERIA
Silently confirm before sending:  
[✓] 2–6 headings, Title Case, no duplicates.  
[✓] Array length equals modules in planning state.  
[✓] JSON parses.

11. FEW-SHOT EXAMPLES
Example A – Module type “procedure”  
→ ["Scope","Applicable Data","Tools & Consumables","Procedure","Functional Test"]

Example B – Module type “description”  
→ ["Scope","System Overview","Functional Description","Applicable Data"]

Example C – Sparse chunk, type unresolved but descriptive cues  
→ ["Scope","Components","Applicable Data"]

12. LLM PARAMETER DIRECTIVES
Caller sets model="gpt-4o-mini", temperature=0, top_p=1, no penalties, max_tokens managed externally.

13. POST-PROCESSING HOOKS / FOLLOW-UP ACTIONS
Downstream merge logic relies on array-index alignment; do not change ordering or add keys.
-- End instructions --
"""
        return await self._call_llm_modules("estimated_content_sections", prompt, **ctx)


    async def _plan_priorities(self, **ctx):
        chunk_num = ctx["chunk_num"]
        total_chunks = ctx["total_chunks"]
        prompt = f"""
1. ROLE & EXPERTISE DECLARATION
You are a senior Integrated Logistics Support (ILS) analyst and S1000D author with 15 years of Australian Defence Land & Aerospace experience. You routinely assess maintenance tasks for criticality and assign priority codes for Issue 5.0 data modules.

2. TASK OBJECTIVE (MISSION)
For each planned module, set the “priority” field to **high**, **medium**, or **low** based on operational criticality, safety impact, and maintenance urgency cues found in the planning state and current chunk.

3. DETAILED CONTEXT & BACKGROUND
INPUTS supplied in the user message:  
• # Chunk {chunk_num}/{total_chunks} – raw text slice (may include risk statements, urgency directives, or scheduled intervals).  
 
Key heuristics (never reveal in output):  
• **high** – Direct flight safety / mission-critical / “immediate action” / catastrophic failure if neglected.  
• **medium** – Operational reliability or scheduled maintenance within normal service intervals; failure degrades capability but not catastrophic.  
• **low** – Cosmetic, informational, or long-interval tasks; deferment has minimal impact on mission or safety.

4. SCOPE & BOUNDARY CONDITIONS
IN-SCOPE: Selecting exactly one of high, medium, low.  
OUT-OF-SCOPE: Adding new keys, grading on numeric scales, or inserting rationale text.

5. OUTPUT FORMAT SPECIFICATION
Return strict JSON only:  
{{
  "planned_modules":[
    {{"priority":"high"}}
  ]
}}
Maintain array order alignment with planning state length.

6. RELEVANCE FILTER & SAFE‑WRITE RULES
• For **each** module now in scope, first decide if the current chunk
  # {chunk_num}/{total_chunks} is *relevant* to that module.  
  – Use the module’s **module_id**, **description**, and existing field
    value(s) to judge topical match.  
• If the chunk is **not relevant**, emit an **empty object** for that module
  so the merge leaves its existing value unchanged  
  (e.g. `{{}}` or `{{ "priority":[] }}` depending on this planner’s field).  
• If the chunk **is relevant** but offers **no improvement** over the module’s
  current value, also emit an empty object—do **not** overwrite with equal or
  lower‑quality content.  
• **Only** when the chunk is relevant **and** provides a *better* or *missing*
  value should you populate the field.

↳ *“Better” means*: clearer title, richer description, more specific type,
  additional unique keywords, more precise item_location, etc.

All other prompt instructions still apply (array length alignment, JSON only,
deterministic output, no duplicate tokens, etc.).

7. STYLE & TONE GUIDELINES
Value must be lowercase ASCII, no surrounding whitespace or punctuation.

8. REASONING INSTRUCTIONS / THINKING SCAFFOLD
Internal 5-phase loop: Analyse cues → Map to heuristics → Draft list → Self-critique for consistency & array length → Finalise.  
Do **not** expose chain-of-thought.

9. CONSTRAINTS & GUARD-RAILS
• Temperature 0 (deterministic).  
• Only the three valid tokens allowed.  
• ≤ 300 tokens output.  
• JSON must parse; no markdown fences.  
• Comply with AU Defence Unclassified disclosure requirements.

10. EVALUATION & SELF-VERIFICATION CRITERIA
Silently confirm before responding:  
[✓] One of high, medium, low per module.  
[✓] Array length equals planning state length.  
[✓] JSON parses.

11. FEW-SHOT EXAMPLES
Example A – Title: “Main Rotor Control Failure – Emergency Procedure” → **high**  
Example B – Title: “Quarterly Lubrication Schedule” → **medium**  
Example C – Title: “Cab Paint Touch-Up” → **low**

12. LLM PARAMETER DIRECTIVES
Caller enforces model="gpt-4o-mini", temperature=0, top_p=1, no penalties, max_tokens handled externally.

13. POST-PROCESSING HOOKS / FOLLOW-UP ACTIONS
Downstream merge logic relies on index order; do not reorder or add keys.
-- End instructions --
"""
        return await self._call_llm_modules("priority", prompt, **ctx)


    async def _plan_chunk_sources(self, **ctx):
        chunk_num = ctx["chunk_num"]
        total_chunks = ctx["total_chunks"]
        prompt = f"""
1. ROLE & EXPERTISE DECLARATION
You are an S1000D provenance auditor with 15 years of defence ILS experience.  
Your speciality is tracking source lineage of data-module content through multi-chunk ingestion pipelines.

2. TASK OBJECTIVE (MISSION STATEMENT)
Populate the “chunk_source” field for **every** planned module with the integer **{chunk_num}**, representing the ordinal of the text chunk that originated the content.

3. DETAILED CONTEXT & BACKGROUND
Inputs in the user message:  
• # Chunk {chunk_num}/{total_chunks} – the current text slice.  

The downstream system merges planning passes; a stable numeric provenance tag is essential for later traceability and de-duplication.

4. SCOPE & BOUNDARY CONDITIONS
IN-SCOPE: Assigning the integer {chunk_num} to “chunk_source”.  
OUT-OF-SCOPE: Altering other keys, adding commentary, changing array length, or re-sequencing modules.

5. OUTPUT FORMAT SPECIFICATION
Return **strict JSON** only, preserving module order:
{{
  "planned_modules":[
    {{"chunk_source":{chunk_num}}}
  ]
}}

6. STYLE & TONE GUIDELINES
• Value is an unquoted integer (no surrounding spaces).  
• No markdown, no comments, no trailing commas.

7. REASONING INSTRUCTIONS / THINKING SCAFFOLD
Internal loop: Analyse planning state → Replicate array length → Insert integer value → Self-critique for alignment → Finalise.  
Do **not** expose chain-of-thought.

8. CONSTRAINTS & GUARD-RAILS
• Temperature 0 (deterministic).  
• Output ≤ 200 tokens.  
• JSON must parse; no extra keys or whitespace artefacts.  
• Comply with AU Defence Unclassified disclosure rules (no classified text).

9. EVALUATION & SELF-VERIFICATION CRITERIA
Silently confirm before responding:  
[✓] Each object contains exactly one key “chunk_source”.  
[✓] Value equals {ctx['chunk_num']} for every module.  
[✓] Array length matches planning state.  
[✓] JSON parses.

10. FEW-SHOT EXAMPLES
Example – planning state has 3 modules, current chunk is 4:  
Input titles: ["Fuel Pump Removal", "Fuel Pump Install", "Fuel Pump Test"]  
Output →  
{{
  "planned_modules":[
    {{"chunk_source":4}},
    {{"chunk_source":4}},
    {{"chunk_source":4}}
  ]
}}

11. LLM PARAMETER DIRECTIVES
model="gpt-4o-mini", temperature=0, top_p=1, frequency_penalty=0, presence_penalty=0; max_tokens enforced by caller.

12. POST-PROCESSING HOOKS / FOLLOW-UP ACTIONS
Downstream merge logic depends on array-index alignment; altering order or length will break the pipeline—do not do so.
-- End instructions --
"""
        return await self._call_llm_modules("chunk_source", prompt, **ctx)


    async def _plan_keywords(self, **ctx):
        chunk_num = ctx["chunk_num"]
        total_chunks = ctx["total_chunks"]
        module_count  = len(ctx["state"].get("planned_modules", []))
        prompt = f"""
1. ROLE & EXPERTISE DECLARATION
You are a defence technical lexicographer with 15 years of experience tagging S1000D data-modules for optimal search and retrieval. You understand weapon-system nomenclature, NATO stock numbers, and ASD-STE100 vocabulary constraints.

2. TASK OBJECTIVE (MISSION STATEMENT)
For **every** planned module, extract **as many lower-case keyword or key-phrase tokens as are genuinely helpful** for search and cross-referencing that module.  
• Typical range is 5 – 10; use more if the source chunk is very rich, fewer if it is sparse.  
• Duplicates across modules are fine, but **no duplicates inside a single module**.

3. DETAILED CONTEXT & BACKGROUND  
INPUTS in the user message include:  
• # Chunk {chunk_num}/{total_chunks} – the raw technical text slice under review.  

Useful token classes: part names (“hydraulic pump”), actions (“line removal”), fault codes (“batt-fail-102”), MIL-/STAN-/AS standards, consumables (“o-ring”), tools (“torque wrench”), figure/table references, descriptive phrases (“crew heater”, “air-bleed line”).  
Tokens **must** appear in the chunk, title, or description; do **not** invent terminology.

4. SCOPE & BOUNDARY CONDITIONS  
IN-SCOPE: Terms and short phrases directly present in the chunk or accumulated planning state.  
OUT-OF-SCOPE: Stop-words, whole sentences, or free-text commentary.

5. CRITICAL RULE – RELEVANCE FIRST
Only extract keywords if the chunk content is **clearly relevant** to the module’s module_id and description.  
• If not relevant, return an empty object for that module (e.g., `{{}}`), leaving existing keywords unchanged.  
• If relevant but the chunk adds no better or new keywords, also return an empty object.  
• When relevant, extract 5–10 unique, lower-case keywords from both the chunk and the module's title/description.

6. WHAT COUNTS AS A KEYWORD
Part names, assemblies, subassemblies, actions (removal, installation, inspection), consumables, tools, torque/size notations (19mm), part numbers, figure/table refs, acronyms, standards.

7. OUTPUT FORMAT SPECIFICATION  
Return **strict JSON** only, preserving module order (exactly {module_count} objects):  
{{
  "planned_modules":[
    {{"keywords":["hydraulic pump","pump removal","o-ring","mil-std-810"]}}
  ]
}}

8. RELEVANCE FILTER & SAFE‑WRITE RULES
• For **each** module now in scope, first decide if the current chunk
  # {chunk_num}/{total_chunks} is *relevant* to that module.  
  – Use the module’s **module_id**, **description**, and existing field
    value(s) to judge topical match.  
• If the chunk is **not relevant**, emit an **empty object** for that module
  so the merge leaves its existing value unchanged  
  (e.g. `{{}}` or `{{ "keywords":[] }}` depending on this planner’s field).  
• If the chunk **is relevant** but offers **no improvement** over the module’s
  current value, also emit an empty object—do **not** overwrite with equal or
  lower‑quality content.  
• **Only** when the chunk is relevant **and** provides a *better* or *missing*
  value should you populate the field.

↳ *“Better” means*: clearer title, richer description, more specific type,
  additional unique keywords, more precise item_location, etc.

All other prompt instructions still apply (array length alignment, JSON only,
deterministic output, no duplicate tokens, etc.).

9. STYLE & TONE GUIDELINES  
• lower-case. • 1-to-4 words per token (hyphenated counts as one).  
• Keep part numbers exactly as shown (“p/n 123-456”).  
• No trailing/leading whitespace.

10. REASONING INSTRUCTIONS / THINKING SCAFFOLD  
Internal loop (not to be shown): analyse → harvest candidates → rank by relevance & uniqueness → self-check → finalise. ***Do not reveal chain-of-thought.***

11. CONSTRAINTS & GUARD-RAILS  
• Temperature 0 (deterministic).  
• No upper limit on keyword count, but every token must add retrieval value.  
• JSON must parse; no markdown fences.  
• ≤ 600 tokens total output.  
• No classified or ITAR-restricted vocabulary unless present in source.

12. EVALUATION & SELF-VERIFICATION CRITERIA (silent)  
[✓] At least 5 keywords per data module.  
[✓] No duplicates inside each module.  
[✓] All tokens appear in source or planning context.  
[✓] Array length == {module_count}.  
[✓] JSON parses.

13. FEW-SHOT EXAMPLES  
Example – planning state has 2 modules, current chunk mentions “remove hydraulic pump p/n 123-456, inspect o-ring, torque wrench”:  
{{
  "planned_modules":[
    {{"keywords":["hydraulic pump","p/n 123-456","pump removal","o-ring","torque wrench"]}},
    {{"keywords":["coolant line","leak inspection","pressure test"]}}
  ]
}}
Exactly {module_count} objects; order must match planning state.

14. LLM PARAMETER DIRECTIVES  
model="gpt-4o-mini", temperature 0, top_p 1; max_tokens set by caller.

15. POST-PROCESSING HOOKS / FOLLOW-UP ACTIONS  
Downstream merge logic depends on index alignment; altering order or adding keys will break the pipeline.
-- End instructions --
"""
        return await self._call_llm_modules("keywords", prompt, **ctx)

    async def _plan_document_summary(self, **ctx):
        chunk_num = ctx["chunk_num"]
        total_chunks = ctx["total_chunks"]
        prompt = f"""
1. ROLE & EXPERTISE DECLARATION
You are a senior S1000D technical summariser with 15 years of tri-service ILS authorship. You craft concise, high-fidelity document abstracts suitable for front-matter in Issue 5.0 publications.

2. TASK OBJECTIVE (MISSION STATEMENT)
Produce a single “document_summary” string (≤ 75 words) that captures the overarching purpose, scope, and audience of the document being planned.

3. DETAILED CONTEXT & BACKGROUND
INPUTS supplied in the user message:  
• # Chunk {chunk_num}/{total_chunks} – current text slice.  
  
The summary represents the **entire publication**, not just this chunk. Rely on accumulated planning data plus any explicit signals in the current chunk.

4. SCOPE & BOUNDARY CONDITIONS
IN-SCOPE: Overall system, operational domain, and primary maintenance/operational intent.  
OUT-OF-SCOPE: Detailed step procedures, page counts, chunk numbers, security markings, or classified content.

5. OUTPUT FORMAT SPECIFICATION
Return strict JSON only:  
{{ "document_summary":"<≤ 75-word paragraph>" }}

6. STYLE & TONE GUIDELINES
• Professional, third-person, present tense.  
• Avoid “this document” lead-ins; start with the subject (e.g., “The crew heater system manual provides…”).  
• Use ASD-STE100 controlled language; plain English, no jargon or abbreviations unless defined in planning state.  
• No lists, line breaks, or markdown.

7. REASONING INSTRUCTIONS / THINKING SCAFFOLD
Internal 5-phase loop: Analyse accumulated data → Identify key purpose & scope → Draft ≤ 75-word sentence/paragraph → Self-critique for clarity & length → Finalise.  
Do **not** expose chain-of-thought.

8. CONSTRAINTS & GUARD-RAILS
• ≤ 75 words.  
• One paragraph, no bullets.  
• Temperature 0 (deterministic).  
• JSON must parse; no extra keys.

9. EVALUATION & SELF-VERIFICATION CRITERIA
Silently ensure before responding:  
[✓] Summary addresses system, scope, intended audience/use.  
[✓] ≤ 75 words, single paragraph.  
[✓] JSON parses.

10. FEW-SHOT EXAMPLES
Example A – Data modules cover removal, installation, and test of an IFV fuel pump  
→ "The manual provides removal, installation, operational testing, and fault-isolation guidance for the infantry fighting vehicle fuel-pump assembly, enabling field and workshop personnel to restore full fuel delivery capability." (34 words)

Example B – Modules include system description, theory of operation, and maintenance planning for UAV avionics cooling  
→ "This publication describes the avionics liquid-cooling system used in the X-47B UAV, explains its heat-exchange operating principle, and details scheduled maintenance tasks to ensure reliable temperature control during unmanned flight operations." (37 words)

11. LLM PARAMETER DIRECTIVES
Caller uses model="gpt-4o-mini", temperature 0, top_p 1; max_tokens handled externally.

12. POST-PROCESSING HOOKS / FOLLOW-UP ACTIONS
Downstream pipeline stores this summary in the document header; any format deviation breaks ingestion—return exactly the specified JSON.
-- End instructions --
"""
        return await self._call_llm_top("document_summary", prompt, **ctx)


    async def _plan_planning_confidence(self, **ctx):
        chunk_num = ctx["chunk_num"]
        total_chunks = ctx["total_chunks"]
        prompt = f"""
1. ROLE & EXPERTISE DECLARATION
You are a meta-analysis assessor with 15 years of S1000D planning and QA experience. You specialise in estimating how fully a given text chunk has been converted into structured planning data.

2. TASK OBJECTIVE (MISSION STATEMENT)
Output a single numeric field “planning_confidence” between 0 and 1 that reflects how completely THIS chunk’s content has been captured by the current planning state.

3. DETAILED CONTEXT & BACKGROUND
INPUTS in the user message:  
• # Chunk {chunk_num}/{total_chunks} – the raw text slice under consideration.  

Heuristic guidance (do **not** reveal):  
• Start at 0.5 baseline.  
• +0.2 if every major topic in chunk appears as a module title or keyword.  
• +0.1 if all mandatory fields (title, type, description, info_code) are filled for each module.  
• −0.2 if obvious additional modules should exist (e.g., multiple procedures mentioned but only one module planned).  
• −0.1 if >25 % of chunk text appears uncategorised (no keywords or descriptions referencing it).  
Clamp to [0,1].

4. SCOPE & BOUNDARY CONDITIONS
IN-SCOPE: Producing a **single float** assessment value.  
OUT-OF-SCOPE: Narrative explanations, multiple keys, or updating the planning state.

5. OUTPUT FORMAT SPECIFICATION
Return strict JSON only:  
{{ "planning_confidence":0.83 }}

6. STYLE & TONE GUIDELINES
• Plain JSON with a float (no quotes).  
• 0–1 inclusive, two decimal places preferred for readability.

7. REASONING INSTRUCTIONS / THINKING SCAFFOLD
Internal 5-phase loop: analyse coverage vs chunk → apply heuristic adjustments → clamp to range → self-critique → finalise.  
**Do not expose chain-of-thought.**

8. CONSTRAINTS & GUARD-RAILS
• Temperature 0 (deterministic).  
• ≤ 100 tokens output.  
• JSON must parse; no markdown fences.  
• No values outside 0–1.

9. EVALUATION & SELF-VERIFICATION CRITERIA
Silently confirm before responding:  
[✓] Value is float 0–1, two decimals.  
[✓] No other keys present.  
[✓] JSON parses.

10. FEW-SHOT EXAMPLES
Example A – Chunk is fully planned, all fields present → 0.95  
Example B – Some missing module descriptions → 0.70  
Example C – Only skeleton modules, many gaps → 0.40

11. LLM PARAMETER DIRECTIVES
Caller sets model="gpt-4o-mini", temperature=0, top_p=1; max_tokens managed externally.

12. POST-PROCESSING HOOKS / FOLLOW-UP ACTIONS
Downstream QA dashboards plot this value; deviation from 1.0 flags the chunk for human review—hence output must be strictly numeric JSON.
-- End instructions --
"""
        return await self._call_llm_top("planning_confidence", prompt, **ctx)


    async def _plan_content_analysis(self, **ctx):
        chunk_num = ctx["chunk_num"]
        total_chunks = ctx["total_chunks"]
        prompt = f"""
1. ROLE & EXPERTISE DECLARATION
You are a defence technical-document classifier with 15 years of S1000D and ASD-STE100 authorship. You quickly identify dominant content types in raw maintenance and operational texts.

2. TASK OBJECTIVE (MISSION STATEMENT)
Assign a concise label (3–5 words) to “content_analysis” that best describes the **primary** content type of the current text chunk.

3. DETAILED CONTEXT & BACKGROUND
INPUTS (in user message):  
• # Chunk {chunk_num}/{total_chunks} – raw text slice under review.  
• JSON so far – cumulative planning state (titles, types, keywords, etc.).  
• existing_plan – optional full-document view.  
Common label archetypes (not exhaustive):  
| Label                        | Typical Signals                                       |  
|------------------------------|-------------------------------------------------------|  
| Removal Procedure            | imperative steps, “remove”, “disconnect”, torques    |  
| Installation Procedure       | “install”, “fit”, “secure”, torques                   |  
| Operational Check            | “operate”, “verify”, test points, pass/fail criteria |  
| General Description          | narrative overview, no steps, continuous prose       |  
| Fault Isolation Table        | symptom columns, fault codes, corrective action rows |  
| Theory of Operation          | functional flow, principles, data paths              |  
| Maintenance Planning Data    | intervals, man-hours, tools list, task reference IDs |  
Edge cases: pick the **most dominant** theme if multiple appear; omit secondary types.

4. SCOPE & BOUNDARY CONDITIONS
IN-SCOPE: One short phrase that fits the chunk as a whole.  
OUT-OF-SCOPE: Multiple labels, long sentences, module restructuring, or explanatory paragraphs.

5. OUTPUT FORMAT SPECIFICATION
Return strict JSON only:  
{{ "content_analysis":"Removal Procedure" }}

6. STYLE & TONE GUIDELINES
• Title Case (capitalise major words).  
• ≤ 5 words, no punctuation other than internal hyphens if required.  
• No abbreviations unless universally recognised (e.g., “FMEA Table”).

7. REASONING INSTRUCTIONS / THINKING SCAFFOLD
Internal 5-phase loop: Analyse signals → Map to archetype → Draft label → Self-critique for brevity & fit → Finalise.  
Do **not** expose chain-of-thought.

8. CONSTRAINTS & GUARD-RAILS
• Temperature 0 (deterministic).  
• ≤ 150 tokens output.  
• JSON must parse; no markdown fences.  
• Do not invent classified terms or reveal internal heuristics.

9. EVALUATION & SELF-VERIFICATION CRITERIA
Silently confirm before responding:  
[✓] Label ≤ 5 words, Title Case.  
[✓] Single string value only, key spelled exactly “content_analysis”.  
[✓] JSON parses.

10. FEW-SHOT EXAMPLES
Example A – Chunk shows numbered removal steps for hydraulic pump  
→ "Removal Procedure"  
Example B – Chunk contains symptom table with fault codes  
→ "Fault Isolation Table"  
Example C – Chunk is narrative overview of crew heater system  
→ "General Description"

11. LLM PARAMETER DIRECTIVES
Caller settings: model="gpt-4o-mini", temperature=0, top_p=1; max_tokens managed externally.

12. POST-PROCESSING HOOKS / FOLLOW-UP ACTIONS
Downstream analytics index uses this label for quick filtering—format deviations break the dashboard.
-- End instructions --
"""
        return await self._call_llm_top("content_analysis", prompt, **ctx)


    # ───────────── generic helpers (unchanged) ─────────────
    async def _call_llm_modules(
        self,
        field: str,
        system_prompt: str,
        *,                       # keyword‑only from here
        chunk,
        existing_plan,
        state,
        chunk_num,
        total_chunks,
    ) -> Dict[str, Any]:
        """
        Wrapper for planners that populate a `"planned_modules"` array.

        Improvements:
        • Injects a STRICT ARRAY LENGTH directive into the system‑prompt so
          the model knows the required array size.
        • Keeps the earlier reconcile logic as a safety‑net.
        """
        expected_len = len(state.get("planned_modules", []))

        # ── 1. augment the system prompt with a hard length instruction ────
        strict_directive = (
            "\n\n"
            "# STRICT ARRAY LENGTH DIRECTIVE\n"
            f"There are exactly {expected_len} module(s) in scope for this pass. "
            f"The `planned_modules` array in your JSON output **must** contain "
            f"exactly {expected_len} object(s) in the same order. "
            "If a module is unaffected by this chunk, output an empty object "
            "`{}` in its position. Do not add or remove array elements.\n"
        )
        augmented_prompt = system_prompt.rstrip() + strict_directive

        # ── 2. call the generic LLM helper ─────────────────────────────────
        payload = await self._call_llm_generic(
            system_prompt=augmented_prompt,
            chunk=chunk,
            existing_plan=existing_plan,
            state=state,
            chunk_num=chunk_num,
            total_chunks=total_chunks,
        )

        modules = payload.get("planned_modules")
        if modules is None or not isinstance(modules, list):
            raise ValueError(f"'planned_modules' missing or non‑list when filling '{field}'")

        # ── 3. length‑reconcile as a final guard (should rarely trigger) ───
        if len(modules) != expected_len:
            print(
                f"[Planner] WARNING: {field} pass returned {len(modules)} "
                f"entries but {expected_len} expected – auto‑reconciling."
            )
            if len(modules) > expected_len:          # too many → truncate
                modules = modules[:expected_len]
            else:                                    # too few → pad
                modules.extend({} for _ in range(expected_len - len(modules)))

        # ensure every element is a dict
        sanitised = [m if isinstance(m, dict) else {} for m in modules]

        return {"planned_modules": sanitised}

    async def _call_llm_top(
        self, key: str, system_prompt: str, **ctx
    ) -> Dict[str, Any]:
        payload = await self._call_llm_generic(system_prompt, **ctx)
        if key not in payload:
            raise ValueError(f"'{key}' missing from planner response")
        return {key: payload[key]}

    async def _call_llm_generic(
        self,
        system_prompt: str,
        *,                                   # all args after * are keyword-only
        chunk: str,
        existing_plan: Dict[str, Any],
        state: Dict[str, Any],
        chunk_num: int,
        total_chunks: int
    ) -> Dict[str, Any]:
        """
        Centralised OpenAI call used by every planner helper.  It constructs the
        user message with:
          • current chunk identifier
          • document-level summary so far
          • JSON state for *this* chunk
          • the accumulating full-document plan

        Any planner that needs those artefacts simply references them in its
        prompt (“JSON so far” etc.); no further wiring is needed.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"# Chunk {chunk_num}/{total_chunks}\n"
                            "## Document summary so far\n"
                            f"{existing_plan.get('document_summary', '')}\n\n"
                            "## JSON so far (this chunk)\n"
                            f"{json.dumps(state, indent=2, ensure_ascii=False)}\n\n"
                            "## Existing full-document plan\n"
                            f"{json.dumps(existing_plan, indent=2, ensure_ascii=False)}\n\n"
                            "## Text chunk\n"
                            f"{chunk}"
                        ),
                    },
                ],
            )
            cleaned = self._clean_json_response(response.choices[0].message.content)
            return json.loads(cleaned)
        except Exception as e:
            print(f"[Planner] AI error: {e}")
            return {}

    # deep‑merge helper for planned_modules list
    @staticmethod
    def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        """
        Recursive merge with domain‑specific rules:

        • IMMUTABLE keys (“module_id”, “description”) are never overwritten.
        • Empty dict {} from a planner == “no update” → ignored.
        • Lists of scalars are union‑merged (order preserved, duplicates removed)
          **without relying on hashing** so un‑hashable items (dict, list…) are
          handled safely.
        • Lists of dicts (planned_modules etc.) are index‑aligned then merged.
        • Scalars follow last‑writer‑wins, respecting IMMUTABLE guard.
        """
        if not src:
            return

        IMMUTABLE = {"module_id", "description"}

        for k, v in src.items():

            # 0. ignore explicit “no‑update” markers
            if isinstance(v, dict) and not v:
                continue

            # 1. planned_modules  (list of dicts, index‑aligned)
            if k == "planned_modules" and isinstance(v, list):
                if k not in dst or not isinstance(dst[k], list):
                    dst[k] = [{} for _ in v]
                while len(dst[k]) < len(v):
                    dst[k].append({})
                for i, mod in enumerate(v):
                    EnhancedDocumentPlanner._deep_merge(dst[k][i], mod)

            # 2. list of scalars  → order‑preserving *de‑dup* (hash‑safe)
            elif isinstance(v, list) and all(not isinstance(x, dict) for x in v):
                existing = dst.get(k, [])
                if not isinstance(existing, list):
                    existing = []

                combined = existing + v

                # manual de‑duplication that does NOT require hashing
                unique: list[Any] = []
                for item in combined:
                    if item not in unique:
                        unique.append(item)

                dst[k] = unique

            # 3. nested dicts  → recurse
            elif isinstance(v, dict) and isinstance(dst.get(k), dict):
                EnhancedDocumentPlanner._deep_merge(dst[k], v)

            # 4. scalars & everything else  → overwrite unless IMMUTABLE set
            else:
                if k in IMMUTABLE and dst.get(k) not in (None, "", []):
                    continue
                dst[k] = v



    @staticmethod
    def _clean_json_response(text: str) -> str:
        t = text.strip()
        if t.startswith("```json"):
            t = t[7:]
        if t.startswith("```"):
            t = t[3:]
        if t.endswith("```"):
            t = t[:-3]
        return t.strip()

# ───────────────────────────────────────────────────────────────
#  ENHANCED  CONTENT  POPULATOR
# ───────────────────────────────────────────────────────────────
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
            print("INFO: No keywords – invoking chunk‑scan fallback")
            combined_content, content_sources = await self._scan_chunks_for_module(
                planned_module, clean_text, operational_context, module_num, total_modules
            )

            if not combined_content:
                # nothing relevant found after full scan
                print("WARNING: Chunk‑scan found no relevant content – using legacy first‑2000‑chars fallback")
                return await self._populate_module_fallback(planned_module, clean_text, operational_context)

            populated_module = await self._populate_module_content(
                planned_module,
                combined_content,
                operational_context,
                module_num,
                total_modules
            )
            populated_module["content_sources"]        = content_sources
            populated_module["total_chunks_analyzed"]  = len(self.chunker.create_population_chunks(clean_text))
            populated_module["relevant_chunks_found"]  = len(content_sources)
            populated_module["keywords_used"]          = []      # none
            return populated_module
        
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

    async def _scan_chunks_for_module(
        self,
        planned_module: Dict[str, Any],
        clean_text: str,
        operational_context: str,
        module_num: int,
        total_modules: int
    ) -> tuple[str, List[str]]:
        """
        Present every population‑sized chunk to the LLM.
        If the model declares the chunk relevant it must also return the
        exact text fragment(s) that belong in this module.

        Returns
        -------
        combined_content : str
            Concatenation of all extracted fragments in document order.
        content_sources  : List[str]
            Human‑readable provenance strings (e.g. "chunk‑12 relevant").
        """
        population_chunks = self.chunker.create_population_chunks(clean_text)
        extracted: List[str] = []
        sources:   List[str] = []

        system_prompt = """
You are an S1000D data‑module population assistant.

Given:
• A target module (title, description, type etc.)
• ONE population‑size text chunk from the source manual

Decide:
1. Is any part of this chunk relevant to the module?  Answer “yes” or “no”.
2. If yes, copy **only** the sentences / paragraphs that belong in the module.

Respond in strict JSON:
{
  "relevant": true|false,
  "extracted": "text that should go into the module or empty string"
}

Rules:
• If relevant == false, extracted MUST be "".
• Never invent or paraphrase; copy verbatim.
• Ignore boiler‑plate or content clearly belonging to *other* modules.
"""

        for idx, chunk in enumerate(population_chunks, 1):
            user_prompt = (
                f"# Module\n{json.dumps(planned_module, ensure_ascii=False, indent=2)}\n\n"
                f"# Chunk {idx}/{len(population_chunks)}\n{chunk}"
            )

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                )
                payload = json.loads(self._clean_json_response(response.choices[0].message.content))
                if payload.get("relevant") and payload.get("extracted"):
                    extracted.append(payload["extracted"].strip())
                    sources.append(f"chunk‑{idx} relevant")
            except Exception as e:
                print(f"[Chunk‑scan] Module {module_num}: chunk {idx} failed – {e}")

        return "\n\n".join(extracted), sources
    
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
                "verbatim_content": "Original text content copied exactly as it appears that is relevant for this module. Preserve capitalisation, punctuation,
  units and part‑numbers. Remove leading/trailing whitespace; keep internal line breaks if they were paragraph boundaries",
                "ste_content": "If this chunk contains content that was copied verbatim into the module during the same pass, rewrite those sentences into ASD‑STE100 Simplified Technical English",
                "prerequisites": "prerequisites / initial conditions that apply to *this* module and are evident in this chunk – e.g. power isolation, fluid drained, equipment removed, environmental constraints.",
                "tools_equipment": "Required tools, test equipment, consumables or materials that the maintainer must have *before* performing the task, as evidenced in this chunk",
                "warnings": "Safety WARNINGS and critical information that belongs to this module and appears in this chunk. Exclude CAUTION and NOTE sentences.",
                "cautions": "Every CAUTION or NOTE sentence that belongs to this module and appears in the chunk. Exclude WARNING sentences.",
                "procedural_steps": [
                    {{
                        "step_number": 1,
                        "action": "Clear, actionable step description",
                        "details": "Additional details or sub-steps"
                    }}
                ],
                "expected_results": "Expected outcomes and verification steps. Concise pass/fail criteria, test outcomes or confirmations that indicate the procedure is successfully completed (e.g., “No hydraulic leaks present”, “System pressure is 2100 psi ± 50”, “Indicator lamp illuminates green”).",
                "specifications": "Technical specifications, tolerances, dimensions or settings that belong to this module and appear in the chunk. Examples: “Torque: 30 N·m ± 2”, “Pressure: 2100 psi ± 50”, “Clearance: 0.15 mm – 0.20 mm”",
                "references": "Reference materials, related documents, any standards, manuals, figure/table numbers, drawing IDs or external document references that the maintainer may need, as cited in this chunk.",
                "completeness_score": one float 0‑1 that estimates the cumulative completeness of this module after evaluating this chunk.,
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

# ───────────────────────────────────────────────────────────────
#  DATA‑MODULE  →  S1000D 5‑0 XML  EXPORTER
# ───────────────────────────────────────────────────────────────
class DataModuleXMLExporter:
    """
    Convert populated data‑module payloads into valid S1000D Issue 5‑0 XML
    files and save them under <project‑root>/XML/.

    N JSON dicts in  ➜ N .xml files out.
    """

    def __init__(
        self,
        project_root: Path,
        *,
        model: str = "gpt-4o-mini",
        max_concurrent: int = 3,
    ):
        self.project_root = Path(project_root)
        self.xml_dir      = self.project_root / "XML"
        self.xml_dir.mkdir(parents=True, exist_ok=True)

        self.client    = openai.OpenAI(api_key=openai.api_key)
        self.model     = model
        self.semaphore = asyncio.Semaphore(max_concurrent)

    # ───────────── PUBLIC API ─────────────
    async def export_modules(self, modules: List[dict]) -> List[str]:
        """Return list of written .xml file paths."""
        tasks = [
            self._export_single_module(mod, idx + 1, len(modules))
            for idx, mod in enumerate(modules)
        ]
        return await asyncio.gather(*tasks)

    # ───────────── INTERNAL ─────────────
    async def _export_single_module(
        self,
        module: dict,
        ordinal: int,
        total: int,
    ) -> str:
        async with self.semaphore:
            title = module.get("title", "Untitled")
            print(f"[XML‑EXPORT] {ordinal}/{total} – {title}")

            xml_text = await self._convert_to_xml(module)
            xml_text = self._strip_markdown_fences(xml_text)

            # sanity‑check: must parse
            try:
                ET.fromstring(xml_text)
            except ET.ParseError as err:
                raise ValueError(
                    f"Invalid XML for module “{title}”: {err}"
                ) from err

            path = self._write_file(module, xml_text)
            print(f"[XML‑EXPORT] {ordinal}/{total} – Written {path}")
            return str(path)

    async def _convert_to_xml(self, payload: dict) -> str:
        """
        Prompt GPT‑4o‑mini to map the minimal payload to Issue 5‑0 XML.
        """
        SYSTEM_PROMPT = """
You are a senior S1000D Issue 5‑0 author.  Convert the incoming JSON payload
(one fully populated data‑module) into **valid S1000D 5‑0 XML**.

RULES
• Root element <dmodule> (no external DTD/DTDID).
• Always include <identAndStatusSection> and a minimal <content>.
• Map fields:
    - dmc/module_id ➜ <dmAddress><dmCode>
    - title         ➜ <dmTitle><techName>
    - info_code     ➜ <infoCode>
    - item_location ➜ <itemLocation>
    - type          ➜ use to populate <dmRefAddrItem type="dmPurpose">
    - content       ➜ narrative <content><para>…
    - prerequisites ➜ first <para> inside content
    - tools_equipment ➜ <toolsAndConsumables>
    - warnings / cautions ➜ <warningAndCautionSection>
    - procedural_steps ➜ <procedure><step>
    - expected_results ➜ closing verification <para>
    - specifications ➜ <specPara>
    - references ➜ <references>

• Keep <step> numbers sequential starting at 1.
• Do **not** wrap output in markdown fences.
• UTF‑8, no BOM.
"""
        USER_PROMPT = json.dumps(payload, indent=2, ensure_ascii=False)

        rsp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT},
            ],
        )
        return rsp.choices[0].message.content

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```xml"):
            t = t[6:]
        if t.startswith("```"):
            t = t[3:]
        if t.endswith("```"):
            t = t[:-3]
        return t.strip()

    def _write_file(self, module: dict, xml_text: str) -> Path:
        stem = (
            module.get("dmc")
            or module.get("module_id")
            or f"module_{uuid.uuid4().hex[:8]}"
        )
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)

        unique = stem
        counter = 1
        while (self.xml_dir / f"{unique}.xml").exists():
            unique = f"{stem}_{counter}"
            counter += 1

        path = self.xml_dir / f"{unique}.xml"
        path.write_text(xml_text, encoding="utf-8")
        return path

def chunk_text(text: str) -> List[str]:
    """Legacy function maintained for backward compatibility"""
    chunker = ChunkingStrategy()
    return chunker.create_planning_chunks(text)

def generate_dmc(context: str, type_info: str, info_code: str, item_loc: str, sequence: int) -> str:
    """Generate DMC according to S1000D standards"""
    return f"{context}-{type_info}-{info_code}-{item_loc}-{sequence:02d}"

def _plan_file_path(plan_dir: Path, module_id: str) -> Path:
    """projects/<id>/planner_json/<module_id>.json"""
    return plan_dir / f"{module_id}.json"

def _read_module(plan_dir: Path, module_id: str) -> dict:
    fp = _plan_file_path(plan_dir, module_id)
    if fp.exists():
        with fp.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def _write_module(plan_dir: Path, module: dict) -> None:
    fp = _plan_file_path(plan_dir, module["module_id"])
    with fp.open("w", encoding="utf-8") as f:
        json.dump(module, f, indent=2, ensure_ascii=False)

def make_xml_payload(populated_module: dict) -> dict:
    """
    Copy **only** the fields needed by the XML exporter, substituting STE
    for verbatim text.

    Returns
    -------
    dict – minimal payload ready for DataModuleXMLExporter.
    """
    FIELD_MAP = {
        # identification
        "dmc"              : "dmc",
        "module_id"        : "module_id",     # fallback if no dmc
        "title"            : "title",
        "type"             : "type",
        "info_code"        : "info_code",
        "item_location"    : "item_location",

        # narrative & task data
        "ste_content"      : "content",       # STE is the narrative body
        "prerequisites"    : "prerequisites",
        "tools_equipment"  : "tools_equipment",
        "warnings"         : "warnings",
        "cautions"         : "cautions",
        "procedural_steps" : "procedural_steps",
        "expected_results" : "expected_results",
        "specifications"   : "specifications",
        "references"       : "references",
    }

    payload: dict = {}
    for src_key, dst_key in FIELD_MAP.items():
        val = populated_module.get(src_key)
        if val not in (None, "", [], {}):
            payload[dst_key] = val

    # guarantee at least one identifier
    if "dmc" not in payload and "module_id" not in payload:
        payload["module_id"] = f"module_{uuid.uuid4().hex[:8]}"

    return payload


def save_modules_to_json(populated_modules: list[dict], current_project_root: Path) -> None:
    """
    Write each populated data-module to its own JSON file inside
    <current_project_root>/JSON.  Collisions never over-write; a numeric
    suffix is added as needed.
    """
    json_dir = current_project_root / "JSON"
    json_dir.mkdir(exist_ok=True)

    used_names: set[str] = set()

    for mod in populated_modules:
        # Step-1: pick a base stem
        stem = (mod.get("dmc") or mod.get("module_id") or "").strip()

        # If still blank, fall back to a UUID
        if not stem:
            stem = f"module_{uuid.uuid4().hex[:8]}"

        # Remove path-unsafe characters
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)

        # Step-2: ensure uniqueness (in memory *and* on disk)
        unique_stem = stem
        counter = 1
        while unique_stem in used_names or (json_dir / f"{unique_stem}.json").exists():
            unique_stem = f"{stem}_{counter}"
            counter += 1
        used_names.add(unique_stem)

        # Step-3: write the file
        filepath = json_dir / f"{unique_stem}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(mod, f, indent=2, ensure_ascii=False)

def extract_images_from_pdf(
    pdf_path: str,
    output_dir: Path,          # ← NEW PARAM
) -> List[str]:
    """
    Extract every raster image from *pdf_path* and save it as a JPEG inside
    *output_dir* (which is created if missing).  Returns a list of file paths.
    """
    from PIL import Image, ImageFile
    import io, hashlib
    import pypdf

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    output_dir.mkdir(parents=True, exist_ok=True)

    images: List[str] = []
    with open(pdf_path, "rb") as fh:
        reader = pypdf.PdfReader(fh)

        for page_idx, page in enumerate(reader.pages, 1):
            if "/XObject" not in page["/Resources"]:
                continue

            x_objects = page["/Resources"]["/XObject"].get_object()
            for name, obj in x_objects.items():
                if obj["/Subtype"] != "/Image":
                    continue

                try:
                    data  = obj.get_data()
                    hash8 = hashlib.sha256(data).hexdigest()[:8]
                    dst   = output_dir / f"page{page_idx}_{hash8}.jpg"

                    # Some images are already JPEG streams
                    if obj.get("/Filter") == "/DCTDecode":
                        dst.write_bytes(data)
                    else:
                        mode = "RGB" if obj["/ColorSpace"] == "/DeviceRGB" else "P"
                        img  = Image.frombytes(mode, (obj["/Width"], obj["/Height"]), data)
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        img.save(dst, "JPEG", quality=85)

                    images.append(str(dst))
                except Exception as ex:
                    print(f"[PDF img] page {page_idx} – {ex}")

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

@app.get("/app_fixed.js")
async def serve_app_fixed_js():
    return FileResponse("app_fixed.js")

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
    upload_dir.mkdir(parents=True, exist_ok=True)
    
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

async def process_document_enhanced(
    doc_id: str,
    file_path: str,
    operational_context: str,
):
    """
    Enhanced document processing:
      1. Notify UI (upload complete)
      2. Extract & clean text
      3. Two‑phase planner that persists JSON skeletons incrementally
      4. Concurrent keyword‑based module population
      5. JSON → Issue 5‑0 XML conversion               ← NEW
      6. Image extraction & captioning
      7. Database + JSON / XML file output
    """
    print("\n" + "=" * 80)
    print("PROCESS_DOCUMENT_ENHANCED – Starting enhanced processing")
    print(f"Document ID        : {doc_id}")
    print(f"File path          : {file_path}")
    print(f"Operational context: {operational_context}")
    print("=" * 80)

    overall_start_time = time.time()
    engine = project_manager.get_current_engine()

    # ────────── workspace paths ──────────────────────────────────────────
    current_project_root = PROJECTS_DIR / project_manager.get_current_project()["id"]
    planner_dir          = current_project_root / PLANNER_DIR_NAME   # planner_json/
    image_dir            = current_project_root / "images"           # images/
    planner              = EnhancedDocumentPlanner(planner_dir)
    # ─────────────────────────────────────────────────────────────────────

    try:
        # ─── Phase‑1  UI ping ───────────────────────────────────────────
        await manager.broadcast({
            "type": "progress",
            "phase": "upload_complete",
            "doc_id": doc_id,
            "detail": "Document uploaded successfully",
            "processing_type": "Upload Complete",
            "current_text": "Starting enhanced processing with concurrent module population…",
        })

        # ─── Phase‑2  Text extraction & cleaning ────────────────────────
        print("\n" + "=" * 60)
        print("PHASE 2: TEXT EXTRACTION AND CLEANING")
        print("=" * 60)

        await manager.broadcast({
            "type": "progress",
            "phase": "text_extraction",
            "doc_id": doc_id,
            "detail": "Extracting and cleaning text from PDF…",
            "processing_type": "Text Extraction",
            "current_text": "Removing headers, footers, and page numbers…",
        })

        text_cleaner  = TextCleaner()
        raw_text      = extract_text(file_path, laparams=LAParams())
        cleaning_res  = text_cleaner.clean_extracted_text(raw_text)
        clean_text    = cleaning_res["clean_text"]

        print(f"Raw text length : {len(raw_text)} chars")
        print(f"Clean text length: {len(clean_text)} chars")
        print(f"Cleaning report : {cleaning_res['cleaning_report']}")

        await manager.broadcast({
            "type": "progress",
            "phase": "text_extracted",
            "doc_id": doc_id,
            "detail": f"Text cleaned successfully. {cleaning_res['cleaning_report']}",
            "processing_type": "Text Cleaning Complete",
            "current_text": f"Removed: {len(cleaning_res['removed_elements'])} non‑content elements",
        })

        # ─── Phase‑3  Two‑phase planner ────────────────────────────────
        print("\n" + "=" * 60)
        print("PHASE 3: ENHANCED DOCUMENT PLANNING")
        print("=" * 60)

        await manager.broadcast({
            "type": "progress",
            "phase": "planning",
            "doc_id": doc_id,
            "detail": "Analyzing document structure and extracting keywords…",
            "processing_type": "Enhanced Document Planning",
            "current_text": "AI is analyzing content to plan data modules with keyword extraction…",
        })

        planning_data = await planner.analyze_and_plan(clean_text, operational_context)

        # Save planning data to DB
        with Session(engine) as session:
            plan_record = DocumentPlan(
                document_id           = doc_id,
                plan_data             = json.dumps(planning_data),
                planning_confidence   = planning_data["planning_confidence"],
                total_chunks_analyzed = planning_data["total_planning_chunks"],
                status                = "planned",
            )
            session.add(plan_record)
            session.commit()
            plan_id = plan_record.id
            print(f"PLAN RECORD = {plan_record}")

        await manager.broadcast({
            "type": "progress",
            "phase": "planning_complete",
            "doc_id": doc_id,
            "detail": f"Planning complete. {len(planning_data['planned_modules'])} modules planned",
            "processing_type": "Planning Complete",
            "current_text": f"Confidence: {planning_data['planning_confidence']:.2f}",
        })

        # ─── Phase‑4  Concurrent population ────────────────────────────
        print("\n" + "=" * 60)
        print("PHASE 4: CONCURRENT MODULE POPULATION")
        print("=" * 60)

        await manager.broadcast({
            "type": "progress",
            "phase": "population",
            "doc_id": doc_id,
            "detail": "Populating planned modules concurrently…",
            "processing_type": "Concurrent Module Population",
            "current_text": "AI is extracting and populating module content…",
        })

        populator         = EnhancedContentPopulator()
        planned_modules   = planning_data["planned_modules"]
        populated_modules = await populator.populate_modules_concurrently(
            planned_modules, clean_text, operational_context, max_concurrent=3
        )

        # ─── Persist populated JSON files ───────────────────────────────
        save_modules_to_json(populated_modules, current_project_root)
        print(f"Saved {len(populated_modules)} data‑module JSON files to {(current_project_root / 'JSON')}")

        # ─── Phase‑4b  JSON → XML conversion (NEW) ─────────────────────
        print("\n" + "=" * 60)
        print("PHASE 4B: JSON → S1000D XML CONVERSION")
        print("=" * 60)

        await manager.broadcast({
            "type": "progress",
            "phase": "xml_conversion",
            "doc_id": doc_id,
            "detail": "Converting populated modules to Issue 5‑0 XML…",
            "processing_type": "XML Conversion",
            "current_text": "Generating XML data modules…",
        })

        trimmed_payloads = [make_xml_payload(m) for m in populated_modules]
        exporter = DataModuleXMLExporter(current_project_root, max_concurrent=3)
        xml_paths = await exporter.export_modules(trimmed_payloads)

        print(f"Written {len(xml_paths)} XML files to {(current_project_root / 'XML')}")

        # ─── Store populated modules in DB ──────────────────────────────
        with Session(engine) as session:
            for seq, pm in enumerate(populated_modules, 1):
                await manager.broadcast({
                    "type": "progress",
                    "phase": "population",
                    "doc_id": doc_id,
                    "detail": f"Saving module {seq}/{len(populated_modules)}: {pm.get('title','Unknown')}",
                    "processing_type": "Saving Populated Modules",
                    "current_text": pm.get("title", "Unknown"),
                    "progress_section": f"{seq}/{len(populated_modules)}",
                })

                data_module = DataModule(
                    document_id       = doc_id,
                    plan_id           = plan_id,
                    module_id         = str(pm.get("module_id", f"module_{seq}")),
                    dmc               = str(pm.get("dmc", "")),
                    title             = str(pm.get("title", "")),
                    info_code         = str(pm.get("info_code", "040")),
                    item_location     = str(pm.get("item_location", "A")),
                    sequence          = seq,
                    verbatim_content  = str(pm.get("verbatim_content", "")),
                    ste_content       = str(pm.get("ste_content", "")),
                    type              = str(pm.get("type", "description")),
                    prerequisites     = str(pm.get("prerequisites", "")),
                    tools_equipment   = str(pm.get("tools_equipment", "")),
                    warnings          = str(pm.get("warnings", "")),
                    cautions          = str(pm.get("cautions", "")),
                    procedural_steps  = json.dumps(pm.get("procedural_steps", []))
                                        if isinstance(pm.get("procedural_steps"), list)
                                        else str(pm.get("procedural_steps", "[]")),
                    expected_results  = str(pm.get("expected_results", "")),
                    specifications    = str(pm.get("specifications", "")),
                    references        = str(pm.get("references", "")),
                    content_sources   = json.dumps(pm.get("content_sources", []))
                                        if isinstance(pm.get("content_sources"), list)
                                        else str(pm.get("content_sources", "[]")),
                    completeness_score    = float(pm.get("completeness_score", 0.0)),
                    relevant_chunks_found = int(pm.get("relevant_chunks_found", 0)),
                    total_chunks_analyzed = int(pm.get("total_chunks_analyzed", 0)),
                    population_status     = str(pm.get("status", "complete")),
                )
                session.add(data_module)

            # mark plan done
            plan_record = session.get(DocumentPlan, plan_id)
            if plan_record:
                plan_record.status = "completed"
            session.commit()

        # ─── Phase‑5  Image extraction & captioning ─────────────────────
        print("\n" + "=" * 60)
        print("PHASE 5: IMAGE PROCESSING")
        print("=" * 60)

        await manager.broadcast({
            "type": "progress",
            "phase": "images_processing",
            "doc_id": doc_id,
            "detail": "Extracting images from PDF…",
            "processing_type": "Image Extraction",
            "current_text": "Scanning PDF for embedded images…",
        })

        images = extract_images_from_pdf(file_path, image_dir)

        if images:
            await manager.broadcast({
                "type": "progress",
                "phase": "images_processing",
                "doc_id": doc_id,
                "detail": f"Found {len(images)} images, generating captions…",
                "processing_type": "Image Analysis",
                "current_text": f"Processing {len(images)} images with AI vision analysis",
            })

            with Session(engine) as session:
                for idx, img_path in enumerate(images, 1):
                    try:
                        with open(img_path, "rb") as fh:
                            img_hash = hashlib.sha256(fh.read()).hexdigest()[:8]
                        icn = f"ICN-{img_hash}"

                        await manager.broadcast({
                            "type": "progress",
                            "phase": "image_analysis",
                            "doc_id": doc_id,
                            "detail": f"Analyzing image {idx}/{len(images)}",
                            "processing_type": "AI Vision Analysis",
                            "current_text": f"Generating caption for image {icn}…",
                            "progress_section": f"{idx}/{len(images)}",
                        })

                        vision   = await caption_objects(img_path)
                        best_mod = find_best_module_for_image(session, doc_id, vision["caption"])

                        if best_mod:
                            session.add(
                                ICN(
                                    document_id    = doc_id,
                                    data_module_id = best_mod.id,
                                    icn            = icn,
                                    image_path     = img_path,
                                    caption        = vision["caption"],
                                    objects        = json.dumps(vision["objects"]),
                                )
                            )

                            await manager.broadcast({
                                "type": "progress",
                                "phase": "image_analysis",
                                "doc_id": doc_id,
                                "detail": f"Caption generated for image {idx}",
                                "processing_type": "Image Caption Generated",
                                "current_text": vision["caption"][:150] + "…" if len(vision["caption"]) > 150 else vision["caption"],
                                "progress_section": f"{idx}/{len(images)}",
                            })

                    except Exception as ex:
                        print(f"[Image] {img_path} – {ex}")
                session.commit()

        # ─── wrap‑up ────────────────────────────────────────────────────
        with Session(engine) as session:
            doc = session.get(Document, doc_id)
            if doc:
                doc.status = "completed"
                session.commit()

        total_time = time.time() - overall_start_time
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print(f"Total time       : {total_time:.2f} s")
        print(f"Modules created  : {len(populated_modules)}")
        print(f"Images processed : {len(images)}")
        print("=" * 80)

        await manager.broadcast({
            "type": "progress",
            "phase": "finished",
            "doc_id": doc_id,
            "detail": f"Enhanced processing completed in {total_time:.2f} seconds",
            "processing_type": "Complete",
            "current_text": f"Created {len(populated_modules)} data modules and {len(xml_paths)} XML files",
        })

    # ───────────── Global error trap ──────────────────────────────────
    except Exception as ex:
        print(f"ERROR: Processing document {doc_id} failed – {ex}")

        with Session(engine) as session:
            doc = session.get(Document, doc_id)
            if doc:
                doc.status = "failed"
                session.commit()

        await manager.broadcast({
            "type": "error",
            "doc_id": doc_id,
            "detail": f"Enhanced document processing failed: {ex}",
            "processing_type": "Error",
            "current_text": f"Processing failed: {ex}",
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
