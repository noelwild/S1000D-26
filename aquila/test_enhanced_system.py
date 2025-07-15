#!/usr/bin/env python3
"""
Test script for the enhanced PDF processing system
"""

import asyncio
import json
import subprocess
import time
from pathlib import Path
import requests
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def create_test_pdf():
    """Create a comprehensive test PDF with various content types"""
    filename = "test_enhanced_manual.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Page 1: Title and Introduction
    c.drawString(50, height - 50, "MAINTENANCE MANUAL")
    c.drawString(50, height - 70, "Engine Oil Change Procedure")
    c.drawString(50, height - 90, "Document ID: TM-123-456")
    c.drawString(50, height - 110, "Page 1 of 3")
    
    # Add some header/footer content that should be cleaned
    c.drawString(50, 50, "Footer: Confidential - Property of ABC Company")
    c.drawString(500, 50, "Page 1")
    
    # Main content
    c.drawString(50, height - 150, "1. INTRODUCTION")
    c.drawString(50, height - 180, "This manual provides step-by-step instructions for performing")
    c.drawString(50, height - 200, "routine engine oil changes on Model XYZ aircraft engines.")
    c.drawString(50, height - 220, "Follow all safety procedures and use approved lubricants only.")
    
    c.drawString(50, height - 260, "2. SAFETY WARNINGS")
    c.drawString(50, height - 280, "WARNING: Engine must be shut down and cooled before beginning")
    c.drawString(50, height - 300, "CAUTION: Wear protective gloves when handling oil")
    c.drawString(50, height - 320, "NOTE: Dispose of used oil according to local regulations")
    
    c.showPage()
    
    # Page 2: Tools and Procedure
    c.drawString(50, height - 50, "MAINTENANCE MANUAL")
    c.drawString(50, height - 70, "Page 2 of 3")
    c.drawString(50, 50, "Footer: Confidential - Property of ABC Company")
    c.drawString(500, 50, "Page 2")
    
    c.drawString(50, height - 120, "3. TOOLS AND EQUIPMENT REQUIRED")
    c.drawString(50, height - 140, "- Oil drain pan (minimum 6 quarts)")
    c.drawString(50, height - 160, "- Socket wrench set")
    c.drawString(50, height - 180, "- Oil filter wrench")
    c.drawString(50, height - 200, "- Funnel")
    c.drawString(50, height - 220, "- Shop rags")
    c.drawString(50, height - 240, "- Approved engine oil (5W-30)")
    c.drawString(50, height - 260, "- New oil filter")
    
    c.drawString(50, height - 300, "4. DRAIN PROCEDURE")
    c.drawString(50, height - 320, "4.1 Position aircraft on level ground")
    c.drawString(50, height - 340, "4.2 Ensure engine is warm but not hot")
    c.drawString(50, height - 360, "4.3 Locate oil drain plug under engine cowling")
    c.drawString(50, height - 380, "4.4 Position drain pan under plug")
    c.drawString(50, height - 400, "4.5 Remove drain plug using socket wrench")
    c.drawString(50, height - 420, "4.6 Allow oil to drain completely (approximately 15 minutes)")
    
    c.showPage()
    
    # Page 3: Completion and Inspection
    c.drawString(50, height - 50, "MAINTENANCE MANUAL")
    c.drawString(50, height - 70, "Page 3 of 3")
    c.drawString(50, 50, "Footer: Confidential - Property of ABC Company")
    c.drawString(500, 50, "Page 3")
    
    c.drawString(50, height - 120, "5. FILTER REPLACEMENT")
    c.drawString(50, height - 140, "5.1 Locate oil filter adjacent to drain plug")
    c.drawString(50, height - 160, "5.2 Remove old filter using filter wrench")
    c.drawString(50, height - 180, "5.3 Clean filter mounting surface")
    c.drawString(50, height - 200, "5.4 Apply thin coat of oil to new filter gasket")
    c.drawString(50, height - 220, "5.5 Install new filter hand-tight plus 3/4 turn")
    
    c.drawString(50, height - 260, "6. REFILL PROCEDURE")
    c.drawString(50, height - 280, "6.1 Replace drain plug with new washer")
    c.drawString(50, height - 300, "6.2 Tighten drain plug to 25 ft-lbs")
    c.drawString(50, height - 320, "6.3 Add 4.5 quarts of approved oil through filler cap")
    c.drawString(50, height - 340, "6.4 Install filler cap")
    c.drawString(50, height - 360, "6.5 Start engine and check for leaks")
    c.drawString(50, height - 380, "6.6 Shut down engine and recheck oil level")
    
    c.drawString(50, height - 420, "7. FINAL INSPECTION")
    c.drawString(50, height - 440, "7.1 Verify no leaks present")
    c.drawString(50, height - 460, "7.2 Record maintenance in aircraft logbook")
    c.drawString(50, height - 480, "7.3 Dispose of used oil and filter properly")
    
    c.save()
    return filename

def test_server_connection():
    """Test if the server is running"""
    try:
        response = requests.get("http://127.0.0.1:8001/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_server():
    """Start the server in background"""
    print("Starting server...")
    process = subprocess.Popen(
        ["python", "server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/app/aquila"
    )
    
    # Wait for server to start
    for i in range(20):
        if test_server_connection():
            print("Server is running!")
            return process
        time.sleep(1)
    
    print("Server failed to start")
    return None

def create_test_project():
    """Create a test project"""
    try:
        data = "name=Enhanced Test Project&description=Testing enhanced PDF processing system"
        response = requests.post(
            "http://127.0.0.1:8001/api/projects",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=data,
            timeout=10
        )
        
        if response.status_code == 200:
            project = response.json()
            print(f"Created project: {project['name']} (ID: {project['id']})")
            
            # Select the project
            select_response = requests.post(
                f"http://127.0.0.1:8001/api/projects/{project['id']}/select",
                timeout=10
            )
            
            if select_response.status_code == 200:
                print("Project selected successfully")
                return project['id']
            else:
                print(f"Failed to select project: {select_response.status_code}")
                return None
        else:
            print(f"Failed to create project: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error creating project: {e}")
        return None

def upload_and_test_document(pdf_filename):
    """Upload document and test enhanced processing"""
    try:
        # Upload document
        with open(pdf_filename, 'rb') as f:
            files = {'file': (pdf_filename, f, 'application/pdf')}
            data = {'operational_context': 'Air'}
            
            response = requests.post(
                "http://127.0.0.1:8001/api/documents/upload",
                files=files,
                data=data,
                timeout=30
            )
        
        if response.status_code == 200:
            result = response.json()
            doc_id = result['document_id']
            print(f"Document uploaded successfully! ID: {doc_id}")
            
            # Wait for processing to complete
            print("Waiting for processing to complete...")
            time.sleep(30)  # Give it time to process
            
            # Check if plan was created
            plan_response = requests.get(
                f"http://127.0.0.1:8001/api/documents/{doc_id}/plan",
                timeout=10
            )
            
            if plan_response.status_code == 200:
                plan_data = plan_response.json()
                print("✅ Document Plan Created Successfully!")
                print(f"Planning confidence: {plan_data['planning_confidence']:.2f}")
                print(f"Modules planned: {len(plan_data['plan_data']['planned_modules'])}")
                
                # Display planned modules
                for i, module in enumerate(plan_data['plan_data']['planned_modules']):
                    print(f"  Module {i+1}: {module['title']}")
                    print(f"    Type: {module['type']}")
                    print(f"    Description: {module['description'][:100]}...")
                    print()
                
                return doc_id
            else:
                print(f"❌ No plan found: {plan_response.status_code}")
                return None
        else:
            print(f"❌ Upload failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error uploading document: {e}")
        return None

def test_data_modules(doc_id):
    """Test data modules retrieval"""
    try:
        response = requests.get(
            f"http://127.0.0.1:8001/api/data-modules?document_id={doc_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            modules = response.json()
            print(f"✅ Retrieved {len(modules)} data modules")
            
            for module in modules:
                print(f"  Module: {module['title']}")
                print(f"    DMC: {module['dmc']}")
                print(f"    Type: {module['type']}")
                print(f"    Completeness: {module.get('completeness_score', 0):.2f}")
                print(f"    Population Status: {module.get('population_status', 'unknown')}")
                print()
            
            return True
        else:
            print(f"❌ Failed to retrieve modules: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error retrieving modules: {e}")
        return False

def main():
    """Main test function"""
    print("Enhanced PDF Processing System Test")
    print("=" * 50)
    
    # Install reportlab for PDF creation
    try:
        import reportlab
    except ImportError:
        print("Installing reportlab...")
        subprocess.run(["pip", "install", "reportlab"], check=True)
    
    # Create test PDF
    print("Creating test PDF...")
    pdf_filename = create_test_pdf()
    print(f"Created: {pdf_filename}")
    
    # Start server
    server_process = start_server()
    if not server_process:
        print("❌ Failed to start server")
        return
    
    try:
        # Create test project
        project_id = create_test_project()
        if not project_id:
            print("❌ Failed to create test project")
            return
        
        # Upload and test document
        doc_id = upload_and_test_document(pdf_filename)
        if not doc_id:
            print("❌ Failed to upload and process document")
            return
        
        # Test data modules
        if test_data_modules(doc_id):
            print("✅ Enhanced PDF processing system working correctly!")
        else:
            print("❌ Data modules test failed")
    
    finally:
        # Clean up
        if server_process:
            server_process.terminate()
            server_process.wait()
        
        # Clean up test files
        try:
            Path(pdf_filename).unlink()
        except:
            pass

if __name__ == "__main__":
    main()