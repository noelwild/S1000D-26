#!/usr/bin/env python3
"""
Upload Workflow Test for Aquila S1000D-AI Application
Tests the complete upload workflow as requested in the review
"""

import requests
import json
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path

class UploadWorkflowTester:
    def __init__(self, base_url="http://127.0.0.1:8001"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.created_projects = []
        self.uploaded_documents = []

    def log(self, message, level="INFO"):
        """Log test messages with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None, form_data=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {}
        
        self.tests_run += 1
        self.log(f"🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=30)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files, data=form_data, headers=headers, timeout=30)
                elif form_data:
                    headers['Content-Type'] = 'application/x-www-form-urlencoded'
                    response = requests.post(url, data=form_data, headers=headers, timeout=30)
                else:
                    headers['Content-Type'] = 'application/json'
                    response = requests.post(url, json=data, headers=headers, timeout=30)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                self.log(f"✅ Passed - Status: {response.status_code}")
                try:
                    return success, response.json()
                except:
                    return success, response.text
            else:
                self.log(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_detail = response.json()
                    self.log(f"   Error details: {error_detail}")
                except:
                    self.log(f"   Response text: {response.text}")
                return False, {}

        except requests.exceptions.RequestException as e:
            self.log(f"❌ Failed - Network error: {str(e)}", "ERROR")
            return False, {}
        except Exception as e:
            self.log(f"❌ Failed - Error: {str(e)}", "ERROR")
            return False, {}

    def create_test_pdf(self):
        """Create a proper test PDF file"""
        # Create a more realistic PDF content
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
  /Font <<
    /F1 5 0 R
  >>
>>
>>
endobj

4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Test Document Content) Tj
ET
endstream
endobj

5 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj

xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000274 00000 n 
0000000373 00000 n 
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
456
%%EOF"""
        return pdf_content

    def test_complete_upload_workflow(self):
        """Test the complete upload workflow as requested"""
        self.log("🚀 Starting Complete Upload Workflow Test")
        self.log("=" * 60)
        
        # Step 1: Health check
        self.log("\n📋 Step 1: Health Check")
        success, health_data = self.run_test(
            "Health Check",
            "GET",
            "api/health",
            200
        )
        if not success:
            self.log("❌ Health check failed - server may not be running", "ERROR")
            return False

        # Step 2: Create a project for testing
        self.log("\n📁 Step 2: Create Test Project")
        success, project_data = self.run_test(
            "Create Test Project",
            "POST",
            "api/projects",
            200,
            form_data="name=Upload Workflow Test&description=Testing upload workflow"
        )
        
        if not success or not project_data:
            self.log("❌ Failed to create test project", "ERROR")
            return False
            
        project_id = project_data.get('id')
        self.created_projects.append(project_id)
        self.log(f"✅ Created project ID: {project_id}")

        # Step 3: Select the project (ensure project context)
        self.log("\n🎯 Step 3: Select Project for Context")
        success, _ = self.run_test(
            "Select Test Project",
            "POST",
            f"api/projects/{project_id}/select",
            200
        )
        
        if not success:
            self.log("❌ Failed to select project", "ERROR")
            return False

        # Verify project selection
        success, current_data = self.run_test(
            "Verify Project Selection",
            "GET",
            "api/projects/current",
            200
        )
        
        if success and current_data.get('id'):
            current_id = current_data['id']
            if current_id == project_id:
                self.log("✅ Project context established successfully")
            else:
                self.log("❌ Project context not established correctly")
                return False
        else:
            self.log("❌ Could not verify project selection")
            return False

        # Step 4: Test upload endpoint with proper PDF file
        self.log("\n📄 Step 4: Upload PDF Document")
        
        # Create test PDF
        pdf_content = self.create_test_pdf()
        files = {'file': ('test_maintenance_manual.pdf', pdf_content, 'application/pdf')}
        form_data = {'operational_context': 'Water'}
        
        success, upload_response = self.run_test(
            "POST /api/documents/upload - Upload PDF",
            "POST",
            "api/documents/upload",
            200,
            files=files,
            form_data=form_data
        )
        
        if not success:
            self.log("❌ Document upload failed", "ERROR")
            return False
            
        document_id = upload_response.get('id') if upload_response else None
        if document_id:
            self.uploaded_documents.append(document_id)
            self.log(f"✅ Document uploaded successfully with ID: {document_id}")
        else:
            self.log("❌ No document ID returned from upload")
            return False

        # Step 5: Verify document appears in GET /api/documents
        self.log("\n📋 Step 5: Verify Document Retrieval")
        success, documents_data = self.run_test(
            "GET /api/documents - List documents",
            "GET",
            "api/documents",
            200
        )
        
        if not success:
            self.log("❌ Failed to retrieve documents", "ERROR")
            return False
            
        # Check if our uploaded document appears in the list
        document_found = False
        if isinstance(documents_data, list):
            for doc in documents_data:
                if doc.get('id') == document_id:
                    document_found = True
                    self.log(f"✅ Document found in list: {doc.get('filename', 'Unknown')}")
                    self.log(f"   Status: {doc.get('status', 'Unknown')}")
                    self.log(f"   Operational Context: {doc.get('operational_context', 'Unknown')}")
                    break
        
        if not document_found:
            self.log("❌ Uploaded document not found in document list")
            return False

        # Step 6: Test data modules generation
        self.log("\n🔧 Step 6: Test Data Modules Generation")
        
        # Wait a moment for processing
        self.log("   Waiting for document processing...")
        time.sleep(2)
        
        success, modules_data = self.run_test(
            f"GET /api/data-modules?document_id={document_id} - Get data modules",
            "GET",
            f"api/data-modules?document_id={document_id}",
            200
        )
        
        if not success:
            self.log("❌ Failed to retrieve data modules", "ERROR")
            return False
            
        # Check if data modules were generated
        if isinstance(modules_data, list):
            if len(modules_data) > 0:
                self.log(f"✅ Data modules generated: {len(modules_data)} modules found")
                for i, module in enumerate(modules_data[:3]):  # Show first 3 modules
                    self.log(f"   Module {i+1}: {module.get('title', 'Unknown')} (DMC: {module.get('dmc', 'Unknown')})")
            else:
                self.log("⚠️ No data modules generated yet (may still be processing)")
        else:
            self.log("❌ Invalid data modules response format")
            return False

        # Step 7: Test upload without project context (should fail)
        self.log("\n🚫 Step 7: Test Upload Without Project Context")
        
        # Clear project selection by creating another project but not selecting it
        success, dummy_project = self.run_test(
            "Create Dummy Project",
            "POST",
            "api/projects",
            200,
            form_data="name=Dummy Project&description=For testing no selection"
        )
        
        if success:
            dummy_id = dummy_project.get('id')
            self.created_projects.append(dummy_id)
        
        # Try to upload without proper project context
        files = {'file': ('test_no_context.pdf', pdf_content, 'application/pdf')}
        form_data = {'operational_context': 'Air'}
        
        # This should fail because we haven't selected the dummy project
        success, _ = self.run_test(
            "Upload Without Project Context (should fail)",
            "POST",
            "api/documents/upload",
            400,  # Should fail
            files=files,
            form_data=form_data
        )
        
        if success:
            self.log("✅ Upload correctly rejected without proper project context")
        else:
            self.log("⚠️ Upload validation may need improvement")

        return True

    def cleanup(self):
        """Clean up test data"""
        self.log("\n🧹 Cleaning up test data...")
        for project_id in self.created_projects.copy():
            success, _ = self.run_test(
                f"Delete Test Project {project_id}",
                "DELETE",
                f"api/projects/{project_id}",
                200
            )
            if success:
                self.created_projects.remove(project_id)

    def run_tests(self):
        """Run the complete upload workflow test"""
        try:
            success = self.test_complete_upload_workflow()
            
            # Cleanup
            self.cleanup()
            
            # Results
            self.log("\n" + "=" * 60)
            self.log(f"📊 Upload Workflow Test Results: {self.tests_passed}/{self.tests_run} tests passed")
            
            if success and self.tests_passed == self.tests_run:
                self.log("🎉 Upload workflow test completed successfully!", "SUCCESS")
                return True
            else:
                self.log(f"⚠️ Upload workflow test had issues", "WARNING")
                return False
                
        except KeyboardInterrupt:
            print("\n\nTest interrupted by user")
            self.cleanup()
            return False
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            self.cleanup()
            return False

def main():
    """Main test execution"""
    print("Aquila S1000D-AI Upload Workflow Test")
    print("=" * 50)
    
    tester = UploadWorkflowTester()
    success = tester.run_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())