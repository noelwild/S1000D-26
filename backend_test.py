#!/usr/bin/env python3
"""
Comprehensive Backend Testing for Aquila S1000D-AI Application
Tests all API endpoints, WebSocket functionality, and database operations
"""

import requests
import json
import time
import asyncio
import websockets
import tempfile
import os
from pathlib import Path
import sqlite3
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Configuration
BACKEND_URL = "http://127.0.0.1:8001/api"
WEBSOCKET_URL = "ws://127.0.0.1:8001/ws"

class AquilaBackendTester:
    def __init__(self):
        self.session = requests.Session()
        self.test_results = {}
        self.uploaded_document_id = None
        
    def log_test(self, test_name, success, details="", error=None):
        """Log test results"""
        self.test_results[test_name] = {
            "success": success,
            "details": details,
            "error": str(error) if error else None
        }
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {details}")
        if error:
            print(f"   Error: {error}")

    def create_test_pdf(self):
        """Create a simple test PDF for upload testing"""
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            c = canvas.Canvas(temp_file.name, pagesize=letter)
            
            # Add some S1000D-like content
            c.drawString(100, 750, "TECHNICAL MANUAL")
            c.drawString(100, 720, "AIRCRAFT MAINTENANCE PROCEDURES")
            c.drawString(100, 690, "")
            c.drawString(100, 660, "1. GENERAL INFORMATION")
            c.drawString(120, 630, "This manual contains maintenance procedures for aircraft systems.")
            c.drawString(120, 600, "Follow all safety procedures when performing maintenance.")
            c.drawString(100, 570, "")
            c.drawString(100, 540, "2. ENGINE MAINTENANCE")
            c.drawString(120, 510, "2.1 Oil Change Procedure")
            c.drawString(140, 480, "- Remove oil drain plug")
            c.drawString(140, 450, "- Drain oil completely")
            c.drawString(140, 420, "- Replace oil filter")
            c.drawString(140, 390, "- Refill with specified oil")
            c.drawString(100, 360, "")
            c.drawString(100, 330, "3. HYDRAULIC SYSTEM")
            c.drawString(120, 300, "Check hydraulic fluid levels regularly.")
            c.drawString(120, 270, "Inspect hydraulic lines for leaks.")
            
            c.showPage()
            c.save()
            
            return temp_file.name
        except Exception as e:
            self.log_test("PDF Creation", False, error=e)
            return None

    def test_health_endpoint(self):
        """Test the health check endpoint"""
        try:
            response = self.session.get(f"{BACKEND_URL}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy" and "Aquila" in data.get("service", ""):
                    self.log_test("Health Check", True, f"Service: {data.get('service')}")
                    return True
                else:
                    self.log_test("Health Check", False, f"Unexpected response: {data}")
                    return False
            else:
                self.log_test("Health Check", False, f"Status code: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Health Check", False, error=e)
            return False

    def test_document_upload(self):
        """Test PDF document upload"""
        try:
            pdf_path = self.create_test_pdf()
            if not pdf_path:
                return False
                
            with open(pdf_path, 'rb') as f:
                files = {'file': ('test_manual.pdf', f, 'application/pdf')}
                data = {'operational_context': 'Air'}
                
                response = self.session.post(
                    f"{BACKEND_URL}/documents/upload",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            # Clean up temp file
            os.unlink(pdf_path)
            
            if response.status_code == 200:
                result = response.json()
                if "document_id" in result and result.get("status") == "processing":
                    self.uploaded_document_id = result["document_id"]
                    self.log_test("Document Upload", True, f"Document ID: {self.uploaded_document_id}")
                    return True
                else:
                    self.log_test("Document Upload", False, f"Unexpected response: {result}")
                    return False
            else:
                self.log_test("Document Upload", False, f"Status code: {response.status_code}, Response: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Document Upload", False, error=e)
            return False

    def test_get_documents(self):
        """Test retrieving all documents"""
        try:
            response = self.session.get(f"{BACKEND_URL}/documents", timeout=10)
            
            if response.status_code == 200:
                documents = response.json()
                if isinstance(documents, list):
                    if len(documents) > 0:
                        doc = documents[0]
                        required_fields = ['id', 'filename', 'status', 'uploaded_at', 'operational_context']
                        if all(field in doc for field in required_fields):
                            self.log_test("Get Documents", True, f"Found {len(documents)} documents")
                            return True
                        else:
                            missing = [f for f in required_fields if f not in doc]
                            self.log_test("Get Documents", False, f"Missing fields: {missing}")
                            return False
                    else:
                        self.log_test("Get Documents", True, "No documents found (empty list)")
                        return True
                else:
                    self.log_test("Get Documents", False, f"Expected list, got: {type(documents)}")
                    return False
            else:
                self.log_test("Get Documents", False, f"Status code: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Get Documents", False, error=e)
            return False

    def test_get_data_modules(self):
        """Test retrieving data modules"""
        try:
            # Test without filter
            response = self.session.get(f"{BACKEND_URL}/data-modules", timeout=10)
            
            if response.status_code == 200:
                modules = response.json()
                if isinstance(modules, list):
                    self.log_test("Get Data Modules (All)", True, f"Found {len(modules)} modules")
                    
                    # Test with document filter if we have an uploaded document
                    if self.uploaded_document_id:
                        response_filtered = self.session.get(
                            f"{BACKEND_URL}/data-modules?document_id={self.uploaded_document_id}",
                            timeout=10
                        )
                        if response_filtered.status_code == 200:
                            filtered_modules = response_filtered.json()
                            self.log_test("Get Data Modules (Filtered)", True, 
                                        f"Found {len(filtered_modules)} modules for document")
                        else:
                            self.log_test("Get Data Modules (Filtered)", False, 
                                        f"Status code: {response_filtered.status_code}")
                    
                    return True
                else:
                    self.log_test("Get Data Modules", False, f"Expected list, got: {type(modules)}")
                    return False
            else:
                self.log_test("Get Data Modules", False, f"Status code: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Get Data Modules", False, error=e)
            return False

    def test_get_icns(self):
        """Test retrieving ICNs (Illustration Control Numbers)"""
        try:
            # Test without filter
            response = self.session.get(f"{BACKEND_URL}/icns", timeout=10)
            
            if response.status_code == 200:
                icns = response.json()
                if isinstance(icns, list):
                    self.log_test("Get ICNs (All)", True, f"Found {len(icns)} ICNs")
                    
                    # Test with document filter if we have an uploaded document
                    if self.uploaded_document_id:
                        response_filtered = self.session.get(
                            f"{BACKEND_URL}/icns?document_id={self.uploaded_document_id}",
                            timeout=10
                        )
                        if response_filtered.status_code == 200:
                            filtered_icns = response_filtered.json()
                            self.log_test("Get ICNs (Filtered)", True, 
                                        f"Found {len(filtered_icns)} ICNs for document")
                        else:
                            self.log_test("Get ICNs (Filtered)", False, 
                                        f"Status code: {response_filtered.status_code}")
                    
                    return True
                else:
                    self.log_test("Get ICNs", False, f"Expected list, got: {type(icns)}")
                    return False
            else:
                self.log_test("Get ICNs", False, f"Status code: {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Get ICNs", False, error=e)
            return False

    async def test_websocket_connection(self):
        """Test WebSocket connection and message handling"""
        try:
            # Remove timeout parameter that's causing issues
            async with websockets.connect(WEBSOCKET_URL) as websocket:
                # Send a test message
                await websocket.send("test_connection")
                
                # Try to receive a message (with timeout)
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    self.log_test("WebSocket Connection", True, "Connected and can send/receive messages")
                    return True
                except asyncio.TimeoutError:
                    # No message received, but connection was successful
                    self.log_test("WebSocket Connection", True, "Connected successfully (no immediate response)")
                    return True
                    
        except Exception as e:
            self.log_test("WebSocket Connection", False, error=e)
            return False

    def test_database_integrity(self):
        """Test database structure and integrity"""
        try:
            # Check if database file exists in aquila directory
            db_path = "/app/aquila/aquila.db"
            if not os.path.exists(db_path):
                # Try backend directory
                db_path = "/app/backend/aquila.db"
                if not os.path.exists(db_path):
                    # Try root directory
                    db_path = "/app/aquila.db"
                    if not os.path.exists(db_path):
                        self.log_test("Database Integrity", False, "Database file not found")
                        return False
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if required tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['document', 'datamodule', 'icn']
            missing_tables = [table for table in required_tables if table not in tables]
            
            if missing_tables:
                self.log_test("Database Integrity", False, f"Missing tables: {missing_tables}")
                conn.close()
                return False
            
            # Check table structures
            for table in required_tables:
                cursor.execute(f"PRAGMA table_info({table});")
                columns = cursor.fetchall()
                if not columns:
                    self.log_test("Database Integrity", False, f"Table {table} has no columns")
                    conn.close()
                    return False
            
            conn.close()
            self.log_test("Database Integrity", True, f"All required tables present: {required_tables} (DB: {db_path})")
            return True
            
        except Exception as e:
            self.log_test("Database Integrity", False, error=e)
            return False

    def test_error_handling(self):
        """Test error handling for invalid requests"""
        try:
            # Test invalid endpoint
            response = self.session.get(f"{BACKEND_URL}/invalid-endpoint", timeout=10)
            if response.status_code == 404:
                self.log_test("Error Handling (404)", True, "Correctly returns 404 for invalid endpoint")
            else:
                self.log_test("Error Handling (404)", False, f"Expected 404, got {response.status_code}")
            
            # Test invalid file upload - the app accepts any file but should fail processing
            files = {'file': ('test.txt', b'not a pdf', 'text/plain')}
            response = self.session.post(f"{BACKEND_URL}/documents/upload", files=files, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                if "document_id" in result and result.get("status") == "processing":
                    # Wait a bit and check if processing failed
                    time.sleep(3)
                    doc_response = self.session.get(f"{BACKEND_URL}/documents", timeout=10)
                    if doc_response.status_code == 200:
                        documents = doc_response.json()
                        uploaded_doc = next((doc for doc in documents if doc['id'] == result['document_id']), None)
                        if uploaded_doc and uploaded_doc['status'] == 'failed':
                            self.log_test("Error Handling (Invalid File)", True, "Correctly fails processing of invalid file")
                        else:
                            self.log_test("Error Handling (Invalid File)", False, f"Expected failed status, got: {uploaded_doc['status'] if uploaded_doc else 'not found'}")
                    else:
                        self.log_test("Error Handling (Invalid File)", False, "Could not check document status")
                else:
                    self.log_test("Error Handling (Invalid File)", False, f"Unexpected upload response: {result}")
            else:
                self.log_test("Error Handling (Invalid File)", False, f"Unexpected status: {response.status_code}")
            
            return True
            
        except Exception as e:
            self.log_test("Error Handling", False, error=e)
            return False

    def wait_for_processing(self, max_wait=60):
        """Wait for document processing to complete"""
        if not self.uploaded_document_id:
            return False
            
        print(f"Waiting for document {self.uploaded_document_id} to finish processing...")
        
        for i in range(max_wait):
            try:
                response = self.session.get(f"{BACKEND_URL}/documents", timeout=10)
                if response.status_code == 200:
                    documents = response.json()
                    for doc in documents:
                        if doc['id'] == self.uploaded_document_id:
                            if doc['status'] == 'completed':
                                self.log_test("Document Processing", True, "Document processed successfully")
                                return True
                            elif doc['status'] == 'failed':
                                self.log_test("Document Processing", False, "Document processing failed")
                                return False
                            # Still processing, continue waiting
                            break
                
                time.sleep(1)
                if i % 10 == 0:
                    print(f"Still waiting... ({i}s)")
                    
            except Exception as e:
                print(f"Error checking processing status: {e}")
                
        self.log_test("Document Processing", False, "Timeout waiting for processing to complete")
        return False

    def run_all_tests(self):
        """Run all backend tests"""
        print("=" * 60)
        print("AQUILA S1000D-AI BACKEND TESTING")
        print("=" * 60)
        
        # Basic connectivity tests
        print("\n1. BASIC CONNECTIVITY TESTS")
        print("-" * 30)
        self.test_health_endpoint()
        
        # Database tests
        print("\n2. DATABASE TESTS")
        print("-" * 30)
        self.test_database_integrity()
        
        # API endpoint tests
        print("\n3. API ENDPOINT TESTS")
        print("-" * 30)
        self.test_get_documents()
        self.test_get_data_modules()
        self.test_get_icns()
        
        # File upload and processing tests
        print("\n4. FILE UPLOAD AND PROCESSING TESTS")
        print("-" * 30)
        upload_success = self.test_document_upload()
        
        if upload_success:
            # Wait for processing to complete
            processing_success = self.wait_for_processing()
            
            if processing_success:
                # Test endpoints again to see processed data
                print("\n5. POST-PROCESSING VERIFICATION")
                print("-" * 30)
                self.test_get_documents()
                self.test_get_data_modules()
                self.test_get_icns()
        
        # WebSocket tests
        print("\n6. WEBSOCKET TESTS")
        print("-" * 30)
        try:
            asyncio.run(self.test_websocket_connection())
        except Exception as e:
            self.log_test("WebSocket Connection", False, error=e)
        
        # Error handling tests
        print("\n7. ERROR HANDLING TESTS")
        print("-" * 30)
        self.test_error_handling()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results.values() if result['success'])
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        print("\nDETAILED RESULTS:")
        for test_name, result in self.test_results.items():
            status = "✅" if result['success'] else "❌"
            print(f"{status} {test_name}")
            if not result['success'] and result['error']:
                print(f"   Error: {result['error']}")
        
        return self.test_results

if __name__ == "__main__":
    tester = AquilaBackendTester()
    results = tester.run_all_tests()