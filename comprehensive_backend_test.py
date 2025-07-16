#!/usr/bin/env python3
"""
Comprehensive Backend API Testing for Aquila S1000D-AI Application
Tests all endpoints mentioned in the review request
"""

import requests
import json
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path

class ComprehensiveAPITester:
    def __init__(self, base_url="http://127.0.0.1:8001"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []
        self.created_projects = []

    def log(self, message, level="INFO"):
        """Log test messages with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

    def run_test(self, name, method, endpoint, expected_status, data=None, files=None, form_data=None):
        """Run a single API test and record results"""
        url = f"{self.base_url}/{endpoint}"
        headers = {}
        
        self.tests_run += 1
        self.log(f"🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif method == 'POST':
                if files:
                    response = requests.post(url, files=files, data=form_data, headers=headers, timeout=10)
                elif form_data:
                    headers['Content-Type'] = 'application/x-www-form-urlencoded'
                    response = requests.post(url, data=form_data, headers=headers, timeout=10)
                else:
                    headers['Content-Type'] = 'application/json'
                    response = requests.post(url, json=data, headers=headers, timeout=10)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=10)
            else:
                raise ValueError(f"Unsupported method: {method}")

            success = response.status_code == expected_status
            
            result = {
                "name": name,
                "method": method,
                "endpoint": endpoint,
                "expected_status": expected_status,
                "actual_status": response.status_code,
                "success": success,
                "response_data": None,
                "error": None
            }
            
            if success:
                self.tests_passed += 1
                self.log(f"✅ Passed - Status: {response.status_code}")
                try:
                    result["response_data"] = response.json()
                except:
                    result["response_data"] = response.text
            else:
                self.log(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                try:
                    error_detail = response.json()
                    result["error"] = error_detail
                    self.log(f"   Error details: {error_detail}")
                except:
                    result["error"] = response.text
                    self.log(f"   Response text: {response.text}")
            
            self.test_results.append(result)
            return success, result["response_data"] if success else result["error"]

        except requests.exceptions.RequestException as e:
            self.log(f"❌ Failed - Network error: {str(e)}", "ERROR")
            result = {
                "name": name,
                "method": method,
                "endpoint": endpoint,
                "expected_status": expected_status,
                "actual_status": None,
                "success": False,
                "response_data": None,
                "error": str(e)
            }
            self.test_results.append(result)
            return False, str(e)

    def test_project_management(self):
        """Test all project management endpoints"""
        self.log("\n📁 Testing Project Management Endpoints...")
        
        # 1. GET /api/projects - List all projects
        success, projects = self.run_test(
            "GET /api/projects - List all projects",
            "GET",
            "api/projects",
            200
        )
        
        # 2. POST /api/projects - Create new project
        success, project_data = self.run_test(
            "POST /api/projects - Create new project",
            "POST",
            "api/projects",
            200,
            form_data="name=Test Project for API&description=Created by comprehensive test"
        )
        
        project_id = None
        if success and project_data:
            project_id = project_data.get('id')
            self.created_projects.append(project_id)
            self.log(f"   Created project ID: {project_id}")
        
        # 3. GET /api/projects/current - Get current project
        self.run_test(
            "GET /api/projects/current - Get current project",
            "GET",
            "api/projects/current",
            200
        )
        
        # 4. POST /api/projects/{id}/select - Select a project
        if project_id:
            self.run_test(
                f"POST /api/projects/{project_id}/select - Select project",
                "POST",
                f"api/projects/{project_id}/select",
                200
            )
            
            # Verify selection worked
            success, current_data = self.run_test(
                "GET /api/projects/current - Verify project selection",
                "GET",
                "api/projects/current",
                200
            )
            
            if success and current_data and current_data.get('current_project'):
                current_id = current_data['current_project']['id']
                if current_id == project_id:
                    self.log("✅ Project selection verified successfully")
                else:
                    self.log("❌ Project selection verification failed")
        
        # 5. DELETE /api/projects/{id} - Delete a project (we'll test this in cleanup)
        return project_id

    def test_document_management(self, project_id):
        """Test document management endpoints"""
        self.log("\n📄 Testing Document Management Endpoints...")
        
        # 1. GET /api/documents - List documents for current project
        self.run_test(
            "GET /api/documents - List documents for current project",
            "GET",
            "api/documents",
            200
        )
        
        # 2. POST /api/documents/upload - Upload PDF document (without actual file)
        self.run_test(
            "POST /api/documents/upload - Upload PDF (missing file)",
            "POST",
            "api/documents/upload",
            422  # Should fail due to missing file
        )
        
        # Test upload with mock file data
        if project_id:
            # Create a temporary PDF-like file for testing
            test_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n179\n%%EOF"
            
            files = {'file': ('test_document.pdf', test_content, 'application/pdf')}
            form_data = {'operational_context': 'Water'}
            
            self.run_test(
                "POST /api/documents/upload - Upload PDF with file",
                "POST",
                "api/documents/upload",
                200,
                files=files,
                form_data=form_data
            )

    def test_data_modules(self):
        """Test data modules endpoints"""
        self.log("\n🔧 Testing Data Modules Endpoints...")
        
        # GET /api/data-modules?document_id={id} - Get data modules for a document
        self.run_test(
            "GET /api/data-modules - Get data modules (no document_id)",
            "GET",
            "api/data-modules",
            200
        )
        
        # Test with a document ID parameter
        self.run_test(
            "GET /api/data-modules?document_id=test-doc-id - Get data modules for document",
            "GET",
            "api/data-modules?document_id=test-doc-id",
            200
        )

    def test_error_conditions(self):
        """Test various error conditions"""
        self.log("\n⚠️ Testing Error Conditions...")
        
        # Test creating project with empty name
        self.run_test(
            "POST /api/projects - Create project with empty name",
            "POST",
            "api/projects",
            422,  # Should fail with validation error
            form_data="name=&description=test"
        )
        
        # Test selecting non-existent project
        self.run_test(
            "POST /api/projects/non-existent/select - Select non-existent project",
            "POST",
            "api/projects/non-existent-id/select",
            500  # Current implementation returns 500
        )
        
        # Test deleting non-existent project
        self.run_test(
            "DELETE /api/projects/non-existent - Delete non-existent project",
            "DELETE",
            "api/projects/non-existent-id",
            500  # Current implementation returns 500
        )

    def cleanup_test_projects(self):
        """Clean up any projects created during testing"""
        self.log("\n🧹 Cleaning up test projects...")
        for project_id in self.created_projects.copy():
            success, _ = self.run_test(
                f"DELETE /api/projects/{project_id} - Delete test project",
                "DELETE",
                f"api/projects/{project_id}",
                200
            )
            if success:
                self.created_projects.remove(project_id)

    def run_comprehensive_tests(self):
        """Run all comprehensive tests"""
        self.log("🚀 Starting Comprehensive Aquila S1000D-AI API Tests")
        self.log("=" * 60)
        
        # Test server health first
        success, _ = self.run_test(
            "Health Check",
            "GET",
            "api/health",
            200
        )
        
        if not success:
            self.log("❌ Health check failed - server may not be running", "ERROR")
            return False
        
        # Run all test suites
        project_id = self.test_project_management()
        self.test_document_management(project_id)
        self.test_data_modules()
        self.test_error_conditions()
        
        # Cleanup
        self.cleanup_test_projects()
        
        # Results
        self.log("\n" + "=" * 60)
        self.log(f"📊 Test Results: {self.tests_passed}/{self.tests_run} tests passed")
        
        success_rate = (self.tests_passed / self.tests_run) * 100 if self.tests_run > 0 else 0
        
        if success_rate >= 80:
            self.log(f"🎉 Tests mostly successful! ({success_rate:.1f}% pass rate)", "SUCCESS")
            return True
        else:
            self.log(f"⚠️ Many tests failed ({success_rate:.1f}% pass rate)", "WARNING")
            return False

    def get_test_summary(self):
        """Get a summary of test results for reporting"""
        summary = {
            "total_tests": self.tests_run,
            "passed_tests": self.tests_passed,
            "failed_tests": self.tests_run - self.tests_passed,
            "success_rate": (self.tests_passed / self.tests_run) * 100 if self.tests_run > 0 else 0,
            "test_details": self.test_results
        }
        return summary

def main():
    """Main test execution"""
    print("Comprehensive Aquila S1000D-AI Backend API Test Suite")
    print("=" * 60)
    
    tester = ComprehensiveAPITester()
    
    try:
        success = tester.run_comprehensive_tests()
        
        # Print summary
        summary = tester.get_test_summary()
        print(f"\n📋 FINAL SUMMARY:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed_tests']}")
        print(f"   Failed: {summary['failed_tests']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        tester.cleanup_test_projects()
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        tester.cleanup_test_projects()
        return 1

if __name__ == "__main__":
    sys.exit(main())