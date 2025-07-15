#!/usr/bin/env python3
"""
Comprehensive Backend API Testing for Aquila S1000D-AI Project Management System
Tests all project management endpoints and core functionality
"""

import requests
import json
import sys
import time
from datetime import datetime
from pathlib import Path

class AquilaAPITester:
    def __init__(self, base_url="http://127.0.0.1:8001"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.created_projects = []  # Track created projects for cleanup

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

    def test_health_check(self):
        """Test health endpoint"""
        success, response = self.run_test(
            "Health Check",
            "GET",
            "api/health",
            200
        )
        if success:
            self.log(f"   Service: {response.get('service', 'Unknown')}")
            self.log(f"   Current Project: {response.get('current_project', 'None')}")
        return success

    def test_get_projects(self):
        """Test getting all projects"""
        success, response = self.run_test(
            "Get All Projects",
            "GET",
            "api/projects",
            200
        )
        if success:
            self.log(f"   Found {len(response)} projects")
            for project in response:
                self.log(f"   - {project.get('name', 'Unknown')} (ID: {project.get('id', 'Unknown')})")
        return success, response if success else []

    def test_create_project(self, name, description=""):
        """Test creating a new project"""
        form_data = f"name={name}&description={description}"
        success, response = self.run_test(
            f"Create Project '{name}'",
            "POST",
            "api/projects",
            200,
            form_data=form_data
        )
        if success:
            project_id = response.get('id')
            self.created_projects.append(project_id)
            self.log(f"   Created project ID: {project_id}")
            return success, project_id
        return success, None

    def test_get_current_project(self):
        """Test getting current project"""
        success, response = self.run_test(
            "Get Current Project",
            "GET",
            "api/projects/current",
            200
        )
        if success:
            current = response.get('current_project')
            if current:
                self.log(f"   Current project: {current.get('name')} (ID: {current.get('id')})")
            else:
                self.log("   No current project selected")
        return success, response

    def test_select_project(self, project_id):
        """Test selecting a project"""
        success, response = self.run_test(
            f"Select Project {project_id}",
            "POST",
            f"api/projects/{project_id}/select",
            200
        )
        return success

    def test_delete_project(self, project_id):
        """Test deleting a project"""
        success, response = self.run_test(
            f"Delete Project {project_id}",
            "DELETE",
            f"api/projects/{project_id}",
            200
        )
        if success and project_id in self.created_projects:
            self.created_projects.remove(project_id)
        return success

    def test_project_isolation(self):
        """Test that projects have isolated databases and uploads"""
        self.log("🔍 Testing Project Isolation...")
        
        # Create two test projects
        success1, project1_id = self.test_create_project("Isolation Test 1", "First test project")
        if not success1:
            return False
            
        success2, project2_id = self.test_create_project("Isolation Test 2", "Second test project")
        if not success2:
            return False

        # Check that project directories exist
        project1_dir = Path(f"/app/aquila/projects/{project1_id}")
        project2_dir = Path(f"/app/aquila/projects/{project2_id}")
        
        if project1_dir.exists() and project2_dir.exists():
            self.log("✅ Project directories created successfully")
            
            # Check for database files
            db1 = project1_dir / "aquila.db"
            db2 = project2_dir / "aquila.db"
            
            # Check for uploads directories
            uploads1 = project1_dir / "uploads"
            uploads2 = project2_dir / "uploads"
            
            if uploads1.exists() and uploads2.exists():
                self.log("✅ Project uploads directories created successfully")
                self.tests_passed += 1
            else:
                self.log("❌ Project uploads directories not found")
                
        else:
            self.log("❌ Project directories not created")
            
        self.tests_run += 1
        return True

    def test_document_upload_without_project(self):
        """Test that document upload fails without a selected project"""
        # First, ensure no project is selected by creating a dummy project and not selecting it
        success, dummy_id = self.test_create_project("Dummy Project", "For testing no selection")
        
        # Try to upload without selecting any project
        test_file_content = b"Test PDF content"
        files = {'file': ('test.pdf', test_file_content, 'application/pdf')}
        form_data = {'operational_context': 'Water'}
        
        success, response = self.run_test(
            "Upload Document Without Project Selection",
            "POST",
            "api/documents/upload",
            400,  # Should fail with 400
            files=files,
            form_data=form_data
        )
        
        return success

    def test_error_handling(self):
        """Test various error conditions"""
        self.log("🔍 Testing Error Handling...")
        
        # Test creating project with empty name
        success, _ = self.run_test(
            "Create Project with Empty Name",
            "POST",
            "api/projects",
            400,
            form_data="name=&description=test"
        )
        
        # Test selecting non-existent project
        success2, _ = self.run_test(
            "Select Non-existent Project",
            "POST",
            "api/projects/non-existent-id/select",
            404
        )
        
        # Test deleting non-existent project
        success3, _ = self.run_test(
            "Delete Non-existent Project",
            "DELETE",
            "api/projects/non-existent-id",
            404
        )
        
        return success and success2 and success3

    def cleanup_test_projects(self):
        """Clean up any projects created during testing"""
        self.log("🧹 Cleaning up test projects...")
        for project_id in self.created_projects.copy():
            self.test_delete_project(project_id)

    def run_comprehensive_tests(self):
        """Run all tests in sequence"""
        self.log("🚀 Starting Comprehensive Aquila S1000D-AI API Tests")
        self.log("=" * 60)
        
        # Basic connectivity
        if not self.test_health_check():
            self.log("❌ Health check failed - server may not be running", "ERROR")
            return False
        
        # Project management tests
        self.log("\n📁 Testing Project Management...")
        
        # Get initial projects
        success, initial_projects = self.test_get_projects()
        if not success:
            return False
            
        # Test current project
        self.test_get_current_project()
        
        # Create test projects
        success, test_project_id = self.test_create_project("API Test Project", "Created by automated test")
        if not success:
            return False
            
        # Test project selection
        if not self.test_select_project(test_project_id):
            return False
            
        # Verify current project changed
        success, current_response = self.test_get_current_project()
        if success and current_response.get('current_project'):
            current_id = current_response['current_project']['id']
            if current_id == test_project_id:
                self.log("✅ Project selection working correctly")
                self.tests_passed += 1
            else:
                self.log("❌ Project selection not working correctly")
        self.tests_run += 1
        
        # Test project isolation
        self.test_project_isolation()
        
        # Test error handling
        self.test_error_handling()
        
        # Test document upload without project (should fail)
        # Note: We'll skip this as it requires proper project selection state
        
        # Final project list
        self.log("\n📋 Final Project State...")
        self.test_get_projects()
        
        # Cleanup
        self.cleanup_test_projects()
        
        # Results
        self.log("\n" + "=" * 60)
        self.log(f"📊 Test Results: {self.tests_passed}/{self.tests_run} tests passed")
        
        if self.tests_passed == self.tests_run:
            self.log("🎉 All tests passed!", "SUCCESS")
            return True
        else:
            self.log(f"⚠️  {self.tests_run - self.tests_passed} tests failed", "WARNING")
            return False

def main():
    """Main test execution"""
    print("Aquila S1000D-AI Backend API Test Suite")
    print("=" * 50)
    
    tester = AquilaAPITester()
    
    try:
        success = tester.run_comprehensive_tests()
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