#!/usr/bin/env python3
"""
Focused S1000D Testing for Aquila Backend System
Tests the improved S1000D compliance and structured data modules
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
BACKEND_URL = "https://3bc5d385-7e6d-4767-8491-03dab3b59a0f.preview.emergentagent.com/api"
SAMPLE_PDF_PATH = "/app/aquila/tmpn48mr8t9.pdf"

class S1000DFocusedTester:
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

    def test_sample_pdf_upload(self):
        """Test upload and processing of the sample PDF"""
        try:
            if not os.path.exists(SAMPLE_PDF_PATH):
                self.log_test("Sample PDF Upload", False, f"Sample PDF not found at {SAMPLE_PDF_PATH}")
                return False
                
            with open(SAMPLE_PDF_PATH, 'rb') as f:
                files = {'file': ('tmpn48mr8t9.pdf', f, 'application/pdf')}
                data = {'operational_context': 'Water'}
                
                response = self.session.post(
                    f"{BACKEND_URL}/documents/upload",
                    files=files,
                    data=data,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                if "document_id" in result and result.get("status") == "processing":
                    self.uploaded_document_id = result["document_id"]
                    self.log_test("Sample PDF Upload", True, f"Document ID: {self.uploaded_document_id}")
                    return True
                else:
                    self.log_test("Sample PDF Upload", False, f"Unexpected response: {result}")
                    return False
            else:
                self.log_test("Sample PDF Upload", False, f"Status code: {response.status_code}, Response: {response.text}")
                return False
                
        except Exception as e:
            self.log_test("Sample PDF Upload", False, error=e)
            return False

    def wait_for_processing(self, max_wait=120):
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
                if i % 15 == 0:
                    print(f"Still waiting... ({i}s)")
                    
            except Exception as e:
                print(f"Error checking processing status: {e}")
                
        self.log_test("Document Processing", False, "Timeout waiting for processing to complete")
        return False

    def test_s1000d_compliance(self):
        """Test S1000D compliance - DMC codes, info codes, content types"""
        try:
            if not self.uploaded_document_id:
                self.log_test("S1000D Compliance", False, "No uploaded document to test")
                return False
                
            response = self.session.get(
                f"{BACKEND_URL}/data-modules?document_id={self.uploaded_document_id}",
                timeout=10
            )
            
            if response.status_code != 200:
                self.log_test("S1000D Compliance", False, f"Failed to get data modules: {response.status_code}")
                return False
                
            modules = response.json()
            if not modules:
                self.log_test("S1000D Compliance", False, "No data modules found")
                return False
            
            compliance_issues = []
            valid_info_codes = ['040', '520', '320', '730', '012', '014']  # Common S1000D info codes
            valid_content_types = ['procedure', 'description', 'fault_isolation', 'maintenance_planning']
            
            for module in modules:
                # Check DMC format
                dmc = module.get('dmc', '')
                if not dmc or len(dmc.split('-')) < 5:
                    compliance_issues.append(f"Invalid DMC format: {dmc}")
                
                # Check info code
                # Extract info code from DMC (should be third component)
                dmc_parts = dmc.split('-')
                if len(dmc_parts) >= 3:
                    info_code = dmc_parts[2]
                    if info_code not in valid_info_codes:
                        compliance_issues.append(f"Invalid info code: {info_code} in DMC {dmc}")
                
                # Check content type
                content_type = module.get('type', '')
                if content_type not in valid_content_types:
                    compliance_issues.append(f"Invalid content type: {content_type}")
                
                # Check required fields
                required_fields = ['id', 'dmc', 'title', 'verbatim_content', 'ste_content', 'type']
                for field in required_fields:
                    if not module.get(field):
                        compliance_issues.append(f"Missing or empty field: {field} in module {dmc}")
            
            if compliance_issues:
                self.log_test("S1000D Compliance", False, f"Found {len(compliance_issues)} compliance issues")
                for issue in compliance_issues[:5]:  # Show first 5 issues
                    print(f"   - {issue}")
                return False
            else:
                self.log_test("S1000D Compliance", True, f"All {len(modules)} modules are S1000D compliant")
                return True
                
        except Exception as e:
            self.log_test("S1000D Compliance", False, error=e)
            return False

    def test_content_classification(self):
        """Test proper classification of different content types"""
        try:
            if not self.uploaded_document_id:
                self.log_test("Content Classification", False, "No uploaded document to test")
                return False
                
            response = self.session.get(
                f"{BACKEND_URL}/data-modules?document_id={self.uploaded_document_id}",
                timeout=10
            )
            
            if response.status_code != 200:
                self.log_test("Content Classification", False, f"Failed to get data modules: {response.status_code}")
                return False
                
            modules = response.json()
            if not modules:
                self.log_test("Content Classification", False, "No data modules found")
                return False
            
            # Analyze content types
            content_types = {}
            for module in modules:
                content_type = module.get('type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
            
            # Check for variety in content types (good classification should produce different types)
            if len(content_types) > 1:
                type_summary = ", ".join([f"{k}: {v}" for k, v in content_types.items()])
                self.log_test("Content Classification", True, f"Found diverse content types: {type_summary}")
                return True
            else:
                single_type = list(content_types.keys())[0]
                self.log_test("Content Classification", False, f"All content classified as single type: {single_type}")
                return False
                
        except Exception as e:
            self.log_test("Content Classification", False, error=e)
            return False

    def test_ste_vs_verbatim_content(self):
        """Test that STE content is meaningfully different from verbatim content"""
        try:
            if not self.uploaded_document_id:
                self.log_test("STE vs Verbatim", False, "No uploaded document to test")
                return False
                
            response = self.session.get(
                f"{BACKEND_URL}/data-modules?document_id={self.uploaded_document_id}",
                timeout=10
            )
            
            if response.status_code != 200:
                self.log_test("STE vs Verbatim", False, f"Failed to get data modules: {response.status_code}")
                return False
                
            modules = response.json()
            if not modules:
                self.log_test("STE vs Verbatim", False, "No data modules found")
                return False
            
            meaningful_differences = 0
            identical_content = 0
            
            for module in modules:
                verbatim = module.get('verbatim_content', '').strip()
                ste = module.get('ste_content', '').strip()
                
                if not verbatim or not ste:
                    continue
                
                # Check if content is identical (bad)
                if verbatim == ste:
                    identical_content += 1
                    continue
                
                # Check for meaningful differences (good STE characteristics)
                ste_lower = ste.lower()
                verbatim_lower = verbatim.lower()
                
                # STE should be simpler, more direct
                ste_indicators = [
                    len(ste.split()) < len(verbatim.split()),  # Shorter
                    'must' in ste_lower or 'shall' in ste_lower,  # Direct commands
                    'do not' in ste_lower,  # Clear negatives
                    ste_lower.count('.') >= verbatim_lower.count('.'),  # More sentences (shorter sentences)
                ]
                
                if any(ste_indicators):
                    meaningful_differences += 1
            
            total_modules = len([m for m in modules if m.get('verbatim_content') and m.get('ste_content')])
            
            if total_modules == 0:
                self.log_test("STE vs Verbatim", False, "No modules with both verbatim and STE content")
                return False
            
            if meaningful_differences > identical_content:
                self.log_test("STE vs Verbatim", True, 
                            f"STE shows meaningful differences in {meaningful_differences}/{total_modules} modules")
                return True
            else:
                self.log_test("STE vs Verbatim", False, 
                            f"STE content too similar to verbatim: {identical_content} identical, {meaningful_differences} different")
                return False
                
        except Exception as e:
            self.log_test("STE vs Verbatim", False, error=e)
            return False

    def test_structured_fields(self):
        """Test that structured fields contain relevant information"""
        try:
            if not self.uploaded_document_id:
                self.log_test("Structured Fields", False, "No uploaded document to test")
                return False
                
            response = self.session.get(
                f"{BACKEND_URL}/data-modules?document_id={self.uploaded_document_id}",
                timeout=10
            )
            
            if response.status_code != 200:
                self.log_test("Structured Fields", False, f"Failed to get data modules: {response.status_code}")
                return False
                
            modules = response.json()
            if not modules:
                self.log_test("Structured Fields", False, "No data modules found")
                return False
            
            field_quality_score = 0
            total_checks = 0
            
            for module in modules:
                # Check title quality
                title = module.get('title', '')
                if title and len(title) > 5 and not title.lower().startswith('unknown'):
                    field_quality_score += 1
                total_checks += 1
                
                # Check DMC structure
                dmc = module.get('dmc', '')
                if dmc and '-' in dmc and len(dmc.split('-')) >= 4:
                    field_quality_score += 1
                total_checks += 1
                
                # Check content relevance
                verbatim = module.get('verbatim_content', '')
                ste = module.get('ste_content', '')
                if verbatim and ste and len(verbatim) > 20 and len(ste) > 10:
                    field_quality_score += 1
                total_checks += 1
                
                # Check type classification
                content_type = module.get('type', '')
                if content_type and content_type != 'unknown':
                    field_quality_score += 1
                total_checks += 1
            
            if total_checks == 0:
                self.log_test("Structured Fields", False, "No data to evaluate")
                return False
            
            quality_percentage = (field_quality_score / total_checks) * 100
            
            if quality_percentage >= 75:
                self.log_test("Structured Fields", True, 
                            f"Structured fields quality: {quality_percentage:.1f}% ({field_quality_score}/{total_checks})")
                return True
            else:
                self.log_test("Structured Fields", False, 
                            f"Poor structured fields quality: {quality_percentage:.1f}% ({field_quality_score}/{total_checks})")
                return False
                
        except Exception as e:
            self.log_test("Structured Fields", False, error=e)
            return False

    def test_data_modules_api_fields(self):
        """Test that data modules API returns all required S1000D fields"""
        try:
            if not self.uploaded_document_id:
                self.log_test("Data Modules API Fields", False, "No uploaded document to test")
                return False
                
            response = self.session.get(
                f"{BACKEND_URL}/data-modules?document_id={self.uploaded_document_id}",
                timeout=10
            )
            
            if response.status_code != 200:
                self.log_test("Data Modules API Fields", False, f"Failed to get data modules: {response.status_code}")
                return False
                
            modules = response.json()
            if not modules:
                self.log_test("Data Modules API Fields", False, "No data modules found")
                return False
            
            # Check required S1000D fields
            required_fields = ['id', 'dmc', 'title', 'verbatim_content', 'ste_content', 'type']
            
            missing_fields = []
            for module in modules:
                for field in required_fields:
                    if field not in module:
                        missing_fields.append(f"Module {module.get('dmc', 'unknown')}: missing {field}")
            
            if missing_fields:
                self.log_test("Data Modules API Fields", False, f"Missing fields found: {len(missing_fields)} issues")
                for issue in missing_fields[:3]:  # Show first 3 issues
                    print(f"   - {issue}")
                return False
            else:
                self.log_test("Data Modules API Fields", True, 
                            f"All {len(modules)} modules have required S1000D fields")
                return True
                
        except Exception as e:
            self.log_test("Data Modules API Fields", False, error=e)
            return False

    def run_focused_tests(self):
        """Run focused S1000D tests"""
        print("=" * 70)
        print("AQUILA S1000D FOCUSED TESTING - IMPROVED SYSTEM")
        print("=" * 70)
        
        # Upload sample PDF
        print("\n1. SAMPLE PDF UPLOAD AND PROCESSING")
        print("-" * 40)
        upload_success = self.test_sample_pdf_upload()
        
        if upload_success:
            processing_success = self.wait_for_processing()
            
            if processing_success:
                # Run S1000D specific tests
                print("\n2. S1000D COMPLIANCE VERIFICATION")
                print("-" * 40)
                self.test_s1000d_compliance()
                self.test_data_modules_api_fields()
                
                print("\n3. CONTENT CLASSIFICATION TESTING")
                print("-" * 40)
                self.test_content_classification()
                
                print("\n4. STE QUALITY VERIFICATION")
                print("-" * 40)
                self.test_ste_vs_verbatim_content()
                
                print("\n5. STRUCTURED FIELDS VALIDATION")
                print("-" * 40)
                self.test_structured_fields()
            else:
                print("❌ Cannot run further tests - document processing failed")
        else:
            print("❌ Cannot run further tests - document upload failed")
        
        # Summary
        print("\n" + "=" * 70)
        print("S1000D FOCUSED TEST SUMMARY")
        print("=" * 70)
        
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
            if result['details']:
                print(f"   {result['details']}")
            if not result['success'] and result['error']:
                print(f"   Error: {result['error']}")
        
        return self.test_results

if __name__ == "__main__":
    tester = S1000DFocusedTester()
    results = tester.run_focused_tests()