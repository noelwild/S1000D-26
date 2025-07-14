#!/usr/bin/env python3
"""
Final S1000D Backend Verification Test
"""

import requests
import json

BACKEND_URL = "http://127.0.0.1:8001/api"

def main():
    print("=" * 70)
    print("FINAL S1000D BACKEND VERIFICATION")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Health Check
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200 and response.json().get("status") == "healthy":
            results["Health Check"] = "✅ PASS"
        else:
            results["Health Check"] = "❌ FAIL"
    except Exception as e:
        results["Health Check"] = f"❌ FAIL: {e}"
    
    # Test 2: Documents API
    try:
        response = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        if response.status_code == 200:
            docs = response.json()
            completed_docs = [d for d in docs if d['status'] == 'completed']
            results["Documents API"] = f"✅ PASS: {len(docs)} total, {len(completed_docs)} completed"
        else:
            results["Documents API"] = f"❌ FAIL: Status {response.status_code}"
    except Exception as e:
        results["Documents API"] = f"❌ FAIL: {e}"
    
    # Test 3: Data Modules API with S1000D Compliance
    try:
        response = requests.get(f"{BACKEND_URL}/data-modules", timeout=10)
        if response.status_code == 200:
            modules = response.json()
            
            # Check S1000D compliance
            compliant_modules = 0
            for module in modules:
                dmc = module.get('dmc', '')
                if (dmc and '-' in dmc and 
                    len(dmc.split('-')) >= 5 and
                    module.get('title') and
                    module.get('verbatim_content') and
                    module.get('ste_content') and
                    module.get('type')):
                    compliant_modules += 1
            
            compliance_rate = (compliant_modules / len(modules)) * 100 if modules else 0
            if compliance_rate >= 90:
                results["S1000D Compliance"] = f"✅ PASS: {compliance_rate:.1f}% ({compliant_modules}/{len(modules)})"
            else:
                results["S1000D Compliance"] = f"❌ FAIL: {compliance_rate:.1f}% ({compliant_modules}/{len(modules)})"
        else:
            results["S1000D Compliance"] = f"❌ FAIL: Status {response.status_code}"
    except Exception as e:
        results["S1000D Compliance"] = f"❌ FAIL: {e}"
    
    # Test 4: STE vs Verbatim Content Quality
    try:
        response = requests.get(f"{BACKEND_URL}/data-modules", timeout=10)
        if response.status_code == 200:
            modules = response.json()
            
            different_content = 0
            total_with_content = 0
            
            for module in modules:
                verbatim = module.get('verbatim_content', '')
                ste = module.get('ste_content', '')
                if verbatim and ste:
                    total_with_content += 1
                    if verbatim != ste:
                        different_content += 1
            
            if total_with_content > 0:
                diff_rate = (different_content / total_with_content) * 100
                if diff_rate >= 80:
                    results["STE Content Quality"] = f"✅ PASS: {diff_rate:.1f}% different from verbatim"
                else:
                    results["STE Content Quality"] = f"❌ FAIL: {diff_rate:.1f}% different from verbatim"
            else:
                results["STE Content Quality"] = "❌ FAIL: No content to analyze"
        else:
            results["STE Content Quality"] = f"❌ FAIL: Status {response.status_code}"
    except Exception as e:
        results["STE Content Quality"] = f"❌ FAIL: {e}"
    
    # Test 5: Content Type Classification
    try:
        response = requests.get(f"{BACKEND_URL}/data-modules", timeout=10)
        if response.status_code == 200:
            modules = response.json()
            
            content_types = set()
            for module in modules:
                content_type = module.get('type', '')
                if content_type:
                    content_types.add(content_type)
            
            if len(content_types) > 1:
                results["Content Classification"] = f"✅ PASS: {len(content_types)} types found: {', '.join(content_types)}"
            else:
                results["Content Classification"] = f"❌ FAIL: Only {len(content_types)} type(s) found"
        else:
            results["Content Classification"] = f"❌ FAIL: Status {response.status_code}"
    except Exception as e:
        results["Content Classification"] = f"❌ FAIL: {e}"
    
    # Test 6: ICNs API
    try:
        response = requests.get(f"{BACKEND_URL}/icns", timeout=10)
        if response.status_code == 200:
            icns = response.json()
            results["ICNs API"] = f"✅ PASS: {len(icns)} ICNs found"
        else:
            results["ICNs API"] = f"❌ FAIL: Status {response.status_code}"
    except Exception as e:
        results["ICNs API"] = f"❌ FAIL: {e}"
    
    # Print Results
    print("\nTEST RESULTS:")
    print("-" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        print(f"{result} {test_name}")
        if result.startswith("✅"):
            passed += 1
    
    print(f"\nSUMMARY: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL S1000D BACKEND TESTS PASSED!")
        print("The improved Aquila S1000D system is working correctly.")
    else:
        print(f"\n⚠️  {total-passed} tests failed. System needs attention.")

if __name__ == "__main__":
    main()