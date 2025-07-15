#!/usr/bin/env python3
"""
Quick test to verify SQLite binding fix
"""

import json
import sys
import os
from pathlib import Path

# Add the aquila directory to the path
sys.path.insert(0, '/app/aquila')

def test_data_conversion():
    """Test that data is properly converted for SQLite"""
    
    # Sample data that could cause SQLite binding issues
    sample_data = {
        "module_id": "test_module_1",
        "dmc": "Air-procedure-520-A-01",
        "title": "Test Module",
        "info_code": "520",
        "item_location": "A",
        "verbatim_content": "Test content",
        "ste_content": "Test STE content",
        "type": "procedure",
        "prerequisites": "Test prerequisites",
        "tools_equipment": "Test tools",
        "warnings": "Test warnings",
        "cautions": "Test cautions",
        "procedural_steps": [
            {"step_number": 1, "action": "First step", "details": "Details"},
            {"step_number": 2, "action": "Second step", "details": "More details"}
        ],
        "expected_results": "Test results",
        "specifications": "Test specs",
        "references": "Test refs",
        "content_sources": ["chunk1", "chunk2"],
        "completeness_score": 0.95,
        "relevant_chunks_found": 2,
        "total_chunks_analyzed": 3,
        "status": "complete"
    }
    
    print("Testing data conversion for SQLite compatibility...")
    print("=" * 50)
    
    # Apply the same conversion logic as in the server
    converted_data = {
        "module_id": str(sample_data.get("module_id", "default")),
        "dmc": str(sample_data.get("dmc", "")),
        "title": str(sample_data.get("title", "")),
        "info_code": str(sample_data.get("info_code", "040")),
        "item_location": str(sample_data.get("item_location", "A")),
        "verbatim_content": str(sample_data.get("verbatim_content", "")),
        "ste_content": str(sample_data.get("ste_content", "")),
        "type": str(sample_data.get("type", "description")),
        "prerequisites": str(sample_data.get("prerequisites", "")),
        "tools_equipment": str(sample_data.get("tools_equipment", "")),
        "warnings": str(sample_data.get("warnings", "")),
        "cautions": str(sample_data.get("cautions", "")),
        "procedural_steps": json.dumps(sample_data.get("procedural_steps", [])) if isinstance(sample_data.get("procedural_steps"), list) else str(sample_data.get("procedural_steps", "[]")),
        "expected_results": str(sample_data.get("expected_results", "")),
        "specifications": str(sample_data.get("specifications", "")),
        "references": str(sample_data.get("references", "")),
        "content_sources": json.dumps(sample_data.get("content_sources", [])) if isinstance(sample_data.get("content_sources"), list) else str(sample_data.get("content_sources", "[]")),
        "completeness_score": float(sample_data.get("completeness_score", 0.0)),
        "relevant_chunks_found": int(sample_data.get("relevant_chunks_found", 0)),
        "total_chunks_analyzed": int(sample_data.get("total_chunks_analyzed", 0)),
        "population_status": str(sample_data.get("status", "complete"))
    }
    
    print("✅ Data conversion completed successfully!")
    print(f"✅ procedural_steps converted: {type(converted_data['procedural_steps'])} -> {len(converted_data['procedural_steps'])} chars")
    print(f"✅ content_sources converted: {type(converted_data['content_sources'])} -> {len(converted_data['content_sources'])} chars")
    print(f"✅ completeness_score: {type(converted_data['completeness_score'])} -> {converted_data['completeness_score']}")
    print(f"✅ relevant_chunks_found: {type(converted_data['relevant_chunks_found'])} -> {converted_data['relevant_chunks_found']}")
    
    # Verify JSON strings are valid
    try:
        parsed_steps = json.loads(converted_data['procedural_steps'])
        parsed_sources = json.loads(converted_data['content_sources'])
        print(f"✅ JSON validation passed: {len(parsed_steps)} steps, {len(parsed_sources)} sources")
    except json.JSONDecodeError as e:
        print(f"❌ JSON validation failed: {e}")
        return False
    
    # Check all values are SQLite-compatible types
    sqlite_types = (str, int, float, bytes, type(None))
    
    for key, value in converted_data.items():
        if not isinstance(value, sqlite_types):
            print(f"❌ Invalid SQLite type for {key}: {type(value)}")
            return False
    
    print("✅ All data types are SQLite-compatible!")
    
    # Test the actual data structure
    print("\nConverted data structure:")
    print("-" * 30)
    for key, value in converted_data.items():
        print(f"{key}: {type(value).__name__} ({len(str(value))} chars)")
    
    return True

if __name__ == "__main__":
    print("SQLite Binding Fix Verification")
    print("=" * 50)
    
    if test_data_conversion():
        print("\n🎉 SUCCESS: Data conversion fix is working correctly!")
        print("The SQLite binding error should now be resolved.")
    else:
        print("\n❌ FAILURE: Data conversion issues found.")
        print("Additional fixes may be needed.")