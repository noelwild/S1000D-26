#!/usr/bin/env python3
"""
Debug test for enhanced processing system
"""

import asyncio
import sys
import os

# Add the aquila directory to the path
sys.path.insert(0, '/app/aquila')

from server import TextCleaner, ChunkingStrategy, DocumentPlanner, ContentPopulator
import openai

# Load API key
with open('/app/aquila/keys.txt', 'r') as f:
    for line in f:
        if line.startswith('OPENAI_API_KEY='):
            openai.api_key = line.split('=', 1)[1].strip()
            break

async def test_text_cleaning():
    """Test text cleaning functionality"""
    print("Testing Text Cleaning...")
    
    sample_text = """
    Page 1
    
    MAINTENANCE MANUAL
    Engine Oil Change Procedure
    
    1. INTRODUCTION
    This manual provides step-by-step instructions for performing
    routine engine oil changes on Model XYZ aircraft engines.
    
    Footer: Confidential - Property of ABC Company
    Page 1
    
    Page 2
    
    2. SAFETY WARNINGS
    WARNING: Engine must be shut down and cooled before beginning
    CAUTION: Wear protective gloves when handling oil
    
    Footer: Confidential - Property of ABC Company
    Page 2
    """
    
    cleaner = TextCleaner()
    result = cleaner.clean_extracted_text(sample_text)
    
    print(f"Original text length: {len(sample_text)}")
    print(f"Clean text length: {len(result['clean_text'])}")
    print(f"Removed elements: {len(result['removed_elements'])}")
    print(f"Cleaning report: {result['cleaning_report']}")
    print("\nCleaned text:")
    print(result['clean_text'])
    print("\nRemoved elements:")
    for element in result['removed_elements']:
        print(f"  - {element}")
    
    return result['clean_text']

async def test_chunking():
    """Test chunking functionality"""
    print("\nTesting Chunking...")
    
    sample_text = """
    MAINTENANCE MANUAL
    Engine Oil Change Procedure
    
    1. INTRODUCTION
    This manual provides step-by-step instructions for performing routine engine oil changes on Model XYZ aircraft engines.
    Follow all safety procedures and use approved lubricants only.
    
    2. SAFETY WARNINGS
    WARNING: Engine must be shut down and cooled before beginning
    CAUTION: Wear protective gloves when handling oil
    NOTE: Dispose of used oil according to local regulations
    
    3. TOOLS AND EQUIPMENT REQUIRED
    - Oil drain pan (minimum 6 quarts)
    - Socket wrench set
    - Oil filter wrench
    - Funnel
    - Shop rags
    - Approved engine oil (5W-30)
    - New oil filter
    
    4. DRAIN PROCEDURE
    4.1 Position aircraft on level ground
    4.2 Ensure engine is warm but not hot
    4.3 Locate oil drain plug under engine cowling
    4.4 Position drain pan under plug
    4.5 Remove drain plug using socket wrench
    4.6 Allow oil to drain completely (approximately 15 minutes)
    """
    
    chunker = ChunkingStrategy()
    
    # Test planning chunks
    planning_chunks = chunker.create_planning_chunks(sample_text)
    print(f"Planning chunks: {len(planning_chunks)}")
    for i, chunk in enumerate(planning_chunks):
        tokens = chunker.count_tokens(chunk)
        print(f"  Chunk {i+1}: {tokens} tokens")
    
    # Test population chunks
    population_chunks = chunker.create_population_chunks(sample_text)
    print(f"Population chunks: {len(population_chunks)}")
    for i, chunk in enumerate(population_chunks):
        tokens = chunker.count_tokens(chunk)
        print(f"  Chunk {i+1}: {tokens} tokens")
    
    return planning_chunks

async def test_planning(clean_text):
    """Test document planning"""
    print("\nTesting Document Planning...")
    
    try:
        planner = DocumentPlanner()
        planning_result = await planner.analyze_and_plan(clean_text, "Air")
        
        print(f"Planning confidence: {planning_result.get('planning_confidence', 0):.2f}")
        print(f"Modules planned: {len(planning_result.get('planned_modules', []))}")
        print(f"Document summary: {planning_result.get('document_summary', '')}")
        
        for i, module in enumerate(planning_result.get('planned_modules', [])):
            print(f"\nModule {i+1}:")
            print(f"  Title: {module.get('title', 'Unknown')}")
            print(f"  Type: {module.get('type', 'Unknown')}")
            print(f"  Description: {module.get('description', 'No description')}")
            print(f"  Priority: {module.get('priority', 'medium')}")
        
        return planning_result
        
    except Exception as e:
        print(f"Error in planning: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_population(clean_text, planning_result):
    """Test content population"""
    print("\nTesting Content Population...")
    
    if not planning_result or not planning_result.get('planned_modules'):
        print("No modules to populate")
        return
    
    try:
        populator = ContentPopulator()
        
        # Test with first module
        first_module = planning_result['planned_modules'][0]
        print(f"Populating module: {first_module.get('title', 'Unknown')}")
        
        populated = await populator.populate_module(first_module, clean_text, "Air")
        
        print(f"Population status: {populated.get('status', 'unknown')}")
        print(f"Completeness score: {populated.get('completeness_score', 0):.2f}")
        print(f"Relevant chunks found: {populated.get('relevant_chunks_found', 0)}")
        print(f"STE content length: {len(populated.get('ste_content', ''))}")
        
        if populated.get('ste_content'):
            print(f"STE content preview: {populated['ste_content'][:200]}...")
        
        return populated
        
    except Exception as e:
        print(f"Error in population: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main test function"""
    print("Enhanced PDF Processing Debug Test")
    print("=" * 50)
    
    # Test text cleaning
    clean_text = await test_text_cleaning()
    
    # Test chunking
    planning_chunks = await test_chunking()
    
    # Test planning
    planning_result = await test_planning(clean_text)
    
    # Test population
    if planning_result:
        populated = await test_population(clean_text, planning_result)
    
    print("\n" + "=" * 50)
    print("Debug test completed!")

if __name__ == "__main__":
    asyncio.run(main())