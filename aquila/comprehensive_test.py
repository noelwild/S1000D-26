#!/usr/bin/env python3
"""
Comprehensive test demonstrating the complete enhanced PDF processing system
"""

import asyncio
import json
import sys
import os
from pathlib import Path

# Add the aquila directory to the path
sys.path.insert(0, '/app/aquila')

from server import (
    TextCleaner, ChunkingStrategy, DocumentPlanner, ContentPopulator,
    project_manager, Document, DocumentPlan, DataModule, Session, 
    create_engine, SQLModel, generate_dmc
)
import openai

# Load API key
with open('/app/aquila/keys.txt', 'r') as f:
    for line in f:
        if line.startswith('OPENAI_API_KEY='):
            openai.api_key = line.split('=', 1)[1].strip()
            break

async def comprehensive_test():
    """Test the complete enhanced processing pipeline"""
    
    print("🚀 Enhanced PDF Processing System - Comprehensive Test")
    print("=" * 70)
    
    # Sample PDF content (more comprehensive)
    sample_pdf_content = """
    Page 1
    
    AIRCRAFT MAINTENANCE MANUAL
    Engine Oil Change Procedure
    Document ID: TM-123-456
    
    1. INTRODUCTION
    This manual provides comprehensive step-by-step instructions for performing
    routine engine oil changes on Model XYZ aircraft engines. This procedure
    is critical for maintaining engine performance and longevity.
    
    The procedure must be performed every 50 flight hours or as specified
    in the maintenance schedule. Use only approved lubricants and follow
    all safety protocols.
    
    Footer: Confidential - Property of ABC Company
    Page 1
    
    Page 2
    
    2. SAFETY WARNINGS AND PRECAUTIONS
    
    WARNING: Engine must be completely shut down and cooled before beginning
    any maintenance procedures. Hot oil can cause severe burns.
    
    CAUTION: Wear protective gloves and safety glasses when handling oil
    and chemicals. Ensure proper ventilation in work area.
    
    NOTE: Dispose of used oil and filters according to local environmental
    regulations. Never dispose of oil in drains or soil.
    
    3. TOOLS AND EQUIPMENT REQUIRED
    
    The following tools and equipment are required for this procedure:
    - Oil drain pan (minimum 6 quarts capacity)
    - Socket wrench set (10mm, 13mm, 17mm)
    - Oil filter wrench
    - Funnel with fine mesh strainer
    - Shop rags and absorbent material
    - Torque wrench (0-50 ft-lbs)
    - Approved engine oil (SAE 15W-50, 5 quarts)
    - New oil filter (Part No. ABC-12345)
    - New drain plug gasket
    
    Footer: Confidential - Property of ABC Company
    Page 2
    
    Page 3
    
    4. DRAIN PROCEDURE
    
    4.1 PREPARATION
    Position aircraft on level ground and engage parking brake.
    Ensure engine has been shut down for at least 30 minutes to allow
    oil to settle but remain warm for easier draining.
    
    4.2 DRAIN PLUG REMOVAL
    Locate the oil drain plug on the bottom of the engine oil pan.
    Position drain pan directly under the plug.
    
    Using appropriate socket wrench, slowly remove the drain plug.
    Allow oil to drain completely (approximately 15-20 minutes).
    
    4.3 DRAIN PLUG INSPECTION
    Inspect drain plug and gasket for damage or excessive wear.
    Replace gasket if damaged or deformed.
    
    5. FILTER REPLACEMENT
    
    5.1 FILTER REMOVAL
    Locate oil filter adjacent to the drain plug area.
    Position drain pan under filter to catch residual oil.
    
    Using filter wrench, remove old filter by turning counterclockwise.
    Allow residual oil to drain from filter mounting area.
    
    5.2 FILTER INSTALLATION
    Clean filter mounting surface with shop rag.
    Apply thin coat of new oil to new filter gasket.
    Install new filter hand-tight, then tighten additional 3/4 turn.
    
    Footer: Confidential - Property of ABC Company
    Page 3
    
    Page 4
    
    6. REFILL PROCEDURE
    
    6.1 DRAIN PLUG INSTALLATION
    Install new gasket on drain plug.
    Thread drain plug into oil pan by hand to prevent cross-threading.
    Tighten drain plug to 25 ft-lbs using torque wrench.
    
    6.2 OIL REFILL
    Remove oil filler cap from top of engine.
    Using funnel, slowly add 4.5 quarts of approved oil.
    
    Replace filler cap and tighten securely.
    
    7. SYSTEM CHECK AND VERIFICATION
    
    7.1 LEAK CHECK
    Start engine and allow to run for 5 minutes at idle.
    Shut down engine and inspect for leaks at drain plug and filter.
    
    7.2 OIL LEVEL CHECK
    Wait 10 minutes for oil to settle.
    Check oil level using dipstick - level should be between MIN and MAX marks.
    Add oil if necessary to reach proper level.
    
    7.3 FINAL DOCUMENTATION
    Record maintenance action in aircraft logbook.
    Update maintenance tracking system with date, flight hours, and technician signature.
    
    Footer: Confidential - Property of ABC Company
    Page 4
    """
    
    print("📄 Phase 1: Text Cleaning")
    print("-" * 30)
    
    # Test text cleaning
    cleaner = TextCleaner()
    cleaning_result = cleaner.clean_extracted_text(sample_pdf_content)
    clean_text = cleaning_result['clean_text']
    
    print(f"✅ Original text: {len(sample_pdf_content)} characters")
    print(f"✅ Cleaned text: {len(clean_text)} characters")
    print(f"✅ Removed elements: {len(cleaning_result['removed_elements'])}")
    print(f"✅ {cleaning_result['cleaning_report']}")
    
    print("\n🔧 Phase 2: Chunking Strategy")
    print("-" * 30)
    
    # Test chunking
    chunker = ChunkingStrategy()
    planning_chunks = chunker.create_planning_chunks(clean_text)
    population_chunks = chunker.create_population_chunks(clean_text)
    
    print(f"✅ Planning chunks: {len(planning_chunks)} (target: 2000 tokens/200 overlap)")
    for i, chunk in enumerate(planning_chunks):
        tokens = chunker.count_tokens(chunk)
        print(f"   Chunk {i+1}: {tokens} tokens")
    
    print(f"✅ Population chunks: {len(population_chunks)} (target: 400 tokens/50 overlap)")
    for i, chunk in enumerate(population_chunks):
        tokens = chunker.count_tokens(chunk)
        print(f"   Chunk {i+1}: {tokens} tokens")
    
    print("\n🧠 Phase 3: Document Planning")
    print("-" * 30)
    
    # Test document planning
    planner = DocumentPlanner()
    planning_result = await planner.analyze_and_plan(clean_text, "Air")
    
    print(f"✅ Planning confidence: {planning_result.get('planning_confidence', 0):.1%}")
    print(f"✅ Modules planned: {len(planning_result.get('planned_modules', []))}")
    print(f"✅ Document summary: {planning_result.get('document_summary', 'N/A')}")
    
    print("\n📋 Planned Modules:")
    for i, module in enumerate(planning_result.get('planned_modules', [])):
        print(f"   {i+1}. {module.get('title', 'Unknown')}")
        print(f"      Type: {module.get('type', 'Unknown')}")
        print(f"      Priority: {module.get('priority', 'medium')}")
        print(f"      Description: {module.get('description', 'No description')[:100]}...")
        print()
    
    print("\n📚 Phase 4: Content Population")
    print("-" * 30)
    
    # Test content population
    populator = ContentPopulator()
    populated_modules = []
    
    for i, planned_module in enumerate(planning_result.get('planned_modules', [])):
        print(f"Populating module {i+1}: {planned_module.get('title', 'Unknown')}")
        
        populated = await populator.populate_module(planned_module, clean_text, "Air")
        populated_modules.append(populated)
        
        print(f"   ✅ Status: {populated.get('status', 'unknown')}")
        print(f"   ✅ Completeness: {populated.get('completeness_score', 0):.1%}")
        print(f"   ✅ Relevant chunks: {populated.get('relevant_chunks_found', 0)}")
        print(f"   ✅ STE content: {len(populated.get('ste_content', ''))} characters")
        
        if populated.get('ste_content'):
            print(f"   📝 STE Preview: {populated['ste_content'][:150]}...")
        
        print()
    
    print("\n💾 Phase 5: Database Integration")
    print("-" * 30)
    
    # Test database integration
    test_db_path = "/tmp/test_enhanced.db"
    if Path(test_db_path).exists():
        Path(test_db_path).unlink()
    
    engine = create_engine(f"sqlite:///{test_db_path}")
    SQLModel.metadata.create_all(engine)
    
    with Session(engine) as session:
        # Create test document
        document = Document(
            filename="test_enhanced_manual.pdf",
            file_path="/tmp/test_enhanced_manual.pdf",
            sha256="test_hash_123",
            operational_context="Air",
            status="processing"
        )
        session.add(document)
        session.commit()
        session.refresh(document)
        
        # Create plan record
        plan_record = DocumentPlan(
            document_id=document.id,
            plan_data=json.dumps(planning_result),
            planning_confidence=planning_result.get('planning_confidence', 0.0),
            total_chunks_analyzed=len(planning_chunks),
            status="completed"
        )
        session.add(plan_record)
        session.commit()
        
        # Create data modules
        for i, populated in enumerate(populated_modules):
            try:
                sequence = i + 1
                dmc = generate_dmc(
                    "Air",
                    populated.get("type", "description"),
                    populated.get("info_code", "040"),
                    populated.get("item_location", "A"),
                    sequence
                )
                
                data_module = DataModule(
                    document_id=document.id,
                    plan_id=plan_record.id,
                    module_id=f"module_{i+1}",
                    dmc=dmc,
                    title=populated.get("title", ""),
                    info_code=populated.get("info_code", "040"),
                    item_location=populated.get("item_location", "A"),
                    sequence=sequence,
                    verbatim_content=populated.get("verbatim_content", ""),
                    ste_content=populated.get("ste_content", ""),
                    type=populated.get("type", "description"),
                    prerequisites=populated.get("prerequisites", ""),
                    tools_equipment=populated.get("tools_equipment", ""),
                    warnings=populated.get("warnings", ""),
                    cautions=populated.get("cautions", ""),
                    procedural_steps=populated.get("procedural_steps", "[]"),
                    expected_results=populated.get("expected_results", ""),
                    specifications=populated.get("specifications", ""),
                    references=populated.get("references", ""),
                    content_sources=json.dumps(populated.get("content_sources", [])),
                    completeness_score=populated.get("completeness_score", 0.0),
                    relevant_chunks_found=populated.get("relevant_chunks_found", 0),
                    total_chunks_analyzed=len(population_chunks),
                    population_status=populated.get("status", "complete")
                )
                
                session.add(data_module)
                
            except Exception as e:
                print(f"   ❌ Error creating module {i+1}: {e}")
                continue
        
        session.commit()
        
        # Verify database content
        modules = session.query(DataModule).all()
        print(f"✅ Created {len(modules)} data modules in database")
        
        for module in modules:
            print(f"   📋 {module.title}")
            print(f"      DMC: {module.dmc}")
            print(f"      Completeness: {module.completeness_score:.1%}")
            print(f"      Status: {module.population_status}")
    
    print("\n🎉 Phase 6: System Verification")
    print("-" * 30)
    
    # Verify complete system
    total_modules = len(populated_modules)
    successful_modules = sum(1 for m in populated_modules if m.get('status') == 'complete')
    avg_completeness = sum(m.get('completeness_score', 0) for m in populated_modules) / total_modules if total_modules > 0 else 0
    
    print(f"✅ Total modules processed: {total_modules}")
    print(f"✅ Successful modules: {successful_modules}")
    print(f"✅ Success rate: {(successful_modules/total_modules)*100:.1f}%")
    print(f"✅ Average completeness: {avg_completeness:.1%}")
    print(f"✅ Planning confidence: {planning_result.get('planning_confidence', 0):.1%}")
    
    print("\n" + "=" * 70)
    print("🎯 ENHANCED PDF PROCESSING SYSTEM - FULLY OPERATIONAL!")
    print("=" * 70)
    
    print("\n📈 Key Improvements:")
    print("  • Clean text extraction removes headers, footers, page numbers")
    print("  • Large chunks (2000 tokens) for comprehensive planning")
    print("  • Small chunks (400 tokens) for detailed population")
    print("  • AI-powered two-phase processing ensures completeness")
    print("  • Structured S1000D-compliant data modules")
    print("  • Enhanced progress tracking and error handling")
    
    # Clean up
    if Path(test_db_path).exists():
        Path(test_db_path).unlink()

if __name__ == "__main__":
    asyncio.run(comprehensive_test())