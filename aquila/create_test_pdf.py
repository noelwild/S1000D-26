from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io

def create_test_pdf():
    """Create a test PDF with sample maintenance content for testing"""
    
    # Create a PDF document
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    
    # Page 1: Title and Overview
    p.drawString(50, 750, "AIRCRAFT MAINTENANCE MANUAL")
    p.drawString(50, 730, "Section: Engine Maintenance Procedures")
    p.drawString(50, 710, "Document Type: Maintenance Instructions")
    p.drawString(50, 690, "Classification: Technical Documentation")
    
    p.drawString(50, 650, "OVERVIEW")
    p.drawString(50, 630, "This document provides comprehensive maintenance procedures for aircraft")
    p.drawString(50, 610, "engines including inspection, repair, and replacement procedures.")
    p.drawString(50, 590, "All procedures must be followed according to S1000D standards.")
    
    p.drawString(50, 550, "SAFETY REQUIREMENTS")
    p.drawString(50, 530, "WARNING: Always ensure engine is completely shut down before maintenance.")
    p.drawString(50, 510, "CAUTION: Use proper protective equipment during all procedures.")
    p.drawString(50, 490, "NOTE: Refer to technical specifications for torque values.")
    
    p.showPage()
    
    # Page 2: Engine Inspection Procedure
    p.drawString(50, 750, "ENGINE INSPECTION PROCEDURE")
    p.drawString(50, 730, "Type: Periodic Inspection")
    p.drawString(50, 710, "Frequency: Every 100 flight hours")
    
    p.drawString(50, 670, "PREREQUISITES")
    p.drawString(50, 650, "- Engine must be shut down for at least 30 minutes")
    p.drawString(50, 630, "- Aircraft must be properly grounded")
    p.drawString(50, 610, "- All safety equipment must be available")
    
    p.drawString(50, 570, "REQUIRED TOOLS AND EQUIPMENT")
    p.drawString(50, 550, "- Standard socket wrench set")
    p.drawString(50, 530, "- Torque wrench (0-150 ft-lbs)")
    p.drawString(50, 510, "- Inspection mirror")
    p.drawString(50, 490, "- Flashlight")
    p.drawString(50, 470, "- Clean rags")
    
    p.drawString(50, 430, "INSPECTION STEPS")
    p.drawString(50, 410, "1. Remove engine cowling panels")
    p.drawString(50, 390, "2. Inspect engine mounts for cracks or damage")
    p.drawString(50, 370, "3. Check all visible bolts and connections")
    p.drawString(50, 350, "4. Examine fuel lines for leaks or deterioration")
    p.drawString(50, 330, "5. Inspect electrical connections and wiring")
    p.drawString(50, 310, "6. Check oil level and quality")
    p.drawString(50, 290, "7. Examine exhaust system for damage")
    p.drawString(50, 270, "8. Replace engine cowling panels")
    
    p.showPage()
    
    # Page 3: Oil Change Procedure
    p.drawString(50, 750, "ENGINE OIL CHANGE PROCEDURE")
    p.drawString(50, 730, "Type: Maintenance Procedure")
    p.drawString(50, 710, "Frequency: Every 50 flight hours or 6 months")
    
    p.drawString(50, 670, "MATERIALS REQUIRED")
    p.drawString(50, 650, "- Engine oil (specification: SAE 20W-50)")
    p.drawString(50, 630, "- Oil filter (Part Number: OF-12345)")
    p.drawString(50, 610, "- Drain pan (minimum 6 quart capacity)")
    p.drawString(50, 590, "- New drain plug gasket")
    
    p.drawString(50, 550, "PROCEDURE STEPS")
    p.drawString(50, 530, "1. Warm engine to operating temperature")
    p.drawString(50, 510, "2. Shut down engine and wait 10 minutes")
    p.drawString(50, 490, "3. Position drain pan under oil drain plug")
    p.drawString(50, 470, "4. Remove drain plug and allow oil to drain completely")
    p.drawString(50, 450, "5. Remove old oil filter using filter wrench")
    p.drawString(50, 430, "6. Clean filter mounting surface")
    p.drawString(50, 410, "7. Install new filter with light coat of oil on gasket")
    p.drawString(50, 390, "8. Reinstall drain plug with new gasket")
    p.drawString(50, 370, "9. Add new oil through oil filler opening")
    p.drawString(50, 350, "10. Check oil level with dipstick")
    p.drawString(50, 330, "11. Run engine and check for leaks")
    p.drawString(50, 310, "12. Recheck oil level after shutdown")
    
    p.drawString(50, 270, "EXPECTED RESULTS")
    p.drawString(50, 250, "- Oil level should be between MIN and MAX marks")
    p.drawString(50, 230, "- No oil leaks should be present")
    p.drawString(50, 210, "- Engine should run smoothly without abnormal noises")
    
    p.showPage()
    
    # Page 4: Troubleshooting Guide
    p.drawString(50, 750, "ENGINE TROUBLESHOOTING GUIDE")
    p.drawString(50, 730, "Type: Fault Isolation Procedure")
    
    p.drawString(50, 690, "PROBLEM: Engine will not start")
    p.drawString(50, 670, "POSSIBLE CAUSES:")
    p.drawString(50, 650, "1. Fuel system problems")
    p.drawString(50, 630, "   - Check fuel quantity")
    p.drawString(50, 610, "   - Inspect fuel lines for blockages")
    p.drawString(50, 590, "   - Verify fuel pump operation")
    p.drawString(50, 570, "2. Ignition system problems")
    p.drawString(50, 550, "   - Check spark plugs")
    p.drawString(50, 530, "   - Test ignition coils")
    p.drawString(50, 510, "   - Verify ignition timing")
    p.drawString(50, 490, "3. Electrical system problems")
    p.drawString(50, 470, "   - Check battery voltage")
    p.drawString(50, 450, "   - Inspect wiring connections")
    p.drawString(50, 430, "   - Test starter motor")
    
    p.drawString(50, 390, "PROBLEM: Engine runs rough")
    p.drawString(50, 370, "POSSIBLE CAUSES:")
    p.drawString(50, 350, "1. Fuel contamination")
    p.drawString(50, 330, "2. Dirty air filter")
    p.drawString(50, 310, "3. Worn spark plugs")
    p.drawString(50, 290, "4. Carburetor adjustment needed")
    p.drawString(50, 270, "5. Compression problems")
    
    p.showPage()
    
    # Page 5: Technical Specifications
    p.drawString(50, 750, "TECHNICAL SPECIFICATIONS")
    p.drawString(50, 730, "Type: Reference Information")
    
    p.drawString(50, 690, "ENGINE SPECIFICATIONS")
    p.drawString(50, 670, "Model: Lycoming O-360-A4A")
    p.drawString(50, 650, "Displacement: 361.0 cubic inches")
    p.drawString(50, 630, "Horsepower: 180 HP @ 2700 RPM")
    p.drawString(50, 610, "Compression Ratio: 8.5:1")
    p.drawString(50, 590, "Fuel System: Carburetor")
    p.drawString(50, 570, "Ignition: Dual magneto")
    p.drawString(50, 550, "Oil Capacity: 8 quarts")
    p.drawString(50, 530, "Dry Weight: 325 lbs")
    
    p.drawString(50, 490, "TORQUE SPECIFICATIONS")
    p.drawString(50, 470, "Spark plugs: 30-35 ft-lbs")
    p.drawString(50, 450, "Oil drain plug: 25-30 ft-lbs")
    p.drawString(50, 430, "Oil filter: 15-20 ft-lbs")
    p.drawString(50, 410, "Cylinder head bolts: 35-40 ft-lbs")
    p.drawString(50, 390, "Propeller bolts: 50-55 ft-lbs")
    
    p.drawString(50, 350, "OPERATING LIMITS")
    p.drawString(50, 330, "Maximum RPM: 2700")
    p.drawString(50, 310, "Oil temperature: 100-245°F")
    p.drawString(50, 290, "Oil pressure: 60-90 PSI")
    p.drawString(50, 270, "Fuel pressure: 0.5-8.0 PSI")
    p.drawString(50, 250, "Manifold pressure: 15-29 inches Hg")
    
    p.save()
    
    # Write to file
    buffer.seek(0)
    with open('/app/aquila/test_maintenance_manual.pdf', 'wb') as f:
        f.write(buffer.read())
    
    print("Test PDF created successfully!")

if __name__ == "__main__":
    create_test_pdf()