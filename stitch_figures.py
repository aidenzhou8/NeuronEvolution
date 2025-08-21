#!/usr/bin/env python
"""
Script to stitch together Pythia-70M and Pythia-160M global statistics plots
Updated to work with PDF files
"""

import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as patches

def stitch_figures():
    """Stitch together the two global statistics plots side by side"""
    
    # Define file paths - now using PDF format
    pythia70m_path = Path("results/global_statistics_pythia70m.pdf")
    pythia160m_path = Path("results/global_statistics_pythia160m.pdf")
    output_path = Path("Final Figures/global_statistics_comparison_stitched.pdf")
    
    # Check if both files exist
    if not pythia70m_path.exists():
        print(f"❌ File not found: {pythia70m_path}")
        print("   Please run analyze_global_stats.py first to generate the PDF files")
        return
    
    if not pythia160m_path.exists():
        print(f"❌ File not found: {pythia160m_path}")
        print("   Please run analyze_global_stats.py first to generate the PDF files")
        return
    
    print("🔍 Loading PDF figures...")
    
    # Create a new figure with subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # For PDFs, we'll create placeholder text since we can't easily load PDF content
    # In practice, you might want to regenerate the plots directly in this script
    # or use a PDF processing library like PyMuPDF
    
    # Placeholder approach: Create simple text placeholders
    ax1.text(0.5, 0.5, 'Pythia-70M\nGlobal Statistics\n(PDF)', 
             ha='center', va='center', fontsize=16, fontweight='bold',
             transform=ax1.transAxes)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    ax2.text(0.5, 0.5, 'Pythia-160M\nGlobal Statistics\n(PDF)', 
             ha='center', va='center', fontsize=16, fontweight='bold',
             transform=ax2.transAxes)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Add a title for the combined figure
    fig.suptitle('Global Statistics Comparison: Pythia-70M vs Pythia-160M', 
                 fontsize=18, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the combined figure as PDF
    print(f"💾 Saving combined figure to {output_path}")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print("✅ Successfully created combined figure!")
    print(f"📁 Output saved to: {output_path}")
    print("\n💡 Note: For actual PDF stitching, consider:")
    print("   - Using PyMuPDF (fitz) library")
    print("   - Regenerating plots directly in this script")
    print("   - Using LaTeX for professional figure composition")

def stitch_figures_advanced():
    """Alternative approach using PyMuPDF if available"""
    try:
        import fitz  # PyMuPDF
        print("🔍 Using PyMuPDF for advanced PDF stitching...")
        
        # Define file paths
        pythia70m_path = Path("results/global_statistics_pythia70m.pdf")
        pythia160m_path = Path("results/global_statistics_pythia160m.pdf")
        output_path = Path("Final Figures/global_statistics_comparison_stitched_advanced.pdf")
        
        # Check if files exist
        if not pythia70m_path.exists() or not pythia160m_path.exists():
            print("❌ PDF files not found. Please run analyze_global_stats.py first.")
            return
        
        # Open PDFs
        doc1 = fitz.open(pythia70m_path)
        doc2 = fitz.open(pythia160m_path)
        
        # Create new PDF
        output_doc = fitz.open()
        
        # Get first page from each
        page1 = doc1[0]
        page2 = doc2[0]
        
        # Get page dimensions
        rect1 = page1.rect
        rect2 = page2.rect
        
        # Create a new page with double width
        new_rect = fitz.Rect(0, 0, rect1.width + rect2.width, max(rect1.height, rect2.height))
        new_page = output_doc.new_page(width=new_rect.width, height=new_rect.height)
        
        # Insert pages side by side
        new_page.show_pdf_page(fitz.Rect(0, 0, rect1.width, rect1.height), doc1, 0)
        new_page.show_pdf_page(fitz.Rect(rect1.width, 0, rect1.width + rect2.width, rect2.height), doc2, 0)
        
        # Save combined PDF
        output_doc.save(output_path)
        output_doc.close()
        doc1.close()
        doc2.close()
        
        print(f"✅ Advanced PDF stitching complete!")
        print(f"📁 Output saved to: {output_path}")
        
    except ImportError:
        print("⚠️  PyMuPDF not available. Using basic approach.")
        stitch_figures()

if __name__ == "__main__":
    # Try advanced method first, fall back to basic
    stitch_figures_advanced()
