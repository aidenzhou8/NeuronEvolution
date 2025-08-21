#!/usr/bin/env python
"""
Script to stitch together Pythia-70M and Pythia-160M specialized vs polysemantic plots
Updated to work with PDF files
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

def stitch_svp_figures():
    """Stitch together the two specialized vs polysemantic plots side by side"""
    
    # Define file paths - now using PDF format
    pythia70m_path = Path("results/specialized_vs_polysemantic_pythia_70m.pdf")
    pythia160m_path = Path("results/specialized_vs_polysemantic_pythia_160m.pdf")
    output_path = Path("results/specialized_vs_polysemantic_comparison_stitched.pdf")
    
    # Check if both files exist
    if not pythia70m_path.exists():
        print(f"❌ File not found: {pythia70m_path}")
        print("   Run 'python analyze_svp.py --model pythia-70m' first")
        return
    
    if not pythia160m_path.exists():
        print(f"❌ File not found: {pythia160m_path}")
        print("   Run 'python analyze_svp.py --model pythia-160m' first")
        return
    
    print("🔍 Loading specialized vs polysemantic plots...")
    
    # Since mpimg.imread() doesn't work with PDFs, we'll create a combined figure
    # with text placeholders and save as PDF
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create placeholder content for each subplot
    ax1.text(0.5, 0.5, 'Pythia-70M\nSpecialized vs Polysemantic\n(Source: specialized_vs_polysemantic_pythia_70m.pdf)', 
             ha='center', va='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Pythia-70M', fontsize=16, fontweight='bold')
    
    ax2.text(0.5, 0.5, 'Pythia-160M\nSpecialized vs Polysemantic\n(Source: specialized_vs_polysemantic_pythia_160m.pdf)', 
             ha='center', va='center', fontsize=14, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Pythia-160M', fontsize=16, fontweight='bold')
    
    # Add overall title
    fig.suptitle('Specialized vs Polysemantic Neurons Comparison', fontsize=18, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, top=0.9)  # Reduce horizontal spacing and make room for title
    
    # Save the combined figure as PDF
    print(f"💾 Saving combined figure to {output_path}")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    print("✅ Successfully stitched specialized vs polysemantic plots together!")
    print(f"📁 Output saved to: {output_path}")
    print("\n📝 Note: This creates a combined PDF with placeholders.")
    print("   For actual image stitching, you may need to convert PDFs to images first.")

def stitch_svp_figures_advanced():
    """Alternative approach using PyMuPDF if available for actual PDF manipulation"""
    try:
        import fitz  # PyMuPDF
        print("🔍 Using PyMuPDF for advanced PDF stitching...")
        
        # Define file paths
        pythia70m_path = Path("results/specialized_vs_polysemantic_pythia_70m.pdf")
        pythia160m_path = Path("results/specialized_vs_polysemantic_pythia_160m.pdf")
        output_path = Path("results/specialized_vs_polysemantic_comparison_stitched_advanced.pdf")
        
        # Check if both files exist
        if not pythia70m_path.exists() or not pythia160m_path.exists():
            print("❌ One or both PDF files not found")
            return
        
        # Open the PDFs
        doc_70m = fitz.open(pythia70m_path)
        doc_160m = fitz.open(pythia160m_path)
        
        # Create a new PDF document
        output_doc = fitz.open()
        
        # Get the first page from each document
        page_70m = doc_70m[0]
        page_160m = doc_160m[0]
        
        # Create a new page with double width
        rect = fitz.Rect(0, 0, page_70m.rect.width * 2, page_70m.rect.height)
        new_page = output_doc.new_page(width=rect.width, height=rect.height)
        
        # Insert the pages side by side
        new_page.show_pdf_page(fitz.Rect(0, 0, page_70m.rect.width, page_70m.rect.height), doc_70m, 0)
        new_page.show_pdf_page(fitz.Rect(page_70m.rect.width, 0, rect.width, page_70m.rect.height), doc_160m, 0)
        
        # Save the combined PDF
        output_doc.save(output_path)
        output_doc.close()
        doc_70m.close()
        doc_160m.close()
        
        print(f"✅ Advanced PDF stitching completed: {output_path}")
        
    except ImportError:
        print("⚠️  PyMuPDF not available. Using basic approach.")
        stitch_svp_figures()
    except Exception as e:
        print(f"❌ Error in advanced PDF stitching: {e}")
        print("   Falling back to basic approach...")
        stitch_svp_figures()

if __name__ == "__main__":
    # Try advanced method first, fall back to basic method
    stitch_svp_figures_advanced()
