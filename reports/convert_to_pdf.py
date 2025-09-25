#!/usr/bin/env python
"""
Convert Jupyter notebooks to PDF for KSCU competition submission
"""

import os
import subprocess
import sys

def convert_notebook_to_pdf(notebook_path, output_path=None):
    """Convert Jupyter notebook to PDF using nbconvert."""

    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return False

    if output_path is None:
        output_path = notebook_path.replace('.ipynb', '.pdf')

    try:
        # Method 1: Try using jupyter nbconvert with LaTeX
        print(f"Converting {notebook_path} to PDF...")
        subprocess.run([
            'jupyter', 'nbconvert',
            '--to', 'pdf',
            '--execute',
            '--no-input',  # Hide code cells for cleaner report
            '--output', output_path.replace('.pdf', ''),
            notebook_path
        ], check=True)
        print(f"✓ Successfully created PDF: {output_path}")
        return True

    except subprocess.CalledProcessError:
        print("LaTeX conversion failed. Trying HTML to PDF method...")

        try:
            # Method 2: Convert via HTML (requires wkhtmltopdf or weasyprint)
            html_path = notebook_path.replace('.ipynb', '.html')

            # First convert to HTML
            subprocess.run([
                'jupyter', 'nbconvert',
                '--to', 'html',
                '--execute',
                '--no-input',
                '--output', html_path.replace('.html', ''),
                notebook_path
            ], check=True)

            # Then convert HTML to PDF using weasyprint
            try:
                import weasyprint
                weasyprint.HTML(filename=html_path).write_pdf(output_path)
                os.remove(html_path)  # Clean up HTML file
                print(f"✓ Successfully created PDF via HTML: {output_path}")
                return True
            except ImportError:
                print("weasyprint not installed. Keeping HTML output.")
                print(f"✓ Created HTML report: {html_path}")
                return True

        except subprocess.CalledProcessError as e:
            print(f"Error during conversion: {e}")
            return False

    except FileNotFoundError:
        print("Jupyter not found. Please ensure Jupyter is installed.")
        print("Run: pip install jupyter nbconvert")
        return False

def create_simplified_pdf():
    """Create a simplified PDF using markdown and pandoc if available."""
    try:
        # Check if pandoc is available
        subprocess.run(['pandoc', '--version'], capture_output=True, check=True)

        # Convert markdown report to PDF
        subprocess.run([
            'pandoc',
            'technical_report.md',
            '-o', 'technical_report_simple.pdf',
            '--pdf-engine=xelatex',
            '-V', 'geometry:margin=1in',
            '-V', 'fontsize=11pt'
        ], check=True)
        print("✓ Created simplified PDF using pandoc")
        return True

    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Pandoc not available for simplified PDF creation")
        return False

def main():
    """Main conversion function."""

    # Change to reports directory
    reports_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(reports_dir)

    print("KSCU Technical Report PDF Converter")
    print("="*40)

    # Convert technical report
    if convert_notebook_to_pdf('technical_report.ipynb'):
        print("\n✓ Technical report PDF ready for submission!")
    else:
        print("\n⚠ PDF conversion failed. Alternative options:")
        print("1. Open the notebook in Jupyter and use File > Download as > PDF")
        print("2. Use the HTML output if generated")
        print("3. Install required dependencies:")
        print("   - For LaTeX method: apt-get install texlive-xetex texlive-fonts-recommended")
        print("   - For HTML method: pip install weasyprint")

        # Try creating a simplified version
        create_simplified_pdf()

    # Check file sizes for submission
    print("\n📊 Report Statistics:")
    for file in ['technical_report.pdf', 'technical_report.html', 'technical_report_simple.pdf']:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # Size in KB
            print(f"  - {file}: {size:.1f} KB")

    print("\n📝 Submission Checklist:")
    print("  ✓ Technical Report (≤6 pages)")
    print("  ⏳ Executive Summary (≤2 pages) - Create separately")
    print("  ✓ Source code and data")
    print("  ✓ README with instructions")

if __name__ == '__main__':
    main()