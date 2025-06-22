import re
import pdfplumber
from pathlib import Path
import PyPDF2
import argparse
import os
import sys
from pdf2image import convert_from_path
from extract_par_text_grant import extractor
from layout_detection_lvlm3 import run_layout_detection
from content_mapping_folders import run_content_mapping
    
def extract_clean_toc(text):
    """
    Extract and clean table of contents from PDF text.
    Removes dot sequences but keeps page numbers with | separator.
    """
    lines = text.split('\n')
    toc_lines = []
    
    # Pattern to match TOC entries with optional numbering
    # Matches: "1.1 Title .................. 123" or "Title .................. 123"
    toc_pattern = r'^(\d+(?:\.\d+)*\.?\s+)?([^.]+?)\.{2,}\s*(\d+)\s*$'
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Match TOC pattern with dots and page numbers
        match = re.match(toc_pattern, line)
        if match:
            section_num = match.group(1) or ''
            title = match.group(2).strip()
            page_num = match.group(3)
            
            # Clean up the title (remove extra spaces)
            title = re.sub(r'\s+', ' ', title)
            
            # Combine section number, title, and page number with | separator
            clean_line = f"{section_num}{title} | {page_num}".strip()
            toc_lines.append(clean_line)
        
        # Also catch lines that might not have dots but look like TOC entries
        elif re.match(r'^\d+(?:\.\d+)*\.?\s+[A-Za-z]', line):
            # Extract page number from end if present
            page_match = re.search(r'\s+(\d+)\s*$', line)
            if page_match:
                page_num = page_match.group(1)
                clean_line = re.sub(r'\s+\d+\s*$', '', line).strip()
                clean_line = f"{clean_line} | {page_num}"
                toc_lines.append(clean_line)
    
    return toc_lines

def read_first_page_pdfplumber(pdf_path):
    """
    Read the first page of PDF using pdfplumber (better text extraction).
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if len(pdf.pages) == 0:
                return None
            
            first_page = pdf.pages[0]
            text = first_page.extract_text()
            return text
    except Exception as e:
        print(f"Error reading PDF with pdfplumber: {e}")
        return None

def read_first_page_pypdf2(pdf_path):
    """
    Read the first page of PDF using PyPDF2.
    """

    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            if len(pdf_reader.pages) == 0:
                return None
            
            first_page = pdf_reader.pages[0]
            text = first_page.extract_text()
            return text
    except Exception as e:
        print(f"Error reading PDF with PyPDF2: {e}")
        return None

def extract_toc_from_pdf_first_page(pdf_path):
    """
    Extract table of contents from the first page of a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
    
    Returns:
        list: Clean TOC entries with page numbers and | separator
    """
    # Check if file exists
    if not Path(pdf_path).exists():
        print(f"File {pdf_path} not found.")
        return []
    
    # Extract text from first page    
    text = read_first_page_pdfplumber(pdf_path)

    if text is None:
        print("Could not extract text from PDF.")
        return []
    
    # Extract clean TOC
    toc_lines = extract_clean_toc(text)
    
    return toc_lines

def print_toc_from_pdf(pdf_path, output_path, save_to_file=True):
    """
    Print clean table of contents from PDF first page and optionally save to file.
    
    Args:
        pdf_path (str): Path to the PDF file
        save_to_file (bool): If True, save TOC to a text file
    """
    print(f"Extracting TOC from: {pdf_path}")
    print("=" * 60)
    
    toc_lines = extract_toc_from_pdf_first_page(pdf_path)
    
    if not toc_lines:
        print("No table of contents found or could not extract text.")
        return
    
    for line in toc_lines:
        print(line)
    
    # Save to text file if requested
    if save_to_file:
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in toc_lines:
                f.write(line + "\n")
        print(f"\nTOC saved to: {output_path}")


def pdf_to_jpg(input_pdf_path, output_folder, num=4, dpi=300):
    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(input_pdf_path, dpi=dpi)

    for i, image in enumerate(images[num-1:], start=num-1):
        image_path = os.path.join(output_folder, f"page_{i + 1}.jpg")
        image.save(image_path, "JPEG")
        print(f"Saved: {image_path}")


def extract_pahe_num(num):

    with open(pdf_file, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        if int(args.num) <= len(pdf_reader.pages):
            pdf_writer = PyPDF2.PdfWriter()
            pdf_writer.add_page(pdf_reader.pages[int(num)-1])
            
            with open(output_pdf, 'wb') as output_file:
                pdf_writer.write(output_file)
            print(f"Extracted page {num} and saved to: {output_pdf}")
            toc_file = output_pdf
        else:
            print(f"Error: PDF has only {len(pdf_reader.pages)} pages, but page {args.num} was requested.")
            toc_file = pdf_file
    
    return toc_file

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract table of contents from a PDF file.')
    #parser.add_argument('pdf_file', type=str, default="/home/vc/Downloads/AMIF-2025-TF2-AG-INTE-04-PATHWAYS_separator_rules-lev-lear-fca_en-5.pdf", help='Path to the PDF file')
    parser.add_argument('--pdf_file', type=str, default="/home/vc/Downloads/GS_grant_sample/AMIF-2025-TF2-AG-INTE-04-PATHWAYS/AMIF-2025-TF2-AG-INTE-04-PATHWAYS_separator_rules-lev-lear-fca_en.pdf", help='Path to the PDF file')
    parser.add_argument('--num', type=str, default=5, help='toc page number to extract')
    parser.add_argument('--dst', type=str, default="output_images_grant_3", help='output folder to save images')
    
    parser.add_argument('--config_path', type=str, default="configs/layout_detection_layoutlmv3_grant_2.yaml", help='config file')
    parser.add_argument('--dst_crops', type=str, default="../cropsAll/", help='output folder to save crops')
    args = parser.parse_args()
    
    pdf_file = args.pdf_file

    
    print("PDF Table of Contents Extractor")
    print("=" * 60)
    
    # Extract the specific page (args.num) from the PDF file and save it as PDF
    output_pdf = os.path.join(args.dst, f"page_{args.num}.pdf")
    os.makedirs(args.dst, exist_ok=True)
    os.makedirs(args.dst_crops, exist_ok=True)
    
    toc_file= extract_pahe_num(args.num)
    print(f"Extracting TOC from: {toc_file}")
    
    print_toc_from_pdf(toc_file, args.dst+"/"+pdf_file[pdf_file.rfind('/')+1:pdf_file.rfind('.')] + "_toc.txt")
    
    print("\n" + "=" * 60)

 
    extractor(
    toc_file=args.dst+"/"+pdf_file[pdf_file.rfind('/')+1:pdf_file.rfind('.')] + "_toc.txt",
    pdf_path=pdf_file, 
    output_dir=args.dst_crops+"extracted_sections"
    )

    
    # Get the parent directory (PDF-Extract-Kit folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Add parent directory to sys.path for imports
    sys.path.append(parent_dir)
    
    # Save current working directory
    original_cwd = os.getcwd()
    
    # Change to parent directory so relative paths in config work
    os.chdir(parent_dir)

    args.dst_crops=args.dst_crops[3:]

    result = run_layout_detection(args.config_path, args.dst_crops)

    # Call the function with arguments
    result = run_content_mapping(
        extracted_files=args.dst_crops+"extracted_titles.txt",
        toc_path="scripts/"+args.dst+"/"+pdf_file[pdf_file.rfind('/')+1:pdf_file.rfind('.')] + "_toc.txt",
        input_dir=args.dst_crops+"text_block",
        output_dir=args.dst_crops+"organized_blocks"
    )

    if result["status"] == "success":
        print("Content mapping completed!")
        print(f"Associations: {result['associations_count']}")
        print(f"Output directory: {result['output_dir']}")
    else:
        print(f"Error occurred: {result['error']}")
