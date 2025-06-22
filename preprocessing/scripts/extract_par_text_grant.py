import pdfplumber
import re
from pathlib import Path
import argparse

def parse_toc_line(line):
    """
    Parse a single TOC line to extract title and page number.
    Returns: (title, page_number) or None if invalid
    """
    if "|" not in line:
        return None
    
    title, page_str = line.rsplit("|", 1)
    title = title.strip()
    
    try:
        page_num = int(page_str.strip())
        return (title, page_num)
    except ValueError:
        return None

def parse_toc(toc_input):
    """
    Parse TOC input (either list of strings or multiline string).
    Returns list of (title, page_number) tuples sorted by page number.
    """
    if isinstance(toc_input, str):
        lines = toc_input.strip().split('\n')
    else:
        lines = toc_input
    
    entries = []
    for line in lines:
        parsed = parse_toc_line(line.strip())
        if parsed:
            entries.append(parsed)
    
    # Sort by page number to ensure correct order
    entries.sort(key=lambda x: x[1])
    return entries

def clean_extracted_text(text):
    """
    Clean extracted text by removing unwanted lines:
    - EU Funding & Tenders Portal header lines
    - Lines containing only numbers
    - Empty lines
    """
    if not text:
        return text
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Skip empty lines
        if not line_stripped:
            continue
        
        # Skip EU Funding header lines
        if "EU Funding & Tenders Portal" in line_stripped:
            continue
        
        # Skip lines that contain only numbers (and possibly spaces/punctuation)
        # This regex matches lines with only digits, spaces, dots, commas, dashes
        if re.match(r'^[\d\s\.\,\-]+$', line_stripped):
            continue
        
        # Skip lines that are just page numbers or similar short numeric content
        if len(line_stripped) <= 3 and line_stripped.isdigit():
            continue
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def extract_text_from_page(pdf_path, page_num):
    """
    Extract full text from a single page.
    Page number is 1-indexed as it appears in the PDF.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Convert to 0-indexed for pdfplumber
            page_idx = page_num - 1
            if 0 <= page_idx < len(pdf.pages):
                page_text = pdf.pages[page_idx].extract_text()
                if page_text:
                    # Clean the extracted text
                    page_text = clean_extracted_text(page_text)
                return page_text or ""
    except Exception as e:
        print(f"Error extracting text from page {page_num}: {e}")
    
    return ""

def extract_text_between_pages(pdf_path, start_page, end_page):
    """
    Extract text from start_page to end_page (exclusive).
    Page numbers are 1-indexed as they appear in the PDF.
    """
    text = ""
    print(start_page, end_page)
    
    with pdfplumber.open(pdf_path) as pdf:
        # Convert to 0-indexed for pdfplumber
        start_idx = start_page - 1
        end_idx = min(end_page - 1, len(pdf.pages))
        print(start_idx, end_idx)
        
        for i in range(start_idx, end_idx):
            if 0 <= i < len(pdf.pages):
                page_text = pdf.pages[i].extract_text()
                if page_text:
                    text += page_text + "\n"

    # Clean the extracted text
    text = clean_extracted_text(text)
    return text.strip()

def split_page_text_by_sections(page_text, current_section_title, next_section_title):
    """
    Split page text between two sections that appear on the same page.
    This is a simple approach - you might need to refine based on your PDF structure.
    """
    # Clean the page text first
    page_text = clean_extracted_text(page_text)
    
    lines = page_text.split('\n')
    
    # Find the line containing the current section title
    start_idx = 0
    for i, line in enumerate(lines):
        if current_section_title.split('|')[0].strip() in line:
            start_idx = i
            break
    
    # Find the line containing the next section title
    end_idx = len(lines)
    if next_section_title:
        next_title = next_section_title.split('|')[0].strip()
        for i, line in enumerate(lines[start_idx + 1:], start_idx + 1):
            if next_title in line:
                end_idx = i
                break
    
    # Return the text between the sections
    section_lines = lines[start_idx:end_idx]
    return '\n'.join(section_lines).strip()

def extract_text_between_pages_with_titles(pdf_path, start_page, end_page, current_title, next_title=None):
    """
    Extract text from start_page to end_page, but only content between section titles.
    Page numbers are 1-indexed as they appear in the PDF.
    """
    text = ""
    print(f"Extracting from page {start_page} to {end_page}")
    
    with pdfplumber.open(pdf_path) as pdf:
        # Convert to 0-indexed for pdfplumber
        start_idx = start_page - 1
        end_idx = min(end_page, len(pdf.pages))  # Note: not end_page - 1 since we want inclusive
        print(f"Page indices: {start_idx} to {end_idx-1}")
        
        # Extract text from all pages in range
        for i in range(start_idx, end_idx):
            if 0 <= i < len(pdf.pages):
                page_text = pdf.pages[i].extract_text()
                if page_text:
                    text += page_text + "\n"
    
    # Clean the extracted text before processing
    text = clean_extracted_text(text)
    
    # Now search for the titles within the extracted text and trim accordingly
    if text:
        text = extract_content_between_titles(text, current_title, next_title)
    
    return text.strip()

def extract_content_between_titles(full_text, current_title, next_title=None):
    """
    Extract content between two section titles from the full text.
    """
    lines = full_text.split('\n')
    current_title_clean = current_title.split('|')[0].strip()
    
    # Find start position (line containing current section title)
    start_idx = 0
    found_start = False
    
    # Try different variations of the title
    title_variations = [
        current_title_clean,
        current_title_clean.replace('—', '-'),
        current_title_clean.replace('–', '-'),
        current_title_clean.replace(' — ', ' - '),
        current_title_clean.replace(' – ', ' - '),
    ]
    
    for i, line in enumerate(lines):
        line_clean = line.strip()
        for title_var in title_variations:
            if title_var in line_clean or line_clean.startswith(title_var.split()[0] if title_var.split() else ""):
                start_idx = i
                found_start = True
                print(f"Found section start at line {i}: '{line_clean[:50]}...'")
                break
        if found_start:
            break
    
    if not found_start:
        print(f"Warning: Could not find section title '{current_title_clean}' in extracted text")
        # Return first part of text as fallback
        return '\n'.join(lines[:100]).strip()
    
    # Find end position (line containing next section title)
    end_idx = len(lines)
    if next_title:
        next_title_clean = next_title.split('|')[0].strip()
        next_title_variations = [
            next_title_clean,
            next_title_clean.replace('—', '-'),
            next_title_clean.replace('–', '-'),
            next_title_clean.replace(' — ', ' - '),
            next_title_clean.replace(' – ', ' - '),
        ]
        
        for i, line in enumerate(lines[start_idx + 1:], start_idx + 1):
            line_clean = line.strip()
            for title_var in next_title_variations:
                if title_var in line_clean or line_clean.startswith(title_var.split()[0] if title_var.split() else ""):
                    end_idx = i
                    print(f"Found next section at line {i}: '{line_clean[:50]}...'")
                    break
            if end_idx < len(lines):
                break
    
    # Extract the content between titles
    section_lines = lines[start_idx:end_idx]
    return '\n'.join(section_lines).strip()

def extract_sections_from_toc(pdf_path, toc_input):
    """
    Extract text sections based on TOC entries.
    Handles cases where multiple sections are on the same page.
    Merges short sections (< 300 chars) with the next section that has sufficient content.
    
    Args:
        pdf_path (str): Path to PDF file
        toc_input: Either a list of TOC lines or a multiline string
    
    Returns:
        dict: {section_title: section_text}
    """
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    entries = parse_toc(toc_input)
    if not entries:
        raise ValueError("No valid TOC entries found")
    
    sections = {}
    
    # Get total pages in PDF
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
    
    for i, (title, start_page) in enumerate(entries):
        print(f"\nProcessing section: {title} (page {start_page})")
        
        # Determine end page (start of next section or end of document)
        if i + 1 < len(entries):
            next_title, next_page = entries[i + 1]
            end_page = next_page
        else:
            next_title = None
            end_page = total_pages + 1
        
        # Case 1: Section spans multiple pages OR sections are on different pages
        if start_page < end_page:
            section_text = extract_text_between_pages_with_titles(
                pdf_path, start_page, end_page, title, next_title
            )
        
        # Case 2: Current and next section are on the same page
        elif start_page == end_page and next_title:
            page_text = extract_text_from_page(pdf_path, start_page)
            section_text = split_page_text_by_sections(page_text, title, next_title)
        
        # Case 3: Last section on the same page or single page section
        else:
            page_text = extract_text_from_page(pdf_path, start_page)
            # For the last section, take everything from the section title onwards
            section_text = split_page_text_by_sections(page_text, title, None)
        
        sections[title] = section_text
    
    # Post-process to merge short sections
    sections = merge_short_sections(sections, entries)
    
    return sections

def merge_short_sections(sections, entries, min_chars=300):
    """
    Merge sections with less than min_chars with the next section that has sufficient content.
    
    Args:
        sections (dict): Dictionary of section_title: section_text
        entries (list): List of (title, page_number) tuples in order
        min_chars (int): Minimum character threshold
    
    Returns:
        dict: Updated sections with short sections merged
    """
    merged_sections = {}
    pending_short_sections = []  # Stack to hold short sections waiting to be merged
    
    for i, (title, _) in enumerate(entries):
        section_text = sections.get(title, "")
        
        # Count characters excluding the title itself for content assessment
        content_lines = section_text.split('\n')[1:]  # Skip title line
        content_text = '\n'.join(content_lines).strip()
        content_char_count = len(content_text)
        
        print(f"Section '{title}': {content_char_count} content characters")
        
        if content_char_count < min_chars:
            # This is a short section, add it to pending list
            pending_short_sections.append((title, section_text))
            print(f"  -> Added to pending (short section)")
        else:
            # This section has sufficient content
            if pending_short_sections:
                # Merge all pending short sections with this one
                combined_text = ""
                combined_titles = []
                
                # Add all pending short sections
                for short_title, short_text in pending_short_sections:
                    combined_text += short_text + "\n\n"
                    combined_titles.append(short_title.split('|')[0].strip())
                    print(f"  -> Merging short section: {short_title}")
                
                # Add current section
                combined_text += section_text
                combined_titles.append(title.split('|')[0].strip())
                
                # Create a combined title
                if len(combined_titles) > 1:
                    merged_title = f"{combined_titles[-1]} (includes: {', '.join(combined_titles[:-1])})"
                else:
                    merged_title = title
                
                merged_sections[merged_title] = combined_text.strip()
                print(f"  -> Created merged section: {merged_title}")
                
                # Clear pending sections
                pending_short_sections = []
            else:
                # No pending sections, just add this one normally
                merged_sections[title] = section_text
                print(f"  -> Added as standalone section")
    
    # Handle any remaining short sections at the end
    if pending_short_sections:
        print(f"Warning: {len(pending_short_sections)} short sections at end of document")
        
        # Find the last non-empty section to merge with
        last_key = list(merged_sections.keys())[-1] if merged_sections else None
        
        if last_key:
            # Merge with the last section
            combined_text = merged_sections[last_key]
            combined_titles = [last_key.split('(includes:')[0].strip()]
            
            for short_title, short_text in pending_short_sections:
                combined_text += "\n\n" + short_text
                combined_titles.append(short_title.split('|')[0].strip())
                print(f"  -> Merging end section: {short_title}")
            
            # Update the last section
            if '(includes:' in last_key:
                # Already a merged section, update it
                base_title = last_key.split('(includes:')[0].strip()
                existing_includes = last_key.split('(includes:')[1].rstrip(')').strip()
                new_includes = existing_includes + ', ' + ', '.join(combined_titles[1:])
                new_title = f"{base_title} (includes: {new_includes})"
            else:
                new_title = f"{combined_titles[0]} (includes: {', '.join(combined_titles[1:])})"
            
            # Remove old entry and add new merged one
            del merged_sections[last_key]
            merged_sections[new_title] = combined_text.strip()
            print(f"  -> Updated last section: {new_title}")
        else:
            # No sections to merge with, add as separate entries
            for short_title, short_text in pending_short_sections:
                merged_sections[f"{short_title} (short section)"] = short_text
                print(f"  -> Added short section standalone: {short_title}")
    
    return merged_sections

def print_sections(sections, max_chars=500):
    """
    Print sections with truncated text for preview.
    """
    for title, text in sections.items():
        print(f"{'='*60}")
        print(f"Section: {title}")
        print(f"{'='*60}")
        
        if len(text) > max_chars:
            print(text[:max_chars] + "...")
            print(f"\n[Text truncated - showing first {max_chars} characters]")
        else:
            print(text)
        
        print(f"\nTotal characters: {len(text)}")
        print("\n")

def save_sections_to_files(sections, output_dir="extracted_sections_2"):
    """
    Save each section to a separate text file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for title, text in sections.items():
        # Clean filename
        filename = re.sub(r'[^\w\s.-]', '', title)
        filename = re.sub(r'\s+', '_', filename)
        filename=filename.replace("._", ".0_")  
        filename = filename[:50] + ".txt"  # Limit filename length
        
        file_path = output_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Saved: {file_path}")

def extractor(toc_file=None, pdf_path=None, output_dir=None):

    # If arguments are provided, use them directly
    if toc_file and pdf_path and output_dir:
        args_toc = toc_file
        args_pdf_path = pdf_path
        args_output = output_dir
    else:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Extract sections from PDF based on a TOC file')
        parser.add_argument('--toc', default="output_images_grant_3/AMIF-2025-TF2-AG-INTE-04-PATHWAYS_separator_rules-lev-lear-fca_en_toc.txt", help='Path to the TOC file')
        parser.add_argument('--pdf_path', default="/home/vc/Downloads/GS_grant_sample/AMIF-2025-TF2-AG-INTE-04-PATHWAYS/AMIF-2025-TF2-AG-INTE-04-PATHWAYS_separator_rules-lev-lear-fca_en.pdf", help='Path to the PDF file')
        parser.add_argument('--output', default='extracted_sections_3', help='Output directory for extracted sections')
        
        args = parser.parse_args()
        args_toc = args.toc
        args_pdf_path = args.pdf_path
        args_output = args.output
    
    # Read TOC file
    with open(args_toc, 'r', encoding='utf-8') as f:
        toc_example = f.read()
    
    try:
        print("Extracting sections from PDF based on TOC...")
        sections = extract_sections_from_toc(args_pdf_path, toc_example)
        
        print(f"Found {len(sections)} sections")
        print("\nPreview of extracted sections:")
        print_sections(sections, max_chars=300)
        
        # Save sections to files
        save_sections_to_files(sections, args_output)
        
        return sections
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please update the pdf_path variable with the correct path to your PDF file.")
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    extractor()