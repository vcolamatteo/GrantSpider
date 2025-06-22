# import myUtils

# list_="/home/vc/Downloads/AMIF-2025-TF2-AG-INTE-04-PATHWAYS_separator_om_en-4.toc.txt"
# lines=myUtils.readLines(list_)

# for i in range(0,len(lines)):
#     lines[i]=lines[i][:lines[i].find(" |")]


# folders=myUtils.getFolderList("/home/vc/Theory/NLP2/RAG/AgenticRAG/myRAg/PDF-Extract-Kit/cropsAll/organized_blocks")[0]
# for f in range(0, len(folders)):
#     fname=folders[f][folders[f].rfind("/")+1:]
#     #print(folders[f])
    
#     if not fname in lines:
#         print("gggggggggg")
# ffff

import os
import re
import shutil
from difflib import get_close_matches
import argparse
import sys
# INPUT_DIR = "/home/vc/Theory/NLP2/RAG/AgenticRAG/myRAg/PDF-Extract-Kit/cropsAll/text_block"
# OUTPUT_DIR = "/home/vc/Theory/NLP2/RAG/AgenticRAG/myRAg/PDF-Extract-Kit/cropsAll/organized_blocks"


# INPUT_DIR = "/home/vc/Theory/NLP2/RAG/AgenticRAG/myRAg/PDF-Extract-Kit/cropsAll2/text_block"
# OUTPUT_DIR = "/home/vc/Theory/NLP2/RAG/AgenticRAG/myRAg/PDF-Extract-Kit/cropsAll2/organized_blocks"

def parse_doc(doc_path):
    with open(doc_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    items = []
    for line in lines:
        match = re.match(r'\s*(.*?)\s*\|\s*(\d+)', line)
        if match:
            title = match.group(1).strip()
            page = int(match.group(2))
            # Updated regex to capture the dot for chapters and full paragraph numbers
            number_match = re.match(r'^(\d+\.(?:\d+)?)', title)
            number = number_match.group(1) if number_match else None
            items.append({'title': title, 'page': page, 'number': number})
    return items

def is_chapter(number):
    return number and re.match(r'^\d+\.$', number)

def is_paragraph(number):
    return number and re.match(r'^\d+\.\d+$', number)

def build_structure(doc_a):
    structure = []
    last_chapter = None
    for item in doc_a:
        if is_chapter(item['number']):
            last_chapter = item['title']
            structure.append({'type': 'chapter', 'title': item['title'], 'page': item['page'], 'number': item['number']})
        elif is_paragraph(item['number']):
            structure.append({'type': 'paragraph', 'title': item['title'], 'page': item['page'], 'number': item['number'], 'chapter': last_chapter})
        else:
            print(item)
            print("Unknown item type:", item['title'])
            exit()
    return structure

def detect_file_type(title):
    """
    Detect if a title represents a chapter or paragraph based on naming pattern
    Returns: 'chapter', 'paragraph', or 'subparagraph'
    """
    # Remove the leading number and underscore (e.g., "1_1._..." -> "1._...")
    title_without_prefix = re.sub(r'^\d+_', '', title)

    # Check for chapter pattern: starts with number followed by dot and underscore (e.g., "1._...")
    if re.match(r'^\d+\._', title_without_prefix):
        return 'chapter'
    
    # Check for paragraph pattern: starts with number.number and underscore (e.g., "1.1_...")
    if re.match(r'^\d+\.\d+_', title_without_prefix):
        return 'paragraph'
    
    # Check for nested paragraph pattern: starts with number_number.number (e.g., "3_1.1_...")
    if re.match(r'^\d+_\d+\.\d+_', title_without_prefix):
        return 'paragraph'
    
    # Default to subparagraph for other cases
    return 'subparagraph'

def extract_chapter_number(title):
    """Extract chapter number from title like '1_1._My_Area' -> '1.'"""
    title_without_prefix = re.sub(r'^\d+_', '', title)
    match = re.match(r'^(\d+)\._', title_without_prefix)
    return match.group(1) + '.' if match else None

def extract_paragraph_number(title):
    """Extract paragraph number from title like '3_1.1_EU_Login' -> '1.1'"""
    title_without_prefix = re.sub(r'^\d+_', '', title)
    
    # Try pattern like "1.1_..."
    match = re.match(r'^(\d+\.\d+)_', title_without_prefix)
    if match:
        return match.group(1)
    
    # Try pattern like "3_1.1_..." (already removed first prefix, so this is "3_1.1_...")
    match = re.match(r'^(\d+)_(\d+\.\d+)_', title_without_prefix)
    if match:
        return match.group(2)
    
    return None

def associate(doc_b, doc_a_struct):
    associations = []
    for i, item in enumerate(doc_b):
        num = item['number']
        if is_chapter(num) or is_paragraph(num):
            continue  # Already part of A
        
        # Detect what type of file this is
        file_type = detect_file_type(item['title'])
        nearest = None
        
        if file_type == 'chapter':
            # Find matching chapter in doc_a_struct
            chapter_num = extract_chapter_number(item['title'])
            if chapter_num:
                for struct_item in doc_a_struct:
                    if struct_item['type'] == 'chapter' and struct_item['number'] == chapter_num:
                        # Check if this association makes sense given previous associations
                        if associations and is_valid_association(item, struct_item, associations[-1]):
                            nearest = struct_item
                            break
                        elif not associations:  # First item, no previous association to check
                            nearest = struct_item
                            break
        
        elif file_type == 'paragraph':
            # Find matching paragraph in doc_a_struct
            paragraph_num = extract_paragraph_number(item['title'])
            if paragraph_num:
                for struct_item in doc_a_struct:
                    if struct_item['type'] == 'paragraph' and struct_item['number'] == paragraph_num:
                        # Check if this association makes sense given previous associations
                        if associations and is_valid_association(item, struct_item, associations[-1]):
                            nearest = struct_item
                            break
                        elif not associations:  # First item, no previous association to check
                            nearest = struct_item
                            break
        
        # If no specific match found, use original logic (nearest by page)
        if not nearest:
            for struct_item in reversed(doc_a_struct):
                if struct_item['page'] <= item['page']:
                    nearest = struct_item
                    break
        
        associations.append({
            'sub_title': item['title'],
            'sub_page': item['page'],
            'associated_to': nearest['title'] if nearest else None,
            'associated_type': nearest['type'] if nearest else None,
            'associated_page': nearest['page'] if nearest else None
        })
    return associations

def is_valid_association(current_item, proposed_struct_item, previous_association):
    """
    Check if associating current_item to proposed_struct_item makes sense
    given the previous association. Prevents jumping too far back.
    """
    if not previous_association['associated_page']:
        return True
    
    # Current item should not jump to a much earlier page than the previous association
    # Allow some flexibility (e.g., 5 pages back) but prevent major jumps
    page_difference = previous_association['associated_page'] - proposed_struct_item['page']
    
    # If the proposed association is more than 10 pages before the previous one,
    # and the current item is close to the previous item's page, reject it
    if (page_difference > 10 and 
        abs(current_item['page'] - previous_association['sub_page']) <= 5):
        return False
    
    return True

def move_files(associations):    

    for assoc in associations:
        sub_title = assoc['sub_title']
        paragraph_title = assoc['associated_to']
        if not paragraph_title:
            continue

        folder_name = re.sub(r'[^\w\s.-]', '', paragraph_title).strip()#.replace(" ", "_")
        target_folder = os.path.join(OUTPUT_DIR, folder_name)
        
        os.makedirs(target_folder, exist_ok=True)

        # Find files starting with the sub_title index
        sub_index_match = re.match(r'^(\d+)_', sub_title)
        if not sub_index_match:
            continue

        prefix = sub_index_match.group(1) + "_"
        for filename in os.listdir(INPUT_DIR):
            if filename.startswith(prefix) and filename.endswith(".txt"):
                source_path = os.path.join(INPUT_DIR, filename)
                dest_path = os.path.join(target_folder, filename)
                shutil.copy(source_path, dest_path)
                print(f"Moved {filename} -> {folder_name}/")


def read_toc_entries(toc_path):
    """Extract all titles from the TOC file (text before ' |')"""
    entries = []
    with open(toc_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if ' | ' in line:
                title = line.split(' | ')[0].strip()
                entries.append(title)
    return entries

def normalize_folder_name(folder_name):
    """Normalize folder name for comparison (remove special chars that might be stripped)"""
    # Remove characters that are typically removed when creating folder names
    normalized = re.sub(r'[^\w\s.-]', '', folder_name).strip()
    return normalized

def find_best_match(folder_name, toc_entries, threshold=0.6):
    """Find the best matching TOC entry for a folder name"""
    # Try exact match first
    #normalized_folder = normalize_folder_name(folder_name)
    
    for entry in toc_entries:
        #normalized_entry = normalize_folder_name(entry)
        if folder_name == entry:
            return entry, 1.0  # Exact match
    
    # Try fuzzy matching
    matches = get_close_matches(folder_name, 
                              [normalize_folder_name(entry) for entry in toc_entries], 
                              n=1, cutoff=threshold)
        
    if matches:
        # Find the original entry that corresponds to the normalized match
        normalized_match = matches[0]
        for entry in toc_entries:
            if normalize_folder_name(entry) == normalized_match:
                return entry, threshold
    
    return None, 0.0

def rename_folders_to_match_toc(organized_blocks_path, toc_path):
    """Main function to rename folders to match TOC entries"""
    
    if not os.path.exists(organized_blocks_path):
        print(f"Error: Path {organized_blocks_path} does not exist!")
        return
    
    if not os.path.exists(toc_path):
        print(f"Error: TOC file {toc_path} does not exist!")
        return
    
    # Read TOC entries
    toc_entries = read_toc_entries(toc_path)
    print(f"Found {len(toc_entries)} entries in TOC file")
    
    # Get all folders in the organized_blocks directory
    folders = [f for f in os.listdir(organized_blocks_path) 
              if os.path.isdir(os.path.join(organized_blocks_path, f))]
    
    print(f"Found {len(folders)} folders to process")
    print("-" * 80)
    
    renamed_count = 0
    exact_matches = 0
    
    for folder_name in folders:
        folder_path = os.path.join(organized_blocks_path, folder_name)
        
        # Find best match
        best_match, score = find_best_match(folder_name, toc_entries)
        print(best_match, score)
        
        if best_match:
            if score == 1.0:
                print(f"✓ EXACT MATCH: '{folder_name}'")
                exact_matches += 1
            else:
                # Create new folder name (normalize for filesystem)
                #new_folder_name = normalize_folder_name(best_match)
                new_folder_path = os.path.join(organized_blocks_path, best_match)
                
                print(f"📝 RENAME:")
                print(f"   From: '{folder_name}'")
                print(f"   To:   '{best_match}'")
                print(f"   Match: '{best_match}' (score: {score:.2f})")
                
                # Perform the rename
                try:
                    if not os.path.exists(new_folder_path):
                        os.rename(folder_path, new_folder_path)
                        print(f"   ✅ Successfully renamed")
                        renamed_count += 1
                    else:
                        print(f"   ⚠️  Target folder already exists, skipping")
                except Exception as e:
                    print(f"   ❌ Error renaming: {e}")
        else:
            print(f"❌ NO MATCH FOUND: '{folder_name}'")
        
        print("-" * 40)
    
    print(f"\nSUMMARY:")
    print(f"Total folders: {len(folders)}")
    print(f"Exact matches: {exact_matches}")
    print(f"Folders renamed: {renamed_count}")
    print(f"No matches found: {len(folders) - exact_matches - renamed_count}")

def print_txt_files_and_folders(organized_blocks_path, output_file="txt_files_report.txt"):
    """Write all txt files and their containing folders to a file"""
    txt_files_found = 0
    
    # Open the output file for writing
    with open(organized_blocks_path+"/"+output_file, 'w', encoding='utf-8') as f:
        
        # Walk through all directories and subdirectories
        for root, dirs, files in os.walk(organized_blocks_path):
            # Sort directories and files for consistent processing
            dirs.sort()
            files.sort()
            # Get the immediate parent folder relative to organized_blocks_path
            relative_path = os.path.relpath(root, organized_blocks_path)
            
            # If we're in the root organized_blocks directory, skip
            if relative_path == '.':
                continue
                
            # Get the top-level folder name (first part of the relative path)
            folder_parts = relative_path.split(os.sep)
            top_level_folder = folder_parts[0]
            
            # Find all txt files in current directory
            txt_files = [f for f in files if f.lower().endswith('.txt')]
            
            for txt_file in txt_files:               
                f.write(f"{txt_file} | {top_level_folder}\n")
                txt_files_found += 1
    
    f.close()
    print(f"Report written to {output_file} with {txt_files_found} txt files")



def run_content_mapping(extracted_files, toc_path, input_dir, output_dir):
    """
    Main function that can be called from other scripts
    
    Args:
        extracted_files (str): Path to the extracted titles file
        toc_path (str): Path to the table of contents file
        input_dir (str): Input directory containing text blocks
        output_dir (str): Output directory for organized blocks
    
    Returns:
        dict: Results containing paths and statistics
    """
    try:
        # Validate input files exist
        if not os.path.exists(extracted_files):
            raise FileNotFoundError(f"Extracted files not found: {extracted_files}")
        if not os.path.exists(toc_path):
            raise FileNotFoundError(f"TOC file not found: {toc_path}")
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Parse documents
        doc_a = parse_doc(toc_path)  # TOC
        doc_b = parse_doc(extracted_files)  # Detailed

        # Build structure and associations
        doc_a_struct = build_structure(doc_a)
        associations = associate(doc_b, doc_a_struct)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Write associations file
        associations_file = os.path.join(output_dir, "associations.txt")
        with open(associations_file, "w", encoding="utf-8") as f:
            for assoc in associations:
                line = f"'{assoc['sub_title']}' (p.{assoc['sub_page']}) -> {assoc['associated_type'].capitalize()} '{assoc['associated_to']}' (p.{assoc['associated_page']})"
                f.write(line + "\n")

        # Set global variables for move_files function
        global INPUT_DIR, OUTPUT_DIR
        INPUT_DIR = input_dir
        OUTPUT_DIR = output_dir

        # Move files
        move_files(associations)

        # Rename folders to match TOC
        rename_folders_to_match_toc(output_dir, toc_path)
        
        # Generate report
        report_file = "txt_files_report.txt"
        print_txt_files_and_folders(output_dir, report_file)

        return {
            "status": "success",
            "extracted_files": extracted_files,
            "toc_path": toc_path,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "associations_file": associations_file,
            "report_file": os.path.join(output_dir, report_file),
            "associations_count": len(associations)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "extracted_files": extracted_files,
            "toc_path": toc_path,
            "input_dir": input_dir,
            "output_dir": output_dir
        }

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process TOC and extracted files for content mapping.')
    parser.add_argument('--extracted', '-e', 
                        default="/home/vc/Theory/NLP2/RAG/AgenticRAG/myRAg/PDF-Extract-Kit/cropsAll2/extracted_titles.txt",
                        help='Path to the extracted titles file')
    parser.add_argument('--toc', '-t', 
                        default="/home/vc/Downloads/AMIF-2025-TF2-AG-INTE-04-PATHWAYS_separator_rules-lev-lear-fca_en-5.toc.txt",
                        help='Path to the table of contents file')
    parser.add_argument('--input', '-i',
                        default="/home/vc/Theory/NLP2/RAG/AgenticRAG/myRAg/PDF-Extract-Kit/cropsAll2/text_block",
                        help='Input directory containing text blocks')
    parser.add_argument('--output', '-o',
                        default="/home/vc/Theory/NLP2/RAG/AgenticRAG/myRAg/PDF-Extract-Kit/cropsAll2/organized_blocks",
                        help='Output directory for organized blocks')
    return parser.parse_args()

if __name__ == "__main__":
    # Command line execution
    args = parse_args()
    
    result = run_content_mapping(
        extracted_files=args.extracted,
        toc_path=args.toc,
        input_dir=args.input,
        output_dir=args.output
    )
    
    if result["status"] == "error":
        print(f"Error: {result['error']}")
        sys.exit(1)
    else:
        print(f"Content mapping completed successfully!")
        print(f"Processed {result['associations_count']} associations")
        print(f"Results saved to: {result['output_dir']}")
        print(f"Report saved to: {result['report_file']}")