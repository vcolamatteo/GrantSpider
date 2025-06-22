import re
from difflib import get_close_matches

def parse_doc(doc_path):
    with open(doc_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    items = []
    for line in lines:
        match = re.match(r'\s*(.*?)\s*\|\s*(\d+)', line)
        if match:
            title = match.group(1).strip()
            page = int(match.group(2))
            # Determine structure level
            number_match = re.match(r'^\d+(\.\d+)?', title)
            if number_match:
                number = number_match.group()
            else:
                number = None
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
    return structure

def associate(doc_b, doc_a_struct):
    associations = []
    for item in doc_b:
        num = item['number']
        if is_chapter(num) or is_paragraph(num):
            # It's already in A, so skip (only subparagraphs needed)
            continue
        # Find the nearest preceding paragraph or chapter
        nearest = None
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

# Main execution
doc_a = parse_doc("/home/vc/Downloads/AMIF-2025-TF2-AG-INTE-04-PATHWAYS_separator_om_en-4.toc.txt")  # TOC
doc_b = parse_doc("/home/vc/Theory/NLP2/RAG/AgenticRAG/myRAg/PDF-Extract-Kit/cropsAll/extracted_titles.txt")  # Detailed

doc_a_struct = build_structure(doc_a)
sub_associations = associate(doc_b, doc_a_struct)

# Output the result
for assoc in sub_associations:
    print(f"'{assoc['sub_title']}' (p.{assoc['sub_page']}) -> {assoc['associated_type'].capitalize()} '{assoc['associated_to']}' (p.{assoc['associated_page']})")
