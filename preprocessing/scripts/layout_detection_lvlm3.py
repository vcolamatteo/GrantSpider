import os
import re
from pathlib import Path
import glob
import sys
import os.path as osp
import argparse
import cv2
import numpy as np
#from paddleocr import PaddleOCR
import base64
from groq import Groq
import pytesseract
from dotenv import load_dotenv

load_dotenv()

def is_title_only_file(file_path, max_lines=2, max_chars=100):
    """
    Determine if a file contains only a title (short content)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Count non-empty lines
        non_empty_lines = [line for line in content.split('\n') if line.strip()]
        
        # Consider it title-only if it has very few lines and/or characters
        return len(non_empty_lines) <= max_lines and len(content) <= max_chars
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

def get_file_number(filename):
    """
    Extract the number from the beginning of the filename
    """
    match = re.match(r'^(\d+)_', filename)
    return int(match.group(1)) if match else float('inf')

def append_content_to_title_files(folder_path):
    """
    Append content from the next substantial file to title-only files
    """
    folder_path = Path(folder_path)
    
    # Get all txt files and sort them by number
    txt_files = [f for f in folder_path.glob('*.txt')]
    txt_files.sort(key=lambda x: get_file_number(x.name))
    
    files_to_delete = []
    
    for i, current_file in enumerate(txt_files):
        if is_title_only_file(current_file):
            print(f"Found title-only file: {current_file.name}")
            
            # Read the title content
            with open(current_file, 'r', encoding='utf-8') as f:
                title_content = f.read().strip()
            
            # Find the next substantial file
            next_file = None
            for j in range(i + 1, len(txt_files)):
                if not is_title_only_file(txt_files[j]):
                    next_file = txt_files[j]
                    break
            
            if next_file:
                print(f"Appending content from {next_file.name} to {current_file.name}")
                
                # Read the content of the next substantial file
                with open(next_file, 'r', encoding='utf-8') as f:
                    substantial_content = f.read().strip()
                
                # Prepare the merged content (title first, then substantial content)
                merged_content = title_content + "\n\n" + substantial_content
                
                # Write the merged content back to the title-only file
                with open(current_file, 'w', encoding='utf-8') as f:
                    f.write(merged_content)
                
                # Mark the substantial file for deletion
                files_to_delete.append(next_file)
                print(f"Content appended successfully")
            else:
                print(f"No next substantial file found for {current_file.name}")
    
    # Delete the substantial files that were merged
    for file_to_delete in files_to_delete:
        try:
            file_to_delete.unlink()
            print(f"Deleted: {file_to_delete.name}")
        except Exception as e:
            print(f"Error deleting {file_to_delete.name}: {e}")
    
    print(f"\nProcessing complete. Appended content to title files and deleted {len(files_to_delete)} substantial files.")

def preview_append_operations(folder_path):
    """
    Preview what append operations would be performed without actually doing them
    """
    folder_path = Path(folder_path)
    
    # Get all txt files and sort them by number
    txt_files = [f for f in folder_path.glob('*.txt')]
    txt_files.sort(key=lambda x: get_file_number(x.name))
    
    print("Preview of append operations:")
    print("-" * 50)
    
    for i, current_file in enumerate(txt_files):
        if is_title_only_file(current_file):
            # Read the title content for preview
            with open(current_file, 'r', encoding='utf-8') as f:
                title_content = f.read().strip()
            
            # Find the next substantial file
            next_file = None
            for j in range(i + 1, len(txt_files)):
                if not is_title_only_file(txt_files[j]):
                    next_file = txt_files[j]
                    break
            
            if next_file:
                print(f"APPEND TO: {current_file.name}")
                print(f"  Current title: {title_content[:50]}{'...' if len(title_content) > 50 else ''}")
                print(f"  CONTENT FROM: {next_file.name} (will be deleted)")
                print()
            else:
                print(f"NO SOURCE: {current_file.name} (no next substantial file found)")
                print()


def loadFile(ext,path=os.getcwd()):
       
    vec=[]
    #ext=[".jpg",".png"]
    for i in ext:
      #print(path+'//*'+ i)
      vec.extend(glob.glob(path+'//*'+ i))
    
    return vec

def checkPath(path):
    #path=os.getcwd()+'\\'+'background images'
    if not os.path.exists(path):
        os.makedirs(path)
    return 


sys.path.append(osp.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pdf_extract_kit.utils.config_loader import load_config, initialize_tasks_and_models
from tqdm import tqdm
import pdf_extract_kit.tasks

TASK_NAME = 'layout_detection'

# Initialize PaddleOCR globally
#ocr_model = PaddleOCR(use_angle_cls=False, lang='en', show_log=False, use_gpu=True) 

def orderFiles(vec,toFind='_',ext='.jpg', final='_seed'):
    
    fname=np.zeros((len(vec)),int)

    vec_2=vec.copy()

    for i in range(0,len(vec)):
        fname[i]=int(vec[i][vec[i].find(toFind)+len(toFind):vec[i].rfind(final)])
    
    fname=np.argsort(fname)
    
    for i in range(0, len(fname)):
        
        vec_2[i]=vec[fname[i]]
    
    return np.reshape(vec_2,(len(vec_2)))


# Function to calculate Intersection over Union (IoU) between two boxes
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_prime, y1_prime, x2_prime, y2_prime = box2

    # Calculate intersection coordinates
    xi1 = max(x1, x1_prime)
    yi1 = max(y1, y1_prime)
    xi2 = min(x2, x2_prime)
    yi2 = min(y2, y2_prime)

    # Calculate the area of intersection
    intersection_width = max(0, xi2 - xi1)
    intersection_height = max(0, yi2 - yi1)
    intersection_area = intersection_width * intersection_height

    # Calculate the area of both bounding boxes
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_prime - x1_prime) * (y2_prime - y1_prime)

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

def remove_overlapping_boxes(boxes, labels, th=0.3):
    # Removing overlapping boxes with IoU > 0.3, prioritizing titles over plainText,
    # and plainText over other classes when title is not present
    filtered_boxes = []
    filtered_labels = []

    for i in range(len(boxes)):
        # Check if the current box has a high overlap with any box in filtered_boxes
        keep = True
        for j in range(len(filtered_boxes)):
            iou = calculate_iou(boxes[i], filtered_boxes[j])
            if iou > th:
                # If overlap detected, check class priorities
                current_class = labels[i]
                existing_class = filtered_labels[j]
                
                # Priority 1: Title (0) has highest priority
                if current_class == 0 and existing_class != 0:  # Current is title, existing is not
                    filtered_boxes[j] = boxes[i]
                    filtered_labels[j] = labels[i]
                    keep = False
                    break
                elif current_class != 0 and existing_class == 0:  # Current is not title, existing is title
                    keep = False
                    break
                # Priority 2: PlainText (1) has second priority over non-title classes
                elif current_class == 1 and existing_class != 0 and existing_class != 1:  # Current is plainText, existing is other (not title, not plainText)
                    filtered_boxes[j] = boxes[i]
                    filtered_labels[j] = labels[i]
                    keep = False
                    break
                elif current_class != 0 and current_class != 1 and existing_class == 1:  # Current is other, existing is plainText
                    keep = False
                    break
                # For same class or other overlapping cases, keep the first one (existing behavior)
                else:
                    keep = False
                    break
        
        # If no high-overlap box was found, add the box to the filtered list
        if keep:
            filtered_boxes.append(boxes[i])
            filtered_labels.append(labels[i])

    # Convert lists back to numpy arrays
    filtered_boxes = np.array(filtered_boxes)
    filtered_labels = np.array(filtered_labels)

    return filtered_boxes, filtered_labels


def main(config_path):

    dst_crops=args.dst_crops
    config = load_config(config_path)
    task_instances = initialize_tasks_and_models(config)

    # get input and output path from config
    input_data = config.get('inputs', None)
    result_path = config.get('outputs', 'outputs'+'/'+TASK_NAME)

    # layout_detection_task
    model_layout_detection = task_instances[TASK_NAME]

    print("\n========================================")
    print("predicting layout detection...")
    # for image detection
    detection_results = model_layout_detection.predict_images(input_data, result_path)
    #print(detection_results)
    

    # for pdf detection
    # detection_results = model_layout_detection.predict_pdfs(input_data, result_path)
    
    classes= {0: 'title', 1: 'plainText', 2: 'abandon', 3: 'figure', 4: 'figureCaption', 5: 'table', 6: 'tableCaption', 7: 'tableFootnote', 8: 'isolateFormula', 9: 'formulaCaption'}    
    paths=[]
    for j in range(0,len(detection_results)):
        #print(detection_results[j]["im_path"])
        paths.append(detection_results[j]["im_path"])
    
    
    # Extract page numbers from file paths and pair with indices
    page_numbers_with_indices = [(int(re.search(r'page_(\d+)', path).group(1)), i) for i, path in enumerate(paths)]

    # Sort by page number
    sorted_pages_with_indices = sorted(page_numbers_with_indices, key=lambda x: x[0])

    # Extract sorted indices
    sorted_indices = [index for _, index in sorted_pages_with_indices]

    # Display sorted indices
    #print("Sorted indices:", sorted_indices)

    # If you want the sorted file paths as well, use:
    # sorted_file_paths = [paths[i] for i in sorted_indices]
    # print("Sorted file paths:\n", sorted_file_paths)
    
    index=0
    for j in range(0,len(detection_results)):

        preds=detection_results[sorted_indices[j]]["classes"]#.cpu().numpy()    
        #print(j, sorted_indices[j],  detection_results[sorted_indices[j]]["im_path"])
        bbox=detection_results[sorted_indices[j]]["boxes"]#.cpu().numpy()       
        
        bbox,preds=remove_overlapping_boxes(bbox,preds)        
        #print(preds)
        # Get the indices that would sort the boxes array by the y-coordinate (second element)
        indices = bbox[:, 1].argsort()
        # Sort both boxes and labels using the sorted indices    
        bbox = bbox[indices]
        preds = preds[indices]

        #dst_crops="cropsAll/"
        checkPath(dst_crops)
        img=cv2.imread(detection_results[sorted_indices[j]]["im_path"])
        #print(img.shape, j, sorted_indices[j], bbox.shape)
        for i in range(0,bbox.shape[0]):
            if classes[preds[i]]=="plainText" or classes[preds[i]]=="title" or classes[preds[i]]=="table":
                ####### enlarge bbox on width dimension for avoiding wrong bbox detection (cutted lines) ####
                ####### could be a problem in case of multile bbox on the same line (formula ad es.)     ####
                # print(bbox[i])
                bbox_i=bbox[i]
                bbox_i[0]=0
                bbox_i[2]=img.shape[1]
                ######################################
                #print(bbox_i, classes[int(preds[i])])
                crop=img[int(bbox_i[1]):int(bbox_i[3]),int(bbox_i[0]):int(bbox_i[2])]
                index=index+1
                cv2.imwrite(dst_crops+"crop_"+str(index)+"_"+str(classes[int(preds[i])])+".jpg",crop)

        #print(f'The predicted results can be found at {result_path}')

    
    #dst_crops="cropsAll/"
    dst_text=dst_crops+"text_block/"
    checkPath(dst_text)

    # Store detection results and sorted indices for title extraction
    detection_results_for_titles = detection_results
    sorted_indices_for_titles = sorted_indices

    # NEW: Extract titles before paragraph extraction
    print("\n========================================")
    print("extracting titles...")
    titles_output_file = dst_crops + "extracted_titles.txt"
    extract_titles_to_file(dst_crops, titles_output_file, detection_results_for_titles, sorted_indices_for_titles)
    
    print("\n========================================")
    print("paragraphs extraction...\n")
    ocr(dst_crops, dst_text)
    
    # Clean up: delete all jpg files in the dst_crops directory
    for file in os.listdir(dst_crops):
        if file.endswith('.jpg'):
            file_path = os.path.join(dst_crops, file)
            os.remove(file_path)


    # Preview operations first
    #preview_append_operations(dst_text)
    
    append_content_to_title_files(dst_text)

    


def extract_text_from_image(image_path):
    """
    Extract text using Tesseract with preprocessing instead of PaddleOCR
    """

    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return ""
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    binary_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Use Tesseract with custom configuration on grayscale image
    custom_config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(gray, lang='eng', config=custom_config)
    
    # Clean the text to remove non-ASCII and problematic characters
    if extracted_text:
        # Remove form feed, carriage return, and other control characters
        extracted_text = re.sub(r'[\f\r\v\x00-\x1F\x7F-\x9F]', ' ', extracted_text)
        # Replace multiple spaces with single space
        extracted_text = re.sub(r'\s+', ' ', extracted_text)
        # Strip leading/trailing whitespace
        extracted_text = extracted_text.strip()
        # Replace "e " with bullet point
        extracted_text = re.sub(r'\be\s', '\n• ', extracted_text)
        # Don't encode to ASCII if you want to keep the bullet character
        
    return extracted_text


# Function to encode the image for LLM
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_table_with_llm(image_path):
    # Getting the base64 string
    base64_image = encode_image(image_path)
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract this table and format it as plain text maintaining the table grid structure. Preserve column and row alignment. Do not add anything else."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error extracting table with LLM: {e}")
        # Fallback to OCR if LLM fails
        return extract_text_from_image(image_path)

def remove_empty_lines(input_string):
    # Split the input string by lines, filter out empty lines, and join the result
    return "\n".join(line for line in input_string.splitlines() if line)

def ocr(dst_crops, dst_text):

    label=""
    vec=loadFile([".jpg"],dst_crops)
    vec=orderFiles(vec,"_",final="_")
    #print(vec)

    file_count=0    
    current_title = None
    pending_table_text = ""  # Store table text to add to next title section
    # Initialize prev_label with a value that won't match "plainText" or "table" initially
    # This ensures that if the first item is a title, it creates a new file.
    prev_label = "initial" 


    # Use tqdm for progress bar
    for i in tqdm(range(0, len(vec)), desc="Processing document sections"):
        # prev_label=label # This was moved down to be set at the end of the relevant block
        current_processing_label=vec[i][vec[i].rfind("_")+1:-4]
        if current_processing_label=="plainText" or current_processing_label=="title" or current_processing_label=="table":
            
            raw_text_from_ocr = "" # Initialize for logging
            # Use different extraction methods based on content type
            if current_processing_label=="table":
                text = extract_table_with_llm(vec[i])
                # raw_text_from_ocr = text # This might be too verbose for LLM output
            else:
                text = extract_text_from_image(vec[i])
            raw_text_from_ocr = text # Assign after extraction for consistent logging
            
            #print(f"DEBUG: Crop: {vec[i]}, Label: {current_processing_label}, Raw Extracted Text: '{raw_text_from_ocr[:200]}...'") # Logging statement (truncated for brevity)

            # Handle table text - store it to be added to the next title section
            if current_processing_label=="table":
                # table_text = text.replace("\n", " ") # Not needed if LLM formats well
                pending_table_text += f"\n\nTable content:\n{text}\n"
                prev_label = current_processing_label # Update prev_label here
                continue  # Skip to next iteration, don't create separate file for table
            
            # If this is a title, store it for the filename
            if current_processing_label=="title":
                # Clean the title text to make it suitable for filename
                title_text = text.replace("\n", " ")
                # Remove invalid filename characters
                title_text = re.sub(r'[<>:"/\\|?*]', '', title_text)
                # Limit length and replace spaces with underscores
                # Ensure current_title is not empty after cleaning, otherwise use a placeholder
                current_title = title_text.replace(" ", "_") if title_text else f"untitled_{file_count+1}"

            # Logic for creating new file or appending
            # A new file is created if:
            # 1. It's the first processed item (i == 0).
            # 2. The current item is a "title" and the previous processed item was "plainText" or "table".
            #    (This implies a new section starts with a title).
            # 3. The current item is "title" and current_title was just updated (meaning this is the first title encountered or a new title after non-title content)
            
            # Ensure text is not empty before processing
            cleaned_text = text.replace("\n", " ")
            if not cleaned_text and current_processing_label != "title": # Don't skip if it's an empty title, as it might be a section start
                print(f"DEBUG: Skipping empty text for {vec[i]}")
                prev_label = current_processing_label # Update prev_label
                continue


            if i == 0 or \
               (current_processing_label == "title" and (prev_label in ["plainText", "table"] or prev_label == "initial")) or \
               (current_processing_label == "title" and vec[i][vec[i].rfind("_")+1:-4] == "title"): # Condition for new file if current is title

                #print(i,vec[i][vec[i].rfind("/")+1:],"new file", "prev ",prev_label, "current", current_processing_label) 
                
                # Use title as filename if available, otherwise fallback to counter
                if current_title and current_processing_label == "title": # Use current_title if this block is a title
                    text_file = f"{file_count+1}_{current_title}.txt"
                elif current_title and current_processing_label == "plainText" and prev_label == "title": # If it's plain text immediately after a title
                     text_file = f"{file_count+1}_{current_title}.txt" # Should append to the title's file initially
                else: # Fallback for first item if not title, or other edge cases
                    text_file = f"{file_count+1}_text.txt"
                    if not current_title and current_processing_label == "title": # If title was empty, generate a name
                        current_title = f"untitled_{file_count+1}"
                        text_file = f"{file_count+1}_{current_title}.txt"


                # Prepare text for writing
                # text_to_write = text.replace("\n"," ") + "\n\n" # Original
                text_to_write = cleaned_text + "\n\n" if cleaned_text else ""


                # Add any pending table text to the title section
                if pending_table_text:
                    text_to_write = pending_table_text + text_to_write # Prepend table if it belongs to this new section
                    pending_table_text = ""  # Reset after adding
                
                if text_to_write: # Only write if there's actual content
                    with open(os.path.join(dst_text, text_file), "w") as file:
                        file.write(text_to_write)       
                    file_count=file_count+1        
                elif not text_to_write and current_processing_label == "title": # Create empty file for title if title text itself was empty
                    with open(os.path.join(dst_text, text_file), "w") as file:
                        file.write("\n\n") # Write newlines to signify an empty section start
                    file_count=file_count+1

            else: # Append to existing file
                #print(i,vec[i][vec[i].rfind("/")+1:],"same file", "prev ",prev_label, "current", current_processing_label)
                # text_to_write = text.replace("\n"," ")+"\n\n" # Original
                text_to_write = cleaned_text + "\n\n" if cleaned_text else ""
                
                # Add any pending table text
                if pending_table_text: # This case might be rare if tables are always followed by titles/new sections
                    text_to_write = pending_table_text + text_to_write
                    pending_table_text = ""  # Reset after adding

                if text_to_write: # Only append if there's actual content
                    # Determine the file to append to. It should be the last created/written file.
                    # This requires text_file to be defined from the "new file" block.
                    # If current_title is set and we are in a plainText block following a title, use that.
                    if current_title: # If a title is active, append to its file
                         active_file_name = f"{file_count}_{current_title}.txt"
                    else: # Fallback if no title context (should be rare if logic is correct)
                         active_file_name = f"{file_count}_text.txt" # Append to the latest numbered file

                    with open(os.path.join(dst_text, active_file_name), "a") as file:
                        file.write(text_to_write)
            
            prev_label = current_processing_label # Update prev_label for the next iteration

    # After the loop, if there's any pending table text, write it to a new file or the last known file.
    if pending_table_text:
        print("DEBUG: Writing pending table text at the end.")
        if current_title:
            final_text_file = f"{file_count+1}_{current_title}_trailing_table.txt" # Or append to current_title.txt
        else:
            final_text_file = f"{file_count+1}_text_trailing_table.txt"
        with open(os.path.join(dst_text, final_text_file), "w") as file:
            file.write(pending_table_text)

def extract_titles_to_file(dst_crops, output_file="titles_extracted.txt", detection_results=None, sorted_indices=None):
    """
    Extract text from all title objects and save them with their original image names
    """
    vec = loadFile([".jpg"], dst_crops)
    vec = orderFiles(vec, "_", final="_")
    
    title_extractions = []
    
    # Create a mapping from crop index to original image
    crop_to_original = {}
    if detection_results and sorted_indices:
        crop_index = 0
        for j in range(len(detection_results)):
            original_image_path = detection_results[sorted_indices[j]]["im_path"]
            original_image_name = os.path.basename(original_image_path)
            
            preds = detection_results[sorted_indices[j]]["classes"]
            bbox = detection_results[sorted_indices[j]]["boxes"]
            
            # Apply the same filtering as in main()
            bbox, preds = remove_overlapping_boxes(bbox, preds)
            indices = bbox[:, 1].argsort()
            bbox = bbox[indices]
            preds = preds[indices]
            
            classes = {0: 'title', 1: 'plainText', 2: 'abandon', 3: 'figure', 4: 'figureCaption', 5: 'table', 6: 'tableCaption', 7: 'tableFootnote', 8: 'isolateFormula', 9: 'formulaCaption'}
            
            for i in range(bbox.shape[0]):
                if classes[preds[i]] in ["plainText", "title", "table"]:
                    crop_index += 1
                    crop_to_original[crop_index] = original_image_name
    
    # Use tqdm for progress bar
    for i in tqdm(range(0, len(vec)), desc="Extracting titles"):
        current_processing_label = vec[i][vec[i].rfind("_")+1:-4]
        
        if current_processing_label == "title":
            # Extract text from the title image
            extracted_text = extract_text_from_image(vec[i])
            
            # Get crop index from filename (crop_X_title.jpg)
            crop_filename = os.path.basename(vec[i])
            crop_number = int(crop_filename.split("_")[1])
            
            # Get original image name
            original_image_name = crop_to_original.get(crop_number, "Unknown_source")
            
            # Clean the extracted text
            if extracted_text:
                cleaned_text = re.sub(r'[\f\r\v\x00-\x1F\x7F-\x9F]', ' ', extracted_text)
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            else:
                cleaned_text = "[No text extracted]"
            
            title_extractions.append({
                'original_image': original_image_name,
                'crop_image': crop_filename,
                'text': cleaned_text
            })
    
    # Save all title extractions to file
    count=1
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Text | Page\n")
        for extraction in title_extractions:
            img_name=extraction['original_image']
            f.write(str(count)+"_"+extraction['text'].replace(" ","_")+" | "+img_name[img_name.rfind("_")+1:-4]+"\n")
            count+=1
    print(f"Extracted {len(title_extractions)} titles and saved to {output_file}")
    return title_extractions

def run_layout_detection(config_path, dst_crops="cropsAll2/"):
    """
    Main function that can be called from other scripts
    
    Args:
        config_path (str): Path to the configuration file
        dst_crops (str): Path to the output crops directory
    
    Returns:
        dict: Results containing paths and statistics
    """
    try:
        # Set up the arguments programmatically
        class Args:
            def __init__(self, config, dst_crops):
                self.config = config
                self.dst_crops = dst_crops
        
        # Create args object
        global args
        args = Args(config_path, dst_crops)
        
        # Run the main function
        main(config_path)
        
        # Return success status and paths
        return {
            "status": "success",
            "config_path": config_path,
            "dst_crops": dst_crops,
            "dst_text": dst_crops + "text_block/",
            "titles_file": dst_crops + "extracted_titles.txt"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "config_path": config_path,
            "dst_crops": dst_crops
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Run a task with a given configuration file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--dst_crops', type=str, default="cropsAll2/", help='Path to the output crops directory.')
    return parser.parse_args()

if __name__ == "__main__":
    # Command line execution
    args = parse_args()
    result = run_layout_detection(args.config, args.dst_crops)
    
    if result["status"] == "error":
        print(f"Error: {result['error']}")
        sys.exit(1)
    else:
        print(f"Layout detection completed successfully!")
        print(f"Results saved to: {result['dst_crops']}")


# python3.10 scripts/layout_detection.py --config configs/layout_detection.yaml
# python3.10 scripts/layout_detection.py --config configs/layout_detection_layoutlmv3.yaml


# python3 scripts/layout_detection.py --config configs/layout_detection_layoutlmv3_grant_2.yaml
# python3 scripts/layout_detection.py --config  configs/layout_detection_layoutlmv3_grant_2.yaml


# retrieval granulare, k 50 o 100
# re-ranker a 5/10
# ogni chunk sopravvissuto lo estendo con tutto il relativo paragrafo/capitolo e lo passo all'llm

# quindi devo aggiungere come metadati ad ogni chunk, il relativo capitolo e sovraparagrafo di apparteneza e anche i token di entrambi


# primo step converto il pdf in jpg con extract_jpg.py e rimuovo le pagie fino al toc (incluso)
# extract_par_text_grant.py per estrarre il TOC
# poi extract_par_text_2.py per estrare il testo dei paragrafi dal TOC
# poi lancio questo che mi estrae le immagini e i titoli e fa l'ocr
# poi lancio content_mapping_folders.py 

# extract_par_text_grant.py
# python3 file_chunks_extractor.py # makes all in one step