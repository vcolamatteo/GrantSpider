from typing import List, Dict, Any, Optional, Tuple
from typing_extensions import TypedDict
from tqdm import tqdm
import utils
import pickle
import os
import time
import torch
import sys
import gc
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from colorama import Fore, Style, init
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph, START
import langsmith

# Initialize colorama
init()

# Import graders
from graders import (
    retrieval_grader, 
    question_rewriter, 
    answer_grader, 
    hallucination_grader,
    generator
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

utils.suppress_warnings()

@dataclass
class EmbeddingResult:
    """Structured container for embedding results"""
    embedding: List[float]
    metadata: Dict[str, Any]
    sparse_embedding: Optional[Dict[str, List]] = None

@dataclass
class ChatMessage:
    """Chat message container"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None

class ChatMemory:
    """Simple chat memory management"""
    def __init__(self, max_messages: int = 20):
        self.messages: List[ChatMessage] = []
        self.max_messages = max_messages
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to chat history"""
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        self.messages.append(message)
        
        # Keep only the last max_messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self, include_last_n: int = 5) -> str:
        """Get recent chat context for query enhancement"""
        if not self.messages:
            return ""
        
        recent_messages = self.messages[-include_last_n:]
        context_parts = []
        
        for msg in recent_messages:
            if msg.role == 'user':
                context_parts.append(f"Previous question: {msg.content}")
            elif msg.role == 'assistant':
                # Use the full answer in the context
                context_parts.append(f"Previous answer: {msg.content}")
        
        return "\n".join(context_parts)
    
    def clear_history(self):
        """Clear chat history"""
        self.messages.clear()
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation for display"""
        if not self.messages:
            return "No conversation history."
        
        summary = []
        for i, msg in enumerate(self.messages[-10:], 1):  # Show last 10 messages
            role_indicator = "👤" if msg.role == 'user' else "🤖"
            summary.append(f"{role_indicator} {msg.content}")
        
        return "\n".join(summary)

class GraphState(TypedDict):
    """Represents the state of our late fusion hybrid RAG graph."""
    question: str
    generation: str
    dense_documents: Dict[str, Any]
    sparse_documents: Dict[str, Any]
    combined_documents: Dict[str, Any]
    hallucination_score: str
    answer_score: str
    fusion_method: str
    alpha: float
    rrf_k: int
    hallucination_retry_count: int
    # Add exit condition fields
    iteration_count: int
    start_time: float
    max_iterations: int
    max_time_seconds: float
    # Add query classification fields
    query_classification: str
    session_end: bool
    chat_context: str
    original_question: str
    # Add follow-up field
    is_followup: bool
    # Keep conversation_history for follow-up handling only
    conversation_history: List[Dict[str, str]]
    history_is_sufficient: str

class LateFusionHybridRAGChatPipeline:
    """Chat-enabled LangGraph-based Late Fusion Hybrid RAG Pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dense_model = None
        self.dense_tokenizer = None
        self.sparse_model = None
        self.sparse_tokenizer = None
        self.reranker = None
        self.reranker_tokenizer = None
        self.dense_index = None
        self.sparse_index = None
        self.pc = None
        self.app = None
        self.llm = None
        
        # Initialize chat memory
        self.chat_memory = ChatMemory(max_messages=config.get('max_chat_history', 20))
        self.session_active = True
    

    # Add helper function to check exit conditions
    def should_exit_early(self, state):
        """Check if we should exit due to iteration or time limits"""
        current_time = time.time()
        elapsed_time = current_time - state.get("start_time", current_time)
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 15)
        max_time = state.get("max_time_seconds", 60.0)
        
        if iteration_count >= max_iterations:
            print(f"{Fore.YELLOW}⛔ Reached maximum iterations ({max_iterations}) ⇒ EARLY EXIT.{Style.RESET_ALL}")
            return True
        
        if elapsed_time >= max_time:
            print(f"{Fore.YELLOW}⛔ Reached time limit ({max_time}s) ⇒ {Style.BRIGHT}EARLY EXIT.{Style.RESET_ALL}")
            return True
        
        return False

    def increment_iteration(self, state):
        """Helper function to increment iteration count and initialize timing"""
        # Initialize timing if not present
        if "start_time" not in state or state["start_time"] == 0.0:
            state["start_time"] = time.time()
        
        # Increment iteration count
        iteration_count = state.get("iteration_count", 0) + 1
        state["iteration_count"] = iteration_count
        
        print(f"{Fore.CYAN}📊 Node iteration: {iteration_count}{Style.RESET_ALL}")
        return iteration_count
    
    def load_models(self):
        """Load all required models with error handling"""
        try:
            logger.info("Loading dense model...")
            self.dense_model, self.dense_tokenizer = utils.load_Qwen()
            
            logger.info("Loading sparse model...")
            self.sparse_model, self.sparse_tokenizer = utils.spalde_v3_load()
            
            if self.config.get('use_reranker', True):
                logger.info("Loading reranker...")                
                self.reranker, self.reranker_tokenizer = utils.load_QwenReranker()
            else:
                logger.info("Loading reranker skipped as per configuration")                
            
            # Initialize LLM for grading
            self.llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model="llama-3.3-70b-versatile",
                temperature=0
            )
            
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def process_documents_batch(self, docs_batch: List[str], doc_info_batch: List[Dict]) -> List[EmbeddingResult]:
        """Process a batch of documents with proper memory management"""
        results = []
        
        try:
            # Process dense embeddings in batch
            dense_embeddings = utils.runDense(
                self.dense_model, 
                self.dense_tokenizer, 
                docs_batch,
            )
            
            # Process sparse embeddings one by one to manage GPU memory
            sparse_embeddings = []
            for doc in docs_batch:
                sparse_emb = self.splade_v3(doc)
                sparse_embeddings.append(sparse_emb)
                # Clear cache after each document
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Combine results
            for i, (dense_emb, sparse_emb, doc_info) in enumerate(
                zip(dense_embeddings, sparse_embeddings, doc_info_batch)
            ):
                result = EmbeddingResult(
                    embedding=dense_emb,
                    metadata=doc_info,
                    sparse_embedding=sparse_emb
                )
                results.append(result)
            
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error processing document batch: {e}")
            # Emergency cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        return results
    
    def load_paragraph_mapping(self, report_file_path: str) -> Dict[str, str]:
        """Load the paragraph mapping from txt_files_report.txt"""
        paragraph_mapping = {}
    
        try:
            with open(report_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Split by " | " separator
                    if " | " in line:
                        filename, paragraph_name = line.split(" | ", 1)
                        filename = filename.strip()
                        paragraph_name = paragraph_name.strip()
                        
                        # Store mapping
                        paragraph_mapping[filename] = paragraph_name
                        logger.debug(f"Mapped {filename} -> {paragraph_name}")
                    else:
                        logger.warning(f"Line {line_num} doesn't follow expected format: {line}")
            
            logger.info(f"Loaded {len(paragraph_mapping)} filename to paragraph mappings")
            return paragraph_mapping
            
        except FileNotFoundError:
            logger.error(f"Report file not found: {report_file_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading paragraph mapping: {e}")
            return {}


    def create_embeddings_multi_doc(self, root_path: str, chunk_dst: str) -> Tuple[List[EmbeddingResult], List[Dict]]:
        """Create embeddings for multiple documents recursively"""
        
        all_dense_embeddings = []
        all_sparse_embeddings = []
        
        # Find all document folders
        doc_folders = []
        for item in os.listdir(root_path):
            item_path = os.path.join(root_path, item)
            if os.path.isdir(item_path):
                text_block_path = os.path.join(item_path, "text_block")
                report_path = os.path.join(item_path, "organized_blocks", "txt_files_report.txt")
                
                if os.path.exists(text_block_path) and os.path.exists(report_path):
                    doc_folders.append({
                        'doc_name': item,
                        'text_block_path': text_block_path,
                        'report_path': report_path
                    })
                    logger.info(f"Found document folder: {item}")
                else:
                    logger.warning(f"Skipping {item}: missing text_block or txt_files_report.txt")
        
        if not doc_folders:
            logger.error(f"No valid document folders found in {root_path}")
            return [], []
        
        logger.info(f"Processing {len(doc_folders)} document folders")
        
        # Process each document folder
        for doc_info in doc_folders:
            doc_name = doc_info['doc_name']
            text_block_path = doc_info['text_block_path']
            report_path = doc_info['report_path']
            
            logger.info(f"Processing document: {doc_name}")
            
            # Load paragraph mapping for this document
            paragraph_mapping = self.load_paragraph_mapping(report_path)
            
            if not paragraph_mapping:
                logger.error(f"No paragraph mapping loaded for {doc_name} - skipping")
                continue
            
            # Load and sort files for this document
            vec = utils.loadFile_recursive([".txt"], text_block_path)
            vec = sorted(vec, key=self.extract_leading_number)
            vec = [f for f in vec if not f.endswith(("associations.txt", "txt_files_report.txt"))]
            
            logger.info(f"Processing {len(vec)} documents for {doc_name}")
            
            # Use smaller batch size to avoid memory issues
            batch_size = self.config.get('processing_batch_size', 1)
            
            for i in tqdm(range(0, len(vec), batch_size), desc=f"Processing {doc_name}"):
                batch_files = vec[i:i + batch_size]
                batch_docs = []
                batch_info = []
                
                # Read documents in batch
                for doc_path in batch_files:
                    try:
                        with open(doc_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                        
                        # Extract metadata
                        filename = doc_path.split('/')[-1]
                        doc_num = filename.split('_')[0]
                        folder_name = doc_path.split('/')[-2]
                        
                        # Get paragraph name from mapping or use folder name as fallback
                        paragraph_related = paragraph_mapping.get(filename, folder_name)
                        paragraph_related = paragraph_related.replace(" ", "_")
                        
                        # Log if we're using fallback
                        if filename not in paragraph_mapping:
                            logger.debug(f"No mapping found for {filename}, using folder name: {folder_name}")
                        
                        doc_info_item = {
                            'filename': filename,
                            'paragraph_related': paragraph_related,
                            'init_doc': doc_name,  # Use the actual document name
                            'chunk_id': f"{doc_name}_{doc_num}",  # Prefix with doc name for uniqueness
                            'context': content,
                        }
                        
                        batch_docs.append(content)
                        batch_info.append(doc_info_item)
                        
                        # Log successful mapping
                        logger.debug(f"Processed {filename} -> paragraph: {paragraph_related} for doc: {doc_name}")
                        
                    except Exception as e:
                        logger.error(f"Error reading {doc_path}: {e}")
                        continue
                
                # Process batch if we have documents
                if batch_docs:
                    results = self.process_documents_batch(batch_docs, batch_info)
                    
                    for result in results:
                        all_dense_embeddings.append({
                            'embedding': result.embedding,
                            'metadata': result.metadata
                        })
                        all_sparse_embeddings.append({
                            'embedding': result.sparse_embedding,
                            'metadata': result.metadata
                        })
            
                # Aggressive memory cleanup after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Save embeddings
        os.makedirs(chunk_dst, exist_ok=True)
        
        with open(os.path.join(chunk_dst, 'chunks_dense.pkl'), 'wb') as f:
            pickle.dump(all_dense_embeddings, f)
        
        with open(os.path.join(chunk_dst, 'chunks_sparse.pkl'), 'wb') as f:
            pickle.dump(all_sparse_embeddings, f)
        
        logger.info(f"Created {len(all_dense_embeddings)} embeddings from {len(doc_folders)} documents")
        
        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return all_dense_embeddings, all_sparse_embeddings

    def load_complete_paragraphs_multi_doc(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Load complete paragraphs for multiple documents based on init_doc field"""
        try:
            # Use a dictionary to store unique paragraphs, keyed by (init_doc, paragraph_name)
            unique_paragraphs = {}
            for match in results['matches']:
                paragraph_related = match['metadata'].get('paragraph_related')
                init_doc = match['metadata'].get('init_doc')
                if paragraph_related and init_doc:
                    # Store the first match found for a unique paragraph, used for fallback
                    if (init_doc, paragraph_related) not in unique_paragraphs:
                        unique_paragraphs[(init_doc, paragraph_related)] = match

            logger.info(f"Found {len(unique_paragraphs)} unique paragraphs to load from multiple documents.")

            paragraph_contents = {}
            source_path = self.config.get('source_path')
            if not source_path:
                logger.error("`source_path` not found in configuration. Cannot load paragraphs.")
                return results # Return original results if path is missing

            for (init_doc, paragraph_name), match in unique_paragraphs.items():
                paragraph_file_path = os.path.join(source_path, init_doc, "extracted_sections", f"{paragraph_name}.txt")
                print(paragraph_file_path)
                #try:
                with open(paragraph_file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    paragraph_contents[(init_doc, paragraph_name)] = content
                    logger.debug(f"Successfully loaded paragraph '{paragraph_name}' from doc '{init_doc}'.")
                # except FileNotFoundError:
                #     logger.warning(f"Paragraph file not found: {paragraph_file_path}. Using chunk content as fallback.")
                # except Exception as e:
                #     logger.error(f"Error reading paragraph file {paragraph_file_path}: {e}. Using chunk content as fallback.")

            enhanced_matches = []
            seen_paragraphs = set()

            # Iterate through original matches to preserve order and scores
            for match in results['matches']:
                paragraph_related = match['metadata'].get('paragraph_related')
                init_doc = match['metadata'].get('init_doc')
                
                if not paragraph_related or not init_doc:
                    continue

                paragraph_key = (init_doc, paragraph_related)
                if paragraph_key not in seen_paragraphs:
                    seen_paragraphs.add(paragraph_key)

                    # Get loaded paragraph content, or fall back to the original chunk's context
                    content = paragraph_contents.get(paragraph_key, match['metadata'].get('context', ''))
                    
                    if not content:
                         logger.error(f"Could not find any content for paragraph: {paragraph_related} in doc: {init_doc}")
                         continue

                    enhanced_match = {
                        'id': f"paragraph_{init_doc}_{paragraph_related.replace(' ', '_')}",
                        'score': match['score'],
                        'metadata': {
                            'paragraph_related': paragraph_related,
                            'context': content,
                            'original_chunk_id': match['id'],
                            'retrieval_source': match.get('retrieval_source', 'unknown'),
                            'filename': f"{paragraph_related}.txt",
                            'init_doc': init_doc
                        }
                    }
                    enhanced_matches.append(enhanced_match)

            enhanced_results = {
                'matches': enhanced_matches,
                'namespace': results.get('namespace', ''),
                'usage': results.get('usage', {})
            }

            logger.info(f"Enhanced results with {len(enhanced_matches)} unique complete paragraphs.")
            return enhanced_results

        except Exception as e:
            logger.error(f"A critical error occurred in load_complete_paragraphs_multi_doc: {e}")
            return results # Fallback to original results on any major error

    # Update the existing create_embeddings method to call the multi-doc version if needed
    def create_embeddings(self, source_path: str, chunk_dst: str, association_file: str = None) -> Tuple[List[EmbeddingResult], List[Dict]]:
        """Create embeddings with support for both single and multiple documents"""
        
        # Check if this is a multi-document setup
        if association_file is None:
            # Multi-document mode: source_path is root containing multiple doc folders
            return self.create_embeddings_multi_doc(source_path, chunk_dst)        
    
    
    # Update the load_paragraphs method to use the multi-doc version
    def load_paragraphs(self, state):
        """Load complete paragraphs from filtered documents (updated for multi-doc support)"""
        print(f"\n{Fore.CYAN}📖 ---LOAD COMPLETE PARAGRAPHS---{Style.RESET_ALL}")
        documents = state["combined_documents"]
        question = state["question"]
        chat_context = state.get("chat_context", "")
    
        # Increment iteration and check limits
        iteration_count = self.increment_iteration(state)
        enhanced_documents = documents  # Initialize with original documents as a fallback

        try:
            # Check if we have documents to process
            if documents and documents.get("matches"):
                logger.info(f"Attempting to load full paragraphs for {len(documents['matches'])} chunks.")
                enhanced_documents = self.load_complete_paragraphs_multi_doc(documents)

                print(f"{Fore.GREEN}✅ Loaded {len(enhanced_documents['matches'])} complete paragraphs{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}⚠️ No documents to load paragraphs from.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Error loading paragraphs: {e}{Style.RESET_ALL}")
            enhanced_documents = documents # Fallback to original chunks on error
        
        return {
            "combined_documents": enhanced_documents,
            "dense_documents": state["dense_documents"],
            "sparse_documents": state["sparse_documents"],
            "question": question,
            "original_question": state.get("original_question", question),
            "chat_context": chat_context,
            "fusion_method": state["fusion_method"],
            "alpha": state["alpha"],
            "rrf_k": state["rrf_k"],
            "iteration_count": iteration_count,
            "start_time": state.get("start_time", time.time()),
            "max_iterations": state.get("max_iterations", 15),
            "max_time_seconds": state.get("max_time_seconds", 60.0),
        }

    
    def load_embeddings(self, chunk_dst: str) -> Tuple[List[Dict], List[Dict]]:
        """Load embeddings from disk"""
        try:
            with open(os.path.join(chunk_dst, 'chunks_dense.pkl'), 'rb') as f:
                dense_embeddings = pickle.load(f)
            
            with open(os.path.join(chunk_dst, 'chunks_sparse.pkl'), 'rb') as f:
                sparse_embeddings = pickle.load(f)            
            
            logger.info(f"Loaded {len(dense_embeddings)} dense and {len(sparse_embeddings)} sparse embeddings")
            
            return dense_embeddings, sparse_embeddings
            
        except FileNotFoundError:
            logger.error("Embedding files not found")
            return [], []
    
    def create_dense_upserts(self, dense_embeddings: List[Dict]) -> List[Dict]:
        """Create upsert format for dense index"""
        upserts = []
        
        for dense_obj in dense_embeddings:
            upsert = {
                'id': dense_obj['metadata']['chunk_id'],
                'values': dense_obj['embedding'],
                'metadata': dense_obj['metadata']
            }
            upserts.append(upsert)
        
        return upserts
    
    def create_sparse_upserts(self, sparse_embeddings: List[Dict]) -> List[Dict]:
        """Create upsert format for sparse index"""
        upserts = []
        
        # Get the sparse dimension from config
        sparse_dim = self.config.get('sparse_top_k', 1000)
        
        for sparse_obj in sparse_embeddings:
            # Skip if sparse embedding is empty
            sparse_emb = sparse_obj['embedding']
            if not sparse_emb or not sparse_emb.get('indices') or not sparse_emb.get('values'):
                logger.warning(f"Skipping empty sparse embedding for chunk {sparse_obj['metadata']['chunk_id']}")
                continue
                
            # For sparse-only index, create a dense vector with the top sparse values
            # This allows Pinecone to store the document even though we only use sparse search
            placeholder_vector = [0.0] * sparse_dim
            
            # Fill the placeholder with the actual sparse values (up to sparse_dim)
            for i, (idx, val) in enumerate(zip(sparse_emb['indices'], sparse_emb['values'])):
                if i < sparse_dim:  # Don't exceed the dimension
                    # Use the index position directly, but ensure it's within bounds
                    pos = min(i, sparse_dim - 1)
                    placeholder_vector[pos] = val
                else:
                    break
            
            # Ensure at least one non-zero value
            if all(v == 0.0 for v in placeholder_vector):
                placeholder_vector[0] = 0.1
            
            upsert = {
                'id': sparse_obj['metadata']['chunk_id'],
                'values': placeholder_vector,  # Placeholder with actual sparse values
                'sparse_values': sparse_emb,   # This is what's actually used for search
                'metadata': sparse_obj['metadata']
            }
            upserts.append(upsert)
        
        logger.info(f"Created {len(upserts)} sparse upserts from {len(sparse_embeddings)} embeddings")
        return upserts
    
    def initialize_indices(self, dense_index_name: str, sparse_index_name: str, 
                          dense_dimension: int = 1024, sparse_dimension: int = 1000, reset_db = True):
        """Initialize separate Pinecone indices for dense and sparse vectors"""
        try:
            if reset_db:
                # First, initialize a Pinecone client to check for existing indices
                pc, _ = utils.initialize_pinecone("temp", dimension=1024, metric='dotproduct')
                
                # Delete existing indices if they exist
                for index_name in [dense_index_name, sparse_index_name]:
                    try:
                        pc.delete_index(index_name)
                        logger.info(f"Attempting to delete db index {index_name}")
                        time.sleep(2)  # Wait longer for deletion to complete
                    except Exception as e:
                        logger.info(f"Index {index_name} doesn't exist: nothing to delete...")

                pc.delete_index("temp")

            # Create dense index
            self.pc, self.dense_index = utils.initialize_pinecone(
                dense_index_name, 
                dimension=dense_dimension, 
                metric='dotproduct'
            )
            
            # Create sparse index with dimension matching sparse_top_k
            _, self.sparse_index = utils.initialize_pinecone(
                sparse_index_name, 
                dimension=sparse_dimension,  # Use sparse_top_k as dimension
                metric='dotproduct'
            )
            
            time.sleep(2)  # Wait longer for indices to be ready
            
            # Verify index dimensions
            # dense_desc = self.dense_index.describe_index_stats()
            # sparse_desc = self.sparse_index.describe_index_stats()
            
            logger.info(f"Dense index created successfully - dimension: {dense_dimension}")
            logger.info(f"Sparse index created successfully - dimension: {sparse_dimension}")
        
        except Exception as e:
            logger.error(f"Error initializing indices: {e}")
            raise
    
    def populate_indices(self, dense_embeddings: List[Dict], sparse_embeddings: List[Dict]):
        """Populate both dense and sparse indices"""
        
        # Check if indices are already populated
        dense_stats = self.dense_index.describe_index_stats()
        sparse_stats = self.sparse_index.describe_index_stats()
        
        batch_size = self.config.get('upsert_batch_size', 100)
        
        # Populate dense index
        if dense_stats['total_vector_count'] == 0:
            logger.info("Populating dense index...")
            dense_upserts = self.create_dense_upserts(dense_embeddings)
            
            for i in tqdm(range(0, len(dense_upserts), batch_size), desc="Upserting to dense index"):
                batch = dense_upserts[i:i + batch_size]
                try:
                    self.dense_index.upsert(batch)
                except Exception as e:
                    logger.error(f"Error upserting dense batch {i}: {e}")
                    exit()
        else:
            logger.info("Dense index already populated")
        
        # Populate sparse index
        if sparse_stats['total_vector_count'] == 0:
            logger.info("Populating sparse index...")
            sparse_upserts = self.create_sparse_upserts(sparse_embeddings)
            
            # Add validation before upserting
            if not sparse_upserts:
                logger.error("No valid sparse upserts created - check SPLADE embeddings")
                exit()
                
            logger.info(f"Starting to upsert {len(sparse_upserts)} sparse vectors")
            
            for i in tqdm(range(0, len(sparse_upserts), batch_size), desc="Upserting to sparse index"):
                batch = sparse_upserts[i:i + batch_size]
                try:
                    # Add validation for each item in batch
                    valid_batch = []
                    for item in batch:
                        if (item.get('sparse_values') and 
                            item['sparse_values'].get('indices') and 
                            item['sparse_values'].get('values') and
                            len(item['sparse_values']['indices']) > 0):
                            valid_batch.append(item)
                        else:
                            logger.warning(f"Skipping invalid sparse item: {item.get('id', 'unknown')}")
                    
                    if valid_batch:
                        result = self.sparse_index.upsert(valid_batch)
                        logger.debug(f"Upserted batch {i}: {result}")
                    else:
                        logger.warning(f"No valid items in batch {i}")
                        
                except Exception as e:
                    logger.error(f"Error upserting sparse batch {i}: {e}")
                    # Log the problematic batch for debugging
                    logger.debug(f"Problematic batch items: {[item.get('id', 'unknown') for item in batch]}")
                    exit()
                    continue
        else:
            logger.info("Sparse index already populated")
        
        time.sleep(2)
        
        # Report final stats
        final_dense_stats = self.dense_index.describe_index_stats()
        final_sparse_stats = self.sparse_index.describe_index_stats()
        
        logger.info(f"Dense index populated with {final_dense_stats['total_vector_count']} vectors")
        logger.info(f"Sparse index populated with {final_sparse_stats['total_vector_count']} vectors")

    @staticmethod
    def extract_leading_number(path: str) -> int:
        """Extract leading number from filename for sorting"""
        import re
        filename = path.split('/')[-1]
        match = re.match(r'^(\d+)', filename)
        return int(match.group(1)) if match else float('inf')

    def splade_v3(self, text: str, device: str = 'cuda') -> Dict[str, List]:
            """Improved SPLADE v3 embedding with proper memory management"""
            if not text or not text.strip():
                logger.warning("Empty text provided to SPLADE")
                return {"indices": [], "values": []}
                
            tokens = self.sparse_tokenizer(
                text, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            
            with torch.no_grad():
                tokens_device = {k: v.to(device) for k, v in tokens.items()}
                
                try:
                    doc_rep = self.sparse_model(d_kwargs=tokens_device)
                    if isinstance(doc_rep, dict) and "d_rep" in doc_rep:
                        doc_rep = doc_rep["d_rep"].squeeze()
                    else:
                        raise AttributeError("d_kwargs method failed")
                        
                except (AttributeError, KeyError, TypeError) as e:
                    logger.debug(f"d_kwargs method failed: {e}, trying alternative methods")
                    
                    try:
                        doc_rep = self.sparse_model(**tokens_device)
                        
                        if isinstance(doc_rep, dict):
                            if "d_rep" in doc_rep:
                                doc_rep = doc_rep["d_rep"].squeeze()
                            elif "last_hidden_state" in doc_rep:
                                doc_rep = doc_rep["last_hidden_state"].mean(dim=1).squeeze()
                            elif "logits" in doc_rep:
                                doc_rep = doc_rep["logits"].squeeze()
                            elif len(doc_rep) > 0:
                                values = list(doc_rep.values())
                                if values:
                                    doc_rep = values[0].squeeze()
                                else:
                                    raise ValueError("Empty model output dictionary")
                            else:
                                raise ValueError("No suitable output found in model response")
                        else:
                            doc_rep = doc_rep.squeeze()
                            
                    except Exception as e2:
                        logger.error(f"All SPLADE model call methods failed: {e2}")
                        return {"indices": [], "values": []}
                
                if doc_rep is None or not torch.is_tensor(doc_rep):
                    logger.error("SPLADE model did not return a valid tensor")
                    return {"indices": [], "values": []}
                
                doc_rep_cpu = doc_rep.cpu()
                del doc_rep
                del tokens_device
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            non_zero_mask = doc_rep_cpu != 0
            non_zero_count = non_zero_mask.sum().item()
            
            if non_zero_count == 0:
                logger.warning("SPLADE produced all-zero embedding")
                return {"indices": [], "values": []}
            
            indices = non_zero_mask.nonzero().squeeze()
            values = doc_rep_cpu[non_zero_mask]
            
            if indices.dim() == 0:
                indices = indices.unsqueeze(0)
                values = values.unsqueeze(0)
            
            top_k = min(self.config.get('sparse_top_k', 1000), len(values))
            if len(values) > top_k:
                top_values, top_indices = torch.topk(values, top_k)
                final_indices = indices[top_indices]
                final_values = top_values
            else:
                final_indices = indices
                final_values = values
            
            result = {
                "indices": final_indices.tolist(),
                "values": final_values.tolist()
            }
            
            del doc_rep_cpu, indices, values, non_zero_mask
            if 'final_indices' in locals():
                del final_indices, final_values
            
            return result
    
    def search_dense(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Perform dense-only search"""
        try:
            dense_query = utils.runDense(
                self.dense_model, 
                self.dense_tokenizer, 
                [query]
            )[0]
            
            if isinstance(dense_query[0], list):
                dense_query = dense_query[0]
            
            results = self.dense_index.query(
                vector=dense_query,
                top_k=top_k,
                include_metadata=True
            )
            # Print dense search results
            print(f"\n=== DENSE SEARCH RESULTS (Top {top_k}) ===")
            for i, match in enumerate(results['matches']):
                print(f"Dense {i+1}: {match['id']} - Score: {match['score']:.4f}")
            
                
            logger.info(f"Dense search retrieved {len(results['matches'])} results")
            return results
            
        except Exception as e:
            logger.error(f"Error during dense search: {e}")
            return {'matches': []}
    
    def search_sparse(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """Perform sparse-only search"""
        try:
            sparse_query = self.splade_v3(query)
            
            sparse_dim = self.config.get('sparse_top_k', 1000)
            placeholder_vector = [0.0] * sparse_dim
            
            results = self.sparse_index.query(
                vector=placeholder_vector,
                sparse_vector=sparse_query,
                top_k=top_k,
                include_metadata=True
            )

            # Print sparse search results
            print(f"\n=== SPARSE SEARCH RESULTS (Top {top_k}) ===")
            for i, match in enumerate(results['matches']):
                print(f"Sparse {i+1}: {match['id']} - Score: {match['score']:.4f}")
            
            logger.info(f"Sparse search retrieved {len(results['matches'])} results")
            return results
            
        except Exception as e:
            logger.error(f"Error during sparse search: {e}")
            return {'matches': []}
    
    def combine_results_rrf(self, dense_results: Dict[str, Any], sparse_results: Dict[str, Any], 
                           k: int = 60) -> Dict[str, Any]:
        """Combine results using Reciprocal Rank Fusion (RRF)"""
        try:
            dense_ranks = {match['id']: rank + 1 for rank, match in enumerate(dense_results['matches'])}
            sparse_ranks = {match['id']: rank + 1 for rank, match in enumerate(sparse_results['matches'])}
            
            metadata_dict = {}
            for match in dense_results['matches'] + sparse_results['matches']:
                metadata_dict[match['id']] = match['metadata']
            
            all_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())
            
            combined_matches = []
            
            for doc_id in all_ids:
                rrf_score = 0
                retrieval_sources = []
                
                if doc_id in dense_ranks:
                    rrf_score += 1 / (k + dense_ranks[doc_id])
                    retrieval_sources.append('dense')
                
                if doc_id in sparse_ranks:
                    rrf_score += 1 / (k + sparse_ranks[doc_id])
                    retrieval_sources.append('sparse')
                
                combined_match = {
                    'id': doc_id,
                    'score': rrf_score,
                    'metadata': metadata_dict[doc_id],
                    'retrieval_source': '+'.join(retrieval_sources),
                    'dense_rank': dense_ranks.get(doc_id, None),
                    'sparse_rank': sparse_ranks.get(doc_id, None)
                }
                
                combined_matches.append(combined_match)
            
            combined_matches.sort(key=lambda x: x['score'], reverse=True)
            
            combined_results = {
                'matches': combined_matches,
                'namespace': dense_results.get('namespace', ''),
                'usage': {
                    'read_units': dense_results.get('usage', {}).get('read_units', 0) + 
                             sparse_results.get('usage', {}).get('read_units', 0)
                }
            }
            
            logger.info(f"RRF combined {len(dense_results['matches'])} dense + {len(sparse_results['matches'])} sparse = {len(combined_matches)} total results")
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in RRF combination: {e}")
            return dense_results
    
    def combine_results(self, dense_results: Dict[str, Any], sparse_results: Dict[str, Any], 
                       alpha: float = 0.5) -> Dict[str, Any]:
        """Combine dense and sparse results using score-based fusion"""
        try:
            dense_dict = {match['id']: match for match in dense_results['matches']}
            sparse_dict = {match['id']: match for match in sparse_results['matches']}
            
            all_ids = set(dense_dict.keys()) | set(sparse_dict.keys())
            
            combined_matches = []
            
            for doc_id in all_ids:
                dense_match = dense_dict.get(doc_id)
                sparse_match = sparse_dict.get(doc_id)
                
                if dense_match and sparse_match:
                    combined_score = alpha * dense_match['score'] + (1 - alpha) * sparse_match['score']
                    metadata = dense_match['metadata']
                elif dense_match:
                    combined_score = alpha * dense_match['score']
                    metadata = dense_match['metadata']
                elif sparse_match:
                    combined_score = (1 - alpha) * sparse_match['score']
                    metadata = sparse_match['metadata']
                else:
                    continue
                
                combined_match = {
                    'id': doc_id,
                    'score': combined_score,
                    'metadata': metadata,
                    'retrieval_source': 'both' if (dense_match and sparse_match) else 
                                      ('dense' if dense_match else 'sparse')
                }
                
                combined_matches.append(combined_match)
            
            combined_matches.sort(key=lambda x: x['score'], reverse=True)
            
            combined_results = {
                'matches': combined_matches,
                'namespace': dense_results.get('namespace', ''),
                'usage': {
                    'read_units': dense_results.get('usage', {}).get('read_units', 0) + 
                             sparse_results.get('usage', {}).get('read_units', 0)
                }
            }
            
            logger.info(f"Combined {len(dense_results['matches'])} dense + {len(sparse_results['matches'])} sparse = {len(combined_matches)} total results")
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error combining results: {e}")
            return dense_results
    
   
    ### Graph Nodes ###

    def check_history_sufficiency(self, state):
        """Check if the chat history is sufficient to answer the query."""
        print(f"\n{Fore.CYAN}🤔 ---CHECKING HISTORY SUFFICIENCY---{Style.RESET_ALL}")
        
        question = state["question"]
        chat_context = state.get("chat_context", "")
        self.increment_iteration(state)
        
        # If there's only one message in memory (the current one), history is not sufficient.
        if len(self.chat_memory.messages) <= 1:
            print(f"{Fore.YELLOW}No prior chat history available. Proceeding to retrieval.{Style.RESET_ALL}")
            state["history_is_sufficient"] = "no"
            return state

        try:
            prompt = f"""You are an expert at analyzing conversations. Your task is to determine if the provided "Chat History" contains enough information to directly answer the "New Question".

            **Analysis Criteria:**
            - The answer must be explicitly present or directly inferable from the history.
            - Do not assume you can answer from general knowledge. The answer must come from the provided text.
            - If the question asks for new details not present in the history, even if the topic is related, the history is NOT sufficient.

            **Chat History:**
            {chat_context}

            **New Question:**
            {question}

            Can the "New Question" be answered using ONLY the "Chat History"? Respond with only "yes" or "no"."""

            response = self.llm.invoke(prompt)
            decision = response.content.strip().lower()
            
            if decision == "yes":
                print(f"{Fore.GREEN}✅ History is sufficient. Generating answer from context.{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}⚠️ History is not sufficient. Proceeding to retrieval.{Style.RESET_ALL}")
            
            state["history_is_sufficient"] = decision

        except Exception as e:
            print(f"{Fore.RED}❌ Error checking history sufficiency: {e}. Defaulting to retrieval.{Style.RESET_ALL}")
            state["history_is_sufficient"] = "no"

        return state

    def generate_from_history(self, state):
        """Generate an answer using only the chat history."""
        print(f"\n{Fore.GREEN}💬 ---GENERATING ANSWER FROM HISTORY---{Style.RESET_ALL}")
        
        question = state["question"]
        chat_context = state.get("chat_context", "")
        self.increment_iteration(state)

        try:
            prompt = f"""You are a helpful assistant. Use the provided "Chat History" to answer the "User's Question".
            If the history does not contain the answer, state that you cannot answer based on the conversation so far.

            Chat History:
            {chat_context}

            User's Question:
            {question}

            Your Answer:"""

            generation = self.llm.invoke(prompt).content.strip()
            print(f"{Fore.GREEN}💬 Generated answer from history (preview): {generation[:100]}...{Style.RESET_ALL}")
            state["generation"] = generation
        
        except Exception as e:
            error_msg = f"Sorry, I encountered an error while trying to answer from our conversation history: {e}"
            print(f"{Fore.RED}❌ {error_msg}{Style.RESET_ALL}")
            state["generation"] = error_msg

        return state

    def retrieve_dense(self, state):
        """Retrieve documents using dense search"""
        print(f"\n{Fore.BLUE}🔍 ---DENSE RETRIEVE---{Style.RESET_ALL}")
        
        # Increment iteration and check limits
        iteration_count = self.increment_iteration(state)
        
        question = state["question"]
        chat_context = state.get("chat_context", "")  
        
        try:
            documents = self.search_dense(
                question, 
                top_k=self.config.get('dense_search_top_k', 5)
            )
            print(f"{Fore.GREEN}📄 Retrieved {len(documents['matches'])} dense documents{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Error in dense retrieval: {e}{Style.RESET_ALL}")
            documents = {"matches": []}
        
        return {
            "dense_documents": documents, 
            "question": question,
            "original_question": state.get("original_question", question),
            "chat_context": chat_context,
            "fusion_method": state.get("fusion_method", "rrf"),
            "alpha": state.get("alpha", 0.5),
            "rrf_k": state.get("rrf_k", 60),
            "iteration_count": iteration_count,
            "start_time": state.get("start_time", time.time()),
            "max_iterations": state.get("max_iterations", 15),
            "max_time_seconds": state.get("max_time_seconds", 60.0),
            "hallucination_retry_count": state.get("hallucination_retry_count", 0)
        }
    
    def retrieve_sparse(self, state):
        """Retrieve documents using sparse search"""
        print(f"\n{Fore.YELLOW}🕸️ ---SPARSE RETRIEVE---{Style.RESET_ALL}")
        
        # Increment iteration and check limits
        iteration_count = self.increment_iteration(state)
        
        question = state["question"]
        chat_context = state.get("chat_context", "")  
        
        try:
            documents = self.search_sparse(
                question, 
                top_k=self.config.get('sparse_search_top_k', 5)
            )
            print(f"{Fore.GREEN}📄 Retrieved {len(documents['matches'])} sparse documents{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Error in sparse retrieval: {e}{Style.RESET_ALL}")
            documents = {"matches": []}
        
        return {
            "sparse_documents": documents, 
            "dense_documents": state.get("dense_documents", {"matches": []}),
            "question": question,
            "original_question": state.get("original_question", question),
            "chat_context": chat_context,
            "fusion_method": state.get("fusion_method", "rrf"),
            "alpha": state.get("alpha", 0.5),
            "rrf_k": state.get("rrf_k", 60),
            "iteration_count": iteration_count,
            "start_time": state.get("start_time", time.time()),
            "max_iterations": state.get("max_iterations", 15),
            "max_time_seconds": state.get("max_time_seconds", 60.0),
            "hallucination_retry_count": state.get("hallucination_retry_count", 0)
        }
    
    def late_fusion(self, state):
        """Combine dense and sparse results using late fusion"""
        print(f"\n{Fore.MAGENTA}🔗 ---LATE FUSION---{Style.RESET_ALL}")
        
        # Increment iteration and check limits
        iteration_count = self.increment_iteration(state)
        
        dense_documents = state["dense_documents"]
        sparse_documents = state["sparse_documents"]
        fusion_method = state.get("fusion_method")
        alpha = state.get("alpha", 0.5)
        rrf_k = state.get("rrf_k", 60)
        question = state["question"]
        chat_context = state.get("chat_context", "")  
        
        try:
            if fusion_method.lower() == "rrf":
                combined_documents = self.combine_results_rrf(
                    dense_documents, sparse_documents, rrf_k
                )
            else:
                combined_documents = self.combine_results(
                    dense_documents, sparse_documents, alpha
                )
            

            # If not using reranker, truncate the results here to the top k.
            if not self.config.get('use_reranker', True):
                top_k = self.config.get('rerank_top_k', 3)
                print(f"{Fore.CYAN}✂️ Reranker is disabled. Truncating to top {top_k} combined results.{Style.RESET_ALL}")
                if combined_documents and combined_documents.get("matches"):
                    combined_documents["matches"] = combined_documents["matches"][:top_k]
            
            print(f"\n{Fore.CYAN}⚡ Late fusion search results ({fusion_method} method):{Style.RESET_ALL}")
            # Print combined results
            print(f"\n{Fore.CYAN}🎯 === COMBINED RESULTS ({fusion_method.upper()} METHOD) ==={Style.RESET_ALL}")

            for i, match in enumerate(combined_documents['matches']):
                retrieval_info = f" [{match.get('retrieval_source', 'unknown')}]"
                if fusion_method.lower() == 'rrf':
                    rank_info = f" (Dense: {match.get('dense_rank', 'N/A')}, Sparse: {match.get('sparse_rank', 'N/A')})"
                else:
                    rank_info = ""
                print(f"📋 Combined {i+1}: {match['id']} - Score: {match['score']:.4f}{retrieval_info}{rank_info}")

            print(f"{Fore.GREEN}✅ Combined into {len(combined_documents['matches'])} documents using {fusion_method}{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}❌ Error in late fusion: {e}{Style.RESET_ALL}")
            combined_documents = dense_documents  # Fallback to dense
        
        return {
            "combined_documents": combined_documents,
            "dense_documents": dense_documents,
            "sparse_documents": sparse_documents,
            "question": question,
            "original_question": state.get("original_question", question),
            "chat_context": chat_context,
            "alpha": alpha,
            "rrf_k": rrf_k,
            "iteration_count": iteration_count,
            "start_time": state.get("start_time", time.time()),
            "max_iterations": state.get("max_iterations", 15),
            "max_time_seconds": state.get("max_time_seconds", 60.0),
        }
    
    def rerank(self, state):
        """Rerank the combined documents"""
        print(f"\n{Fore.CYAN}📊 ---RERANK DOCUMENTS---{Style.RESET_ALL}")
        question = state["question"]
        documents = state["combined_documents"]
        chat_context = state.get("chat_context", "") 
        
        # Increment iteration and check limits
        iteration_count = self.increment_iteration(state)

        try:
            reranked_documents = utils.reranker(
                self.reranker, 
                self.reranker_tokenizer, 
                question, 
                documents, 
                top_k=self.config.get('rerank_top_k')
            )

            print(f"\n{Fore.CYAN}🏆 Top {self.config.get('rerank_top_k')} reranked results:{Style.RESET_ALL}")
            # print(reranked_documents["matches"][0])
            # kkkk
            for i, match in enumerate(reranked_documents['matches']):
                retrieval_info = f" [{match.get('retrieval_source', 'unknown')}]"
                chunk_id = match['metadata']['chunk_id']
                paragraph_related = match['metadata'].get('paragraph_related', 'N/A')
                print(f"🥇 {i+1}. Chunk {chunk_id} (Paragraph: {paragraph_related}){retrieval_info}")
            
            print(f"{Fore.GREEN}✅ Re-ranked to {len(reranked_documents['matches'])} documents{Style.RESET_ALL}")
        
        except Exception as e:
            print(f"{Fore.RED}❌ Error in reranking: {e}{Style.RESET_ALL}")
            reranked_documents = documents
    
        return {
            "combined_documents": reranked_documents,
            "dense_documents": state["dense_documents"],
            "sparse_documents": state["sparse_documents"],
            "question": question,
            "original_question": state.get("original_question", question),
            "chat_context": chat_context,
            "fusion_method": state["fusion_method"],
            "alpha": state["alpha"],
            "rrf_k": state["rrf_k"],
            "iteration_count": iteration_count,
            "start_time": state.get("start_time", time.time()),
            "max_iterations": state.get("max_iterations", 15),
            "max_time_seconds": state.get("max_time_seconds", 60.0),
        }
    
    def grade_documents(self, state):
        """Grade document relevance to question"""
        
        print(f"\n{Fore.BLUE}🎯 === FILTERING CHUNKS FOR RELEVANCE ==={Style.RESET_ALL}")
        print(f"📊 Total chunks to evaluate: {len(state['combined_documents']['matches'])}")
        
        question = state["question"]
        documents = state["combined_documents"]
        chat_context = state.get("chat_context", "")  

        # Increment iteration and check limits
        iteration_count = self.increment_iteration(state)

        filtered_docs = []
        i=0
        for doc in documents["matches"]:
            doc_text = doc['metadata']['context']
            
            paragraph_info = doc['metadata'].get('paragraph_related', 'N/A')
            print(f"\n{Fore.YELLOW}🔍 Evaluating chunk {i+1}/{len(documents['matches'])} (ID: {doc['id']}, Paragraph: {paragraph_info}), length: {len(doc_text)} characters{Style.RESET_ALL}")
            
            score = retrieval_grader(self.llm, doc_text, question)
            
            if score == "yes":                
                print(f"{Fore.GREEN}✅ RELEVANT - Adding to filtered results{Style.RESET_ALL}")
                filtered_docs.append(doc)
            else:
                print(f"{Fore.RED}❌ NOT RELEVANT - Excluding from results{Style.RESET_ALL}")
                continue
            i=i+1

        filtered_result = {"matches": filtered_docs}

        return {
            "combined_documents": filtered_result,
            "dense_documents": state["dense_documents"],
            "sparse_documents": state["sparse_documents"],
            "question": question,
            "original_question": state.get("original_question", question),
            "chat_context": chat_context,
            "fusion_method": state["fusion_method"],
            "alpha": state["alpha"],
            "rrf_k": state["rrf_k"],
            "iteration_count": iteration_count,
            "start_time": state.get("start_time", time.time()),
            "max_iterations": state.get("max_iterations", 15),
            "max_time_seconds": state.get("max_time_seconds", 60.0),
        }
    
   
    def generate(self, state):
        """Generate answer using retrieved documents"""
        print(f"\n{Fore.GREEN}🤖 ---GENERATE ANSWER---{Style.RESET_ALL}")
        question = state["question"]
        documents = state["combined_documents"]
        chat_context = state.get("chat_context", "")  

        # Increment iteration and check limits
        iteration_count = self.increment_iteration(state)

        doc_content = []
        for doc in documents["matches"]:
            doc_content.append(doc["metadata"].get("context", ""))
        
        try:
            generation = generator(self.llm, "\n\n--- PARAGRAPH SEPARATOR ---\n\n".join(doc_content), question)
            print(f"{Fore.GREEN}💬 Generated answer (preview): {generation[:100]}...{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Error in generation: {e}{Style.RESET_ALL}")
            generation = f"Error generating answer: {e}"
        
        return {
            "combined_documents": documents, 
            "question": question, 
            "original_question": state.get("original_question", question),
            "chat_context": chat_context,
            "generation": generation,
            "dense_documents": state["dense_documents"],
            "sparse_documents": state["sparse_documents"],
            "fusion_method": state["fusion_method"],
            "alpha": state["alpha"],
            "rrf_k": state["rrf_k"],
            "iteration_count": iteration_count,
            "start_time": state.get("start_time", time.time()),
            "max_iterations": state.get("max_iterations", 15),
            "max_time_seconds": state.get("max_time_seconds", 60.0),
        }
    
    def regenerate_after_hallucination(self, state):
        """Regenerate answer after hallucination failure and increment retry count"""
        print(f"\n{Fore.YELLOW}🔄 ---REGENERATE AFTER HALLUCINATION---{Style.RESET_ALL}")
        question = state["question"]
        documents = state["combined_documents"]
        retry_count = state.get("hallucination_retry_count", 0) + 1
        chat_context = state.get("chat_context", "")  

        # Increment iteration and check limits
        iteration_count = self.increment_iteration(state)

        doc_content = []
        for doc in documents["matches"]:
            doc_content.append(doc["metadata"].get("context", ""))
        
        try:
            generation = generator(self.llm, "\n\n--- PARAGRAPH SEPARATOR ---\n\n".join(doc_content), question)
            print(f"{Fore.YELLOW}🔄 Regenerated answer (attempt {retry_count}): {generation[:100]}...{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Error in regeneration: {e}{Style.RESET_ALL}")
            generation = f"Error generating answer: {e}"
        
        return {
            "combined_documents": documents, 
            "question": question, 
            "original_question": state.get("original_question", question),
            "chat_context": chat_context,
            "generation": generation,
            "hallucination_retry_count": retry_count,
            "dense_documents": state["dense_documents"],
            "sparse_documents": state["sparse_documents"],
            "fusion_method": state["fusion_method"],
            "alpha": state["alpha"],
            "rrf_k": state["rrf_k"],
            "iteration_count": iteration_count,
            "start_time": state.get("start_time", time.time()),
            "max_iterations": state.get("max_iterations", 15),
            "max_time_seconds": state.get("max_time_seconds", 60.0),
        }
    
    def transform_query(self, state):
        """Transform the query to improve retrieval"""
        
        print(f"\n{Fore.MAGENTA}✨ ---TRANSFORM QUERY---{Style.RESET_ALL}")
        question = state["question"]
        chat_context = state.get("chat_context", "")  

        # Increment iteration and check limits
        iteration_count = self.increment_iteration(state)

        try:
            # First try context-aware enhancement if we have chat history
            if chat_context:
                enhanced_question = self.enhance_query_with_context(question, chat_context)
            else:
                enhanced_question = question
        
            # Then apply the regular question rewriter
            better_question = question_rewriter(self.llm, enhanced_question)
            print(f"{Fore.MAGENTA}🔄 Query transformed: {question} -> {better_question}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Error in query rewriting: {e}{Style.RESET_ALL}")
            better_question = question
    
        return {
            "question": better_question,
            "original_question": state.get("original_question", question),
            "chat_context": chat_context,
            "fusion_method": state["fusion_method"],
            "alpha": state["alpha"],
            "rrf_k": state["rrf_k"],
            "iteration_count": iteration_count,
            "start_time": state.get("start_time", time.time()),
            "max_iterations": state.get("max_iterations", 15),
            "max_time_seconds": state.get("max_time_seconds", 60.0),
        }

    def grade_hallucination(self, state):
        """Check if generation is grounded in documents"""
        print(f"\n{Fore.RED}🚨 CHECK HALLUCINATIONS{Style.RESET_ALL}")
        generation = state["generation"]
        documents = state["combined_documents"]
        retry_count = state.get("hallucination_retry_count", 0)
        question = state["question"]
        chat_context = state.get("chat_context", "")  
    
        # Increment iteration and check limits
        iteration_count = self.increment_iteration(state)

        doc_content = []
        for doc in documents["matches"]:
            doc_content.append(doc["metadata"].get("context", ""))
        
        try:
            score = hallucination_grader(self.llm, doc_content, generation)
            
            if score == "yes":
                print(f"{Fore.GREEN}Hallucination Score (Retry count: {retry_count}): 🟢 ⇒  GENERATION IS GROUNDED IN DOCUMENTS{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Hallucination Score (Retry count: {retry_count}): 🔴 ⇒  GENERATION IS NOT GROUNDED IN DOCUMENTS{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ Error in hallucination grading: {e}{Style.RESET_ALL}")
            score = "yes"
    
        return {
            "combined_documents": documents,
            "question": question,
            "original_question": state.get("original_question", question),
            "chat_context": chat_context,
            "generation": generation,
            "hallucination_score": score,
            "hallucination_retry_count": retry_count,
            "dense_documents": state["dense_documents"],
            "sparse_documents": state["sparse_documents"],
            "fusion_method": state["fusion_method"],
            "alpha": state["alpha"],
            "rrf_k": state["rrf_k"],
            "iteration_count": iteration_count,
            "start_time": state.get("start_time", time.time()),
            "max_iterations": state.get("max_iterations", 15),
            "max_time_seconds": state.get("max_time_seconds", 60.0),
        }

    def grade_answer_initial(self, state):
        """Check if generation addresses the question (initial attempt with chunks)"""
        print(f"\n{Fore.BLUE}📝 ---GRADE ANSWER (INITIAL ATTEMPT WITH CHUNKS)---{Style.RESET_ALL}")
        question = state["question"]
        generation = state["generation"]
        chat_context = state.get("chat_context", "")  

        # Increment iteration and check limits
        iteration_count = self.increment_iteration(state)

        try:

            score = answer_grader(self.llm, question, generation)
            
            if score == "yes":
                print(f"{Fore.GREEN}Answer Score: 🟢 ⇒ GENERATION ADDRESSES QUESTION (CHUNKS SUFFICIENT){Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Answer Score: 🔴 ⇒ GENERATION DOES NOT ADDRESS QUESTION (NEED PARAGRAPHS){Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ Error in initial answer grading: {e}{Style.RESET_ALL}")
            score = "yes"
    
        return {
            "combined_documents": state["combined_documents"],
            "question": question,
            "original_question": state.get("original_question", question),
            "chat_context": chat_context,
            "generation": generation,
            "answer_score": score,
            "hallucination_score": state.get("hallucination_score", ""),
            "dense_documents": state["dense_documents"],
            "sparse_documents": state["sparse_documents"],
            "fusion_method": state["fusion_method"],
            "alpha": state["alpha"],
            "rrf_k": state["rrf_k"],
            "attempt_type": "chunks",
            "iteration_count": iteration_count,
            "start_time": state.get("start_time", time.time()),
            "max_iterations": state.get("max_iterations", 15),
            "max_time_seconds": state.get("max_time_seconds", 60.0),
        }

    def generate_with_paragraphs(self, state):
        """Generate answer using complete paragraphs (fallback strategy)"""
        print(f"\n{Fore.GREEN}📚 ---GENERATE ANSWER WITH PARAGRAPHS (FALLBACK)---{Style.RESET_ALL}")
        question = state["question"]
        documents = state["combined_documents"]
        chat_context = state.get("chat_context", "")  

        # Increment iteration and check limits
        iteration_count = self.increment_iteration(state)

        doc_content = []
        #print(len(documents["matches"]))
        for doc in documents["matches"]:
            doc_content.append(doc["metadata"].get("context", ""))
        
        try:            
            
            generation = generator(self.llm, "\n\n--- PARAGRAPH SEPARATOR ---\n\n".join(doc_content), question)
            
            print(f"{Fore.GREEN}💬 Generated answer with paragraphs (preview): {generation[:100]}...{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Error in paragraph generation: {e}{Style.RESET_ALL}")
            generation = f"Error generating answer: {e}"
        
        return {
            "combined_documents": documents, 
            "question": question, 
            "original_question": state.get("original_question", question),
            "chat_context": chat_context,
            "generation": generation,
            "attempt_type": "paragraphs",
            "dense_documents": state["dense_documents"],
            "sparse_documents": state["sparse_documents"],
            "fusion_method": state["fusion_method"],
            "alpha": state["alpha"],
            "rrf_k": state["rrf_k"],
            "iteration_count": iteration_count,
            "start_time": state.get("start_time", time.time()),
            "max_iterations": state.get("max_iterations", 15),
            "max_time_seconds": state.get("max_time_seconds", 60.0),
        }

    def grade_answer_paragraphs(self, state):
        """Check if generation with paragraphs addresses the question"""
        print(f"\n{Fore.BLUE}📚 ---GRADE ANSWER (PARAGRAPH ATTEMPT)---{Style.RESET_ALL}")
        question = state["question"]
        generation = state["generation"]
        chat_context = state.get("chat_context", "")  
    
        # Increment iteration and check limits
        iteration_count = self.increment_iteration(state)

        try:
            score = answer_grader(self.llm, question, generation)
            
            if score == "yes":
                print(f"{Fore.GREEN}Answer Score: 🟢 ⇒ GENERATION ADDRESSES QUESTION (PARAGRAPHS SUFFICIENT){Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Answer Score: 🔴 ⇒ GENERATION DOES NOT ADDRESS QUESTION (NEED QUERY TRANSFORM){Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ Error in paragraph answer grading: {e}{Style.RESET_ALL}")
            score = "yes"
    
        return {
            "combined_documents": state["combined_documents"],
            "question": question,
            "original_question": state.get("original_question", question),
            "chat_context": chat_context,
            "generation": generation,
            "answer_score": score,
            "hallucination_score": state.get("hallucination_score", ""),
            "dense_documents": state["dense_documents"],
            "sparse_documents": state["sparse_documents"],
            "fusion_method": state["fusion_method"],
            "alpha": state["alpha"],
            "rrf_k": state["rrf_k"],
            "attempt_type": state.get("attempt_type", "paragraphs"),
            "iteration_count": iteration_count,
            "start_time": state.get("start_time", time.time()),
            "max_iterations": state.get("max_iterations", 15),
            "max_time_seconds": state.get("max_time_seconds", 60.0),
        }


    def early_exit(self, state):
        """Handle early exit conditions"""
        current_time = time.time()
        elapsed_time = current_time - state.get("start_time", current_time)
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 15)
        max_time = state.get("max_time_seconds", 60.0)
        question = state.get("question", "")
        chat_context = state.get("chat_context", "")  
    
        # Determine the reason for early exit
        if iteration_count >= max_iterations:
            exit_reason = f"Maximum iterations reached ({iteration_count}/{max_iterations})"
        elif elapsed_time >= max_time:
            exit_reason = f"Time limit reached ({elapsed_time:.2f}s/{max_time}s)"
        else:
            exit_reason = "Early exit condition met"
            
        # Use the current generation if available, otherwise provide a default message
        generation = state.get("generation", "")
        if not generation:            
            generation = f"🤔 Sorry, I have no answer at the moment for this query..."
        
        return {
            "combined_documents": state.get("combined_documents", {"matches": []}),
            "question": question,
            "original_question": state.get("original_question", question),
            "chat_context": chat_context,
            "generation": generation,
            "dense_documents": state.get("dense_documents", {"matches": []}),
            "sparse_documents": state.get("sparse_documents", {"matches": []}),
            "fusion_method": state.get("fusion_method", "rrf"),
            "alpha": state.get("alpha", 0.5),
            "rrf_k": state.get("rrf_k", 60),
            "iteration_count": iteration_count,
            "start_time": state.get("start_time", current_time),
            "max_iterations": max_iterations,
            "max_time_seconds": max_time,
            "exit_reason": exit_reason
        }
    



    def enhance_query_with_context(self, query: str, chat_context: str) -> str:
        """Enhance query with chat context using LLM"""
        if not chat_context.strip():
            return query
        
        try:
            prompt = f"""Given the conversation context and the new question, reformulate the question to be more specific and clear. 
            If the new question refers to previous context, incorporate that context appropriately.

            Conversation context:
            {chat_context}

            New question: {query}

            Enhanced question (be concise and specific):"""
            
            enhanced_query = self.llm.invoke(prompt).content.strip()
            
            if enhanced_query and len(enhanced_query) > 10:  # Basic validation
                print(f"{Fore.CYAN}🔄 Query enhanced with context: {query} -> {enhanced_query}{Style.RESET_ALL}")
                return enhanced_query
            else:
                return query
                
        except Exception as e:
            logger.warning(f"Failed to enhance query with context: {e}")
            return query


    ### Updated Conditional Edge Functions ###



    def decide_history_or_classification(self, state):
        """Decide whether to generate from history or proceed to query classification."""
        if self.should_exit_early(state):
            return "early_exit"
        
        if state.get("history_is_sufficient") == "yes":
            print(f"\n{Fore.GREEN}📚 ---DECISION: GENERATE FROM HISTORY---{Style.RESET_ALL}")
            return "generate_from_history"
        else:
            print(f"\n{Fore.YELLOW}🔍 ---DECISION: PROCEED TO RETRIEVAL---{Style.RESET_ALL}")
            return "need_retrieval"

    def decide_to_generate(self, state):
        """Decide whether to generate an answer or transform query based on document relevance"""
        
        # Check if we should exit early
        if self.should_exit_early(state):
            return "early_exit"
            
        filtered_documents = state["combined_documents"]

        if not filtered_documents["matches"]:
            print(f"\n{Fore.RED}🚫 ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION ---> TRANSFORMING QUERY{Style.RESET_ALL}")
            return "docs_not_relevant"
        else:
            print(f"\n{Fore.GREEN}✅ Number of relevant docs: {len(filtered_documents['matches'])} ---> DECISION: GENERATE ANSWER!{Style.RESET_ALL}")
            return "docs_relevant"

    def decide_after_hallucination_check(self, state):
        """Decide next step after hallucination check"""
        
        # Check if we should exit early
        if self.should_exit_early(state):
            return "early_exit"
            
        hallucination_score = state.get("hallucination_score", "yes")
        retry_count = state.get("hallucination_retry_count", 0)
        
        if hallucination_score == "yes":
            print(f"{Fore.GREEN}✅ ---DECISION: GENERATION IS GROUNDED, CHECK IF IT ANSWERS QUESTION---{Style.RESET_ALL}")
            return "grounded"
        else:
            if retry_count < 1:  # Allow only 1 retry
                print(f"{Fore.YELLOW}🔄 ---DECISION: GENERATION IS NOT GROUNDED, REGENERATING (ATTEMPT {retry_count + 1})---{Style.RESET_ALL}")
                return "regenerate"
            else:
                print(f"{Fore.BLUE}📖 ---DECISION: GENERATION STILL NOT GROUNDED AFTER RETRY, LOADING PARAGRAPHS---{Style.RESET_ALL}")
                return "load_paragraphs"

    def decide_after_initial_answer_check(self, state):
        """Decide next step after initial answer check with chunks"""
        
        # Check if we should exit early
        if self.should_exit_early(state):
            return "early_exit"
            
        answer_score = state.get("answer_score", "no")
        
        if answer_score == "yes":
            print(f"{Fore.GREEN}🎉 ---DECISION: CHUNKS ARE SUFFICIENT, ENDING---{Style.RESET_ALL}")
            return "chunks_sufficient"
        else:
            print(f"{Fore.BLUE}📚 ---DECISION: CHUNKS NOT SUFFICIENT, TRYING PARAGRAPHS---{Style.RESET_ALL}")
            return "need_paragraphs"

    def decide_after_paragraph_answer_check(self, state):
        """Decide final step after paragraph answer check"""
        
        # Check if we should exit early
        if self.should_exit_early(state):
            return "early_exit"
            
        answer_score = state.get("answer_score", "no")
        
        if answer_score == "yes":
            print(f"{Fore.GREEN}🎉 ---DECISION: PARAGRAPHS ARE SUFFICIENT, ENDING---{Style.RESET_ALL}")
            return "paragraphs_sufficient"
        else:
            print(f"{Fore.MAGENTA}✨ ---DECISION: PARAGRAPHS NOT SUFFICIENT, TRANSFORMING QUERY---{Style.RESET_ALL}")
            return "need_query_transform"


    def classify_initial_intent(self, state):
        """Performs a quick, initial classification of the user's query."""
        print(f"\n{Fore.CYAN}🚦 ---CLASSIFY INITIAL INTENT---{Style.RESET_ALL}")
        
        question = state["question"]
        iteration_count = self.increment_iteration(state)
        
        # Initial classification prompt, no context needed
        initial_classification_prompt = f"""
You are a high-level query router. Your task is to perform a quick, initial classification of the user's query into one of three categories based *only* on the current query text.

1.  **DIRECT_ANSWER**: The query is a general greeting, a simple conversational phrase, asks about your capabilities, or is a personal question directed at you (the AI) that cannot be answered by searching documents. It does not require any document search or context analysis.
    - Examples: "Hello", "Hi", "How are you?", "What can you do for me?", "Who are you?", "Do you know my name?"

2.  **GOODBYE**: The query clearly indicates the user wants to end the conversation.
    - Examples: "Bye", "See you", "Goodbye", "Thanks, that's all", "Exit"

3.  **NEEDS_ANALYSIS**: The query asks a specific question that requires deeper understanding, context from the conversation, or information from documents. This is the default for any substantive question, especially those related to grant funding.
    - Examples: "What is the deadline for submission?", "tell me more", "who can sign the lear?", "Can electronic signatures be used?"

Current query: {question}

Respond with only one word: DIRECT_ANSWER, GOODBYE, or NEEDS_ANALYSIS.
"""
        
        try:
            response = self.llm.invoke(initial_classification_prompt)
            classification = response.content.strip().upper()
            
            if classification not in ["DIRECT_ANSWER", "GOODBYE", "NEEDS_ANALYSIS"]:
                print(f"{Fore.YELLOW}⚠️ Unknown classification '{classification}' → Defaulting to NEEDS_ANALYSIS{Style.RESET_ALL}")
                classification = "NEEDS_ANALYSIS"
            else:
                                                                                                            
                print(f"{Fore.CYAN}🚦 Initial Intent: {classification}{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}❌ Error in initial classification: {e} → Defaulting to NEEDS_ANALYSIS{Style.RESET_ALL}")
            classification = "NEEDS_ANALYSIS"
            
        state["query_classification"] = classification
        return state


    def classify_followup_query(self, state):
        """Classify if the query is a follow-up to previous conversation"""
        print(f"\n{Fore.CYAN}🔄 ---CLASSIFY FOLLOW-UP QUERY---{Style.RESET_ALL}")
        
        question = state["question"]
        chat_context = state.get("chat_context", "")
        conversation_history = state.get("conversation_history", [])
        
        # Increment iteration and check limits
        iteration_count = self.increment_iteration(state)
        
        # Check if this is the first message in the chat session
        # Skip follow-up detection if no chat history AND no conversation history
        if not conversation_history and not chat_context.strip():
            print(f"{Fore.BLUE}📝 First message in chat session → Skipping follow-up detection{Style.RESET_ALL}")
            return {
                "question": question,
                "original_question": state.get("original_question", question),
                "chat_context": chat_context,
                "is_followup": False,
                "conversation_history": conversation_history,
                "fusion_method": state.get("fusion_method", "rrf"),
                "alpha": state.get("alpha", 0.5),
                "rrf_k": state.get("rrf_k", 60),
                "iteration_count": iteration_count,
                "start_time": state.get("start_time", time.time()),
                "max_iterations": state.get("max_iterations", 15),
                "max_time_seconds": state.get("max_time_seconds", 60.0),
                "hallucination_retry_count": state.get("hallucination_retry_count", 0)
            }
        
        # Additional check: if chat memory is empty, also skip
        if hasattr(self, 'chat_memory') and len(self.chat_memory.messages) <= 1:
            print(f"{Fore.BLUE}📝 First message in chat memory → Skipping follow-up detection{Style.RESET_ALL}")
            return {
                "question": question,
                "original_question": state.get("original_question", question),
                "chat_context": chat_context,
                "is_followup": False,
                "conversation_history": conversation_history,
                "fusion_method": state.get("fusion_method", "rrf"),
                "alpha": state.get("alpha", 0.5),
                "rrf_k": state.get("rrf_k", 60),
                "iteration_count": iteration_count,
                "start_time": state.get("start_time", time.time()),
                "max_iterations": state.get("max_iterations", 15),
                "max_time_seconds": state.get("max_time_seconds", 60.0),
                "hallucination_retry_count": state.get("hallucination_retry_count", 0)
            }
        
        # Check if the last query was classified as RETRIEVAL_NEEDED
        # Look at the last assistant message metadata to get the classification
        last_query_classification = None
        if self.chat_memory.messages:
            # Find the last assistant message and check its metadata
            for msg in reversed(self.chat_memory.messages):
                if msg.role == 'assistant' and msg.metadata:
                    last_query_classification = msg.metadata.get('query_classification')
                    break
        
        # Skip follow-up detection if last query wasn't RETRIEVAL_NEEDED
        if last_query_classification and last_query_classification != 'RETRIEVAL_NEEDED':
            print(f"{Fore.BLUE}📝 Previous query was '{last_query_classification}' (not RETRIEVAL_NEEDED) → Skipping follow-up detection{Style.RESET_ALL}")
            return {
                "question": question,
                "original_question": state.get("original_question", question),
                "chat_context": chat_context,
                "is_followup": False,
                "conversation_history": conversation_history,
                "fusion_method": state.get("fusion_method", "rrf"),
                "alpha": state.get("alpha", 0.5),
                "rrf_k": state.get("rrf_k", 60),
                "iteration_count": iteration_count,
                "start_time": state.get("start_time", time.time()),
                "max_iterations": state.get("max_iterations", 15),
                "max_time_seconds": state.get("max_time_seconds", 60.0),
                "hallucination_retry_count": state.get("hallucination_retry_count", 0)
            }
        
        # Follow-up classification prompt (only executed if we have conversation history)
         # Extract only the previous query from chat_context
        previous_query = ""
        if chat_context:
            lines = chat_context.split('\n')
            for line in lines:
                if line.startswith("Previous question:"):
                    previous_query = line.replace("Previous question:", "").strip()
                    break
        
        # If we can't extract from chat_context, try conversation_history
        if not previous_query and conversation_history:
            previous_query = conversation_history[-1]['question']
        
        if not previous_query:
            print(f"{Fore.YELLOW}⚠️ No previous query found for follow-up detection{Style.RESET_ALL}")
            return {
                "question": question,
                "original_question": state.get("original_question", question),
                "chat_context": chat_context,
                "is_followup": False,
                "conversation_history": conversation_history,
                "fusion_method": state.get("fusion_method", "rrf"),
                "alpha": state.get("alpha", 0.5),
                "rrf_k": state.get("rrf_k", 60),
                "iteration_count": iteration_count,
                "start_time": state.get("start_time", time.time()),
                "max_iterations": state.get("max_iterations", 15),
                "max_time_seconds": state.get("max_time_seconds", 60.0),
                "hallucination_retry_count": state.get("hallucination_retry_count", 0)
            }
        
        print("previous_query:", previous_query, "question:", question,"\n")
        followup_prompt = f"""You are a classifier. Your task is to determine if a user's "Current query" is a "FOLLOWUP" or a "NEW_QUERY" based on the "Previous query".

        **Definition of a FOLLOWUP Query:**
        A query is a FOLLOWUP if it directly asks for clarification, elaboration, or repetition of the *immediately preceding answer* from the assistant. These queries often don't make sense on their own.
        - Examples: "tell me more", "explain that again", "what does that mean?", "in other words?".

        **Definition of a NEW_QUERY:**
        A query is a NEW_QUERY if it asks for new information, even if it's about the same topic or entity as the previous question. It can be understood as a standalone question.
        - **Crucial Rule:** A related question is NOT a followup.
        - **Example:**
            - Previous Query: "Who can sign the LEAR?"
            - Current query: "Can electronic signatures be used for signing the LEAR?"
            - **Classification:** This is a NEW_QUERY. It introduces a new aspect (electronic signatures) rather than asking for clarification on who the LEAR is.

        **Analysis Task:**
        Analyze the following queries and classify the "Current query".

        **Previous query:** {previous_query}

        **Current query:** {question}

        **Your Response:**
        Respond with only one of two possible classifications: FOLLOWUP or NEW_QUERY.
        """        
        
        try:
            response = self.llm.invoke(followup_prompt)
            print(response.content)
            classification = response.content.strip().upper()
            
            is_followup = classification == "FOLLOWUP"
            
            if is_followup:
                print(f"{Fore.YELLOW}🔄 Follow-up query detected → Will enhance with previous context{Style.RESET_ALL}")
            else:
                print(f"{Fore.GREEN}📝 New independent query → Proceeding to normal classification{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ Error in follow-up classification: {e} → Treating as new query{Style.RESET_ALL}")
            is_followup = False
        
        return {
            "question": question,
            "original_question": state.get("original_question", question),
            "chat_context": chat_context,
            "is_followup": is_followup,
            "conversation_history": conversation_history,
            "fusion_method": state.get("fusion_method", "rrf"),
            "alpha": state.get("alpha", 0.5),
            "rrf_k": state.get("rrf_k", 60),
            "iteration_count": iteration_count,
            "start_time": state.get("start_time", time.time()),
            "max_iterations": state.get("max_iterations", 15),
            "max_time_seconds": state.get("max_time_seconds", 60.0),
            "hallucination_retry_count": state.get("hallucination_retry_count", 0)
        }


    def handle_followup_query(self, state):
        """
        Handle follow-up queries by reusing the last retrieved documents
        and preparing the state for the paragraph loading pipeline.
        """
        print(f"\n{Fore.YELLOW}🔄 ---HANDLE FOLLOW-UP QUERY (REUSING DOCS)---{Style.RESET_ALL}")

        iteration_count = self.increment_iteration(state)

        # Find the last retrieved documents from chat history
        last_retrieved_docs = None
        if hasattr(self, 'chat_memory'):
            for msg in reversed(self.chat_memory.messages):
                if msg.role == 'assistant' and msg.metadata and 'retrieved_documents' in msg.metadata:
                    last_retrieved_docs = msg.metadata['retrieved_documents']
                    break

        if not last_retrieved_docs or not last_retrieved_docs.get("matches"):
            print(f"{Fore.RED}❌ No previous documents found for follow-up. Generating fallback message.{Style.RESET_ALL}")
            return {
                **state,
                "generation": "I'm sorry, I don't have any previous search results to provide more details on. Could you please ask your original question again?",
                "iteration_count": iteration_count,
            }

        print(f"{Fore.GREEN}✅ Found {len(last_retrieved_docs['matches'])} documents from previous query. Reusing for follow-up.{Style.RESET_ALL}")

        # For follow-up queries, use the *previous* question for generation, not the follow-up phrase (e.g., "tell me more").
        conversation_history = state.get("conversation_history", [])
        question_for_generation = state["question"]  # Default to current question

        if conversation_history:
            # The last item in the history is the question we are following up on.
            previous_question = conversation_history[-1]['question']
            question_for_generation = previous_question
            print(f"{Fore.CYAN}🔄 Follow-up detected. Using previous question for generation: '{previous_question}'{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠️ Could not find previous question for follow-up. Using current query.{Style.RESET_ALL}")

        # Update state with the retrieved documents to pass to the 'load_paragraphs' node
        return {
            **state,
            "question": question_for_generation,
            "combined_documents": last_retrieved_docs,
            "iteration_count": iteration_count,
        }

    def decide_followup_or_classify(self, state):
        """Decide whether to handle as follow-up or proceed to normal classification"""
        
        # Check if we should exit early
        if self.should_exit_early(state):
            return "early_exit"
        
        is_followup = state.get("is_followup", False)
        
        if is_followup:
            print(f"\n{Fore.YELLOW}🔄 ---DECISION: HANDLE AS FOLLOW-UP---{Style.RESET_ALL}")
            return "handle_followup"
        else:
            print(f"\n{Fore.GREEN}📝 ---DECISION: PROCEED TO NORMAL CLASSIFICATION---{Style.RESET_ALL}")
            return "normal_classification"


    def decide_after_followup_handle(self, state):
        """Decide whether to proceed to paragraph loading or end."""
        if state.get("generation"):
            print(f"{Fore.YELLOW}↪️ Follow-up failed, ending with fallback message.{Style.RESET_ALL}")
            return "end_followup"
        else:
            print(f"{Fore.GREEN}✅ Follow-up successful, proceeding to load paragraphs.{Style.RESET_ALL}")
            return "load_paragraphs"


    def decide_initial_intent(self, state):
        """Decide the next step based on the initial query classification."""
        if self.should_exit_early(state):
            return "early_exit"
        
        classification = state.get("query_classification", "NEEDS_ANALYSIS")
        
        if classification == "GOODBYE":
            print(f"\n{Fore.YELLOW}👋 ---DECISION: GOODBYE DETECTED---{Style.RESET_ALL}")
            return "goodbye"
        elif classification == "DIRECT_ANSWER":
            print(f"\n{Fore.BLUE}💬 ---DECISION: PROVIDE DIRECT ANSWER---{Style.RESET_ALL}")
            return "direct_answer"
        else:  # NEEDS_ANALYSIS
            print(f"\n{Fore.GREEN}🤔 ---DECISION: QUERY NEEDS FURTHER ANALYSIS---{Style.RESET_ALL}")
            return "needs_analysis"


    def generate_direct_answer(self, state):
        """Generate direct answer for general queries that don't need retrieval"""
        print(f"\n{Fore.BLUE}💬 ---GENERATE DIRECT ANSWER---{Style.RESET_ALL}")
        
        question = state["question"]
        chat_context = state.get("chat_context", "")
        
        # Increment iteration and check limits
        iteration_count = self.increment_iteration(state)
        
        # Direct answer prompt
        direct_answer_prompt = f"""
    You are a helpful assistant specialized in answering questions about grant funding documents. 
    The user has asked a general question that doesn't require searching through documents.

    Your capabilities:
    - You can answer questions about grant funding, deadlines, requirements, eligibility criteria, and application processes
    - You have access to specialized documents about grant funding
    - You can help users understand funding opportunities and application procedures
    - You focus specifically on grant-related topics

    Chat context (if any):
    {chat_context}

    User's question: {question}

    Provide a helpful, friendly response. If they're asking about your capabilities, explain that you're designed to help with grant funding questions by searching through specialized documents. Keep your response concise and relevant.
    """
        
        try:
            response = self.llm.invoke(direct_answer_prompt)
            generation = response.content.strip()
            print(f"{Fore.BLUE}💬 Generated direct answer (preview): {generation[:100]}...{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Error generating direct answer: {e}{Style.RESET_ALL}")
            generation = "Hello! I'm here to help you with questions about grant funding. I can search through specialized documents to find information about deadlines, requirements, eligibility criteria, and application processes. How can I assist you today?"
        
        return {
            "question": question,
            "original_question": state.get("original_question", question),
            "chat_context": chat_context,
            "generation": generation,
            "query_classification": state.get("query_classification", "DIRECT_ANSWER"),
            "conversation_history": state.get("conversation_history", []),
            "dense_documents": {"matches": []},
            "sparse_documents": {"matches": []},
            "combined_documents": {"matches": []},
            "fusion_method": state.get("fusion_method", "rrf"),
            "alpha": state.get("alpha", 0.5),
            "rrf_k": state.get("rrf_k", 60),
            "iteration_count": iteration_count,
            "start_time": state.get("start_time", time.time()),
            "max_iterations": state.get("max_iterations", 15),
            "max_time_seconds": state.get("max_time_seconds", 60.0),
        }

    def generate_goodbye_message(self, state):
        """Generate a goodbye message and prepare for session end"""
        print(f"\n{Fore.YELLOW}👋 ---GENERATE GOODBYE MESSAGE---{Style.RESET_ALL}")
        
        question = state["question"]
        chat_context = state.get("chat_context", "")
        
        # Increment iteration and check limits
        iteration_count = self.increment_iteration(state)
        
        # Goodbye message prompt
        goodbye_prompt = f"""
    The user is saying goodbye. Provide a brief, friendly farewell message.
    Keep it professional but warm, and thank them for using the grant funding assistance service.

    User's goodbye: {question}

    Generate a short, appropriate farewell response.
    """
        
        try:
            response = self.llm.invoke(goodbye_prompt)
            generation = response.content.strip()
            print(f"{Fore.YELLOW}👋 Generated goodbye message: {generation}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}❌ Error generating goodbye message: {e}{Style.RESET_ALL}")
            generation = "Goodbye! Thank you for using the grant funding assistance. Feel free to return anytime if you have more questions about grants or funding opportunities. Have a great day!"
        
        # Mark session for termination
        self.session_active = False
        
        return {
            "question": question,
            "original_question": state.get("original_question", question),
            "chat_context": chat_context,
            "generation": generation,
            "query_classification": state.get("query_classification", "GOODBYE"),
            "conversation_history": state.get("conversation_history", []),
            "dense_documents": {"matches": []},
            "sparse_documents": {"matches": []},
            "combined_documents": {"matches": []},
            "fusion_method": state.get("fusion_method", "rrf"),
            "alpha": state.get("alpha", 0.5),
            "rrf_k": state.get("rrf_k", 60),
            "iteration_count": iteration_count,
            "start_time": state.get("start_time", time.time()),
            "max_iterations": state.get("max_iterations", 15),
            "max_time_seconds": state.get("max_time_seconds", 60.0),
            "session_end": True
        }

    def build_graph(self):
        """Build the LangGraph workflow with follow-up classification as the first step"""
        workflow = StateGraph(GraphState)

        # Define nodes
        workflow.add_node("classify_initial_intent", self.classify_initial_intent)
        workflow.add_node("classify_followup_query", self.classify_followup_query)
        workflow.add_node("handle_followup_query", self.handle_followup_query)
        workflow.add_node("generate_direct_answer", self.generate_direct_answer)
        workflow.add_node("generate_goodbye_message", self.generate_goodbye_message)
        workflow.add_node("check_history_sufficiency", self.check_history_sufficiency)
        workflow.add_node("generate_from_history", self.generate_from_history)
        workflow.add_node("retrieve_dense", self.retrieve_dense)
        workflow.add_node("retrieve_sparse", self.retrieve_sparse)
        workflow.add_node("late_fusion", self.late_fusion)
        workflow.add_node("rerank", self.rerank)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("regenerate_after_hallucination", self.regenerate_after_hallucination)
        workflow.add_node("load_paragraphs", self.load_paragraphs)
        workflow.add_node("generate_with_paragraphs", self.generate_with_paragraphs)
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("grade_hallucination", self.grade_hallucination)
        workflow.add_node("grade_answer_initial", self.grade_answer_initial)
        workflow.add_node("grade_answer_paragraphs", self.grade_answer_paragraphs)
        workflow.add_node("early_exit", self.early_exit)

        # Start with initial intent classification
        workflow.add_edge(START, "classify_initial_intent")

        # Decision point: initial intent
        workflow.add_conditional_edges(
            "classify_initial_intent",
            self.decide_initial_intent,
            {
                "goodbye": "generate_goodbye_message",
                "direct_answer": "generate_direct_answer",
                "needs_analysis": "classify_followup_query",
                "early_exit": "early_exit",
            }
        )
        
        # Decision point: follow-up or normal classification
        workflow.add_conditional_edges(
            "classify_followup_query",
            self.decide_followup_or_classify,
            {
                "handle_followup": "handle_followup_query",
                "normal_classification": "check_history_sufficiency",
                "early_exit": "early_exit",
            }
        )
        
        # New decision point: history sufficient or not
        workflow.add_conditional_edges(
            "check_history_sufficiency",
            self.decide_history_or_classification,
            {
                "generate_from_history": "generate_from_history",
                "need_retrieval": "retrieve_dense",
                "early_exit": "early_exit",
            }
        )
        workflow.add_edge("generate_from_history", END)

        # Follow-up path now goes to load_paragraphs or ends
        workflow.add_conditional_edges(
            "handle_followup_query",
            self.decide_after_followup_handle,
            {
                "end_followup": END,
                "load_paragraphs": "load_paragraphs",
            }
        )
        
        # Direct paths to END for non-retrieval queries
        workflow.add_edge("generate_direct_answer", END)
        workflow.add_edge("generate_goodbye_message", END)
        
        # Simplified retrieval workflow
        workflow.add_edge("retrieve_dense", "retrieve_sparse")
        workflow.add_edge("retrieve_sparse", "late_fusion")

        if self.config.get('use_reranker', True):
            workflow.add_edge("late_fusion", "rerank")
            workflow.add_edge("rerank", "grade_documents")
        else:
            workflow.add_edge("late_fusion", "grade_documents")
        
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "docs_not_relevant": "transform_query",
                "docs_relevant": "generate",
                "early_exit": "early_exit",
            }
        )
        
        # First generation attempt
        workflow.add_edge("generate", "grade_hallucination")
        
        # Add regeneration node to workflow
        workflow.add_edge("regenerate_after_hallucination", "grade_hallucination")
        
        workflow.add_conditional_edges(
            "grade_hallucination",
            self.decide_after_hallucination_check,
            {
                "regenerate": "regenerate_after_hallucination",
                "load_paragraphs": "load_paragraphs",
                "grounded": "grade_answer_initial",
                "early_exit": "early_exit",
            }
        )
        
        # Decision point after initial answer check
        workflow.add_conditional_edges(
            "grade_answer_initial",
            self.decide_after_initial_answer_check,
            {
                "chunks_sufficient": END,
                "need_paragraphs": "load_paragraphs",
                "early_exit": "early_exit",
            }
        )
        
        # Fallback: load paragraphs and generate again
        workflow.add_edge("load_paragraphs", "generate_with_paragraphs")
        workflow.add_edge("generate_with_paragraphs", "grade_answer_paragraphs")
        
        # Final decision after paragraph attempt
        workflow.add_conditional_edges(
            "grade_answer_paragraphs",
            self.decide_after_paragraph_answer_check,
            {
                "paragraphs_sufficient": END,
                "need_query_transform": "transform_query",
                "early_exit": "early_exit",
            }
        )
        
        # Transform query goes back to dense retrieval to start over
        workflow.add_edge("transform_query", "retrieve_dense")
        
        # Early exit leads to END
        workflow.add_edge("early_exit", END)

        self.app = workflow.compile()

    
    def deinitialize(self):
        """
        De-initializes the pipeline by releasing models and deleting indices.
        This is a destructive operation designed to clear resources.
        """
        print(f"\n{Fore.RED}🔴 ---DE-INITIALIZING PIPELINE---{Style.RESET_ALL}")
        
        try:
            # Clear models from memory
            print("Releasing models from memory...")
            models_to_release = ['llm', 'embed_model', 'reranker', 'sparse_encoder']
            for model_attr in models_to_release:
                if hasattr(self, model_attr):
                    setattr(self, model_attr, None)
            
            # Clear other state
            attributes_to_clear = ['dense_index', 'sparse_index', 'app', 'pinecone_handler']
            for attr in attributes_to_clear:
                 if hasattr(self, attr):
                    setattr(self, attr, None)

            if hasattr(self, 'chat_memory') and self.chat_memory:
                self.chat_memory.clear_history()
            
            # Release CUDA memory
            if 'torch' in sys.modules and torch.cuda.is_available():
                print("Clearing CUDA cache...")
                gc.collect()
                torch.cuda.empty_cache()
                
            print(f"{Fore.GREEN}✅ Pipeline de-initialized successfully.{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}❌ Error during de-initialization: {e}{Style.RESET_ALL}")


    def save_graph_visualization(self, output_path: str = "late_fusion_rag_graph.png"):
        """Save graph visualization"""
        try:
            graph_image = self.app.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(graph_image)
            print(f"{Fore.GREEN}💾 Late Fusion RAG Graph saved as '{output_path}'{Style.RESET_ALL}")
            return True
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️ Could not save graph visualization: {e}{Style.RESET_ALL}")
            return False

   
    # chat modules

    def run_chat_query(self, query: str, fusion_method: str = "rrf", 
                alpha: float = 0.5, rrf_k: int = 60, 
                max_iterations: int = 15, max_time_seconds: float = 60.0):
        """Run a single query in the chat session"""
        
        # Get chat context BEFORE adding the new query
        chat_context = self.chat_memory.get_context(include_last_n=5)
        
        # Add user message to memory
        self.chat_memory.add_message("user", query)
        
        # Prepare conversation history for the graph (only for follow-up handling)
        conversation_history = []
        for msg in self.chat_memory.messages[-10:]:
            if msg.role == 'user':
                next_idx = self.chat_memory.messages.index(msg) + 1
                if next_idx < len(self.chat_memory.messages) and self.chat_memory.messages[next_idx].role == 'assistant':
                    conversation_history.append({
                        'question': msg.content,
                        'answer': self.chat_memory.messages[next_idx].content
                    })
        
        try:
            # Initialize state with query classification support (REMOVED use_history field)
            initial_state = {
                "question": query,
                "original_question": query,
                "chat_context": chat_context,
                "generation": "",
                "query_classification": "",
                "is_followup": False,
                "dense_documents": {},
                "sparse_documents": {},
                "combined_documents": {},
                "hallucination_score": "",
                "answer_score": "",
                "fusion_method": fusion_method,
                "alpha": alpha,
                "rrf_k": rrf_k,
                "hallucination_retry_count": 0,
                "iteration_count": 0,
                "start_time": 0.0,
                "max_iterations": max_iterations,
                "max_time_seconds": max_time_seconds,
                "conversation_history": conversation_history,
                "session_end": False
            }

            # Run the graph
            result = self.app.invoke(initial_state)

            print(f"{Fore.WHITE}\n\n{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.BLUE}{Style.BRIGHT}❓ Query:\n{Style.RESET_ALL}{query}\n")
            print(f"{Fore.GREEN}{Style.BRIGHT}💡 Answer:\n{Style.RESET_ALL}{result['generation']}")
            print(f"{Fore.WHITE}{'='*60}{Style.RESET_ALL}\n")
            
            # Add assistant response to memory
            assistant_metadata = {
                'fusion_method': fusion_method,
                'iterations': result.get('iteration_count', 0),
                'elapsed_time': time.time() - result.get("start_time", time.time()),
                'used_history': result.get('use_history', False),
                'query_classification': result.get('query_classification', 'UNKNOWN')
            }
            # If retrieval was used, store the documents for potential follow-up
            if result.get('combined_documents') and result.get('combined_documents').get('matches'):
                assistant_metadata['retrieved_documents'] = result.get('combined_documents')
                # If documents were retrieved, the classification must be RETRIEVAL_NEEDED.
                # This corrects the state if it wasn't propagated correctly through the graph.
                assistant_metadata['query_classification'] = 'RETRIEVAL_NEEDED'

            self.chat_memory.add_message("assistant", result['generation'], assistant_metadata)
            
            # Check if session should end (for goodbye messages)
            if result.get('session_end', False) or result.get('query_classification') == 'GOODBYE':
                self.session_active = False
            
            # Print execution stats
            elapsed_time = time.time() - result.get("start_time", time.time())
            print(f"{Fore.BLUE}📊 Execution Stats:{Style.RESET_ALL}")
            print(f"  - Query type: {result.get('query_classification', 'UNKNOWN')}")
            print(f"  - Used history: {result.get('use_history', False)}")
            print(f"  - Iterations: {result.get('iteration_count', 0)}")
            print(f"  - Time elapsed: {elapsed_time:.2f} seconds")
            print(f"  - Chat history: {len(self.chat_memory.messages)} messages")
            
            return result
        except Exception as e:
            error_msg = f"Error processing your question: {e}"
            print(f"{Fore.RED}❌ {error_msg}{Style.RESET_ALL}")
            self.chat_memory.add_message("assistant", error_msg)
            return None


    def start_chat_session(self):
        """Start an interactive chat session"""
        print(f"\n{Fore.GREEN}{Style.BRIGHT}🤖 Grant Funding Assistant - Chat Session Started!{Style.RESET_ALL}")        
        print(f"{Fore.YELLOW}Special Commands:{Style.RESET_ALL}")
        
        print(f"{Fore.YELLOW}  - {Fore.CYAN}📚 '/history'{Fore.YELLOW} - Show conversation history{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  - {Fore.CYAN}🧹 '/clear'{Fore.YELLOW} - Clear conversation history{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  - {Fore.CYAN}⚙️ '/config'{Fore.YELLOW} - Show current configuration{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  - {Fore.CYAN}📊 '/stats'{Fore.YELLOW} - Show memory statistics{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
        
        while self.session_active:
            try:
                # Get user input
                user_input = input(f"{Fore.BLUE}👤 You: {Style.RESET_ALL}").strip()
                
                if not user_input:
                    continue
                
                # Handle commands (but don't interfere with natural goodbye detection)
                if user_input.startswith('/'):
                    self.handle_command(user_input)
                    continue
                
                # Process the query (classification will handle goodbyes naturally)
                result = self.run_chat_query(
                    user_input,
                    fusion_method=self.config['fusion_method'],
                    alpha=self.config['alpha'],
                    rrf_k=self.config['rrf_k'],
                    max_iterations=self.config.get('max_iterations', 15),
                    max_time_seconds=self.config.get('max_time_seconds', 60.0)
                )
                
                # Check if the session should end based on the result
                if not self.session_active:
                    print(f"\n{Fore.GREEN}👋 Chat session ended. De-initializing pipeline...{Style.RESET_ALL}")
                    self.deinitialize()
                    break
                
                print()  # Add spacing between interactions
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Chat session interrupted.{Style.RESET_ALL}")
                break
            except EOFError:
                print(f"\n{Fore.YELLOW}Chat session ended.{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}❌ Unexpected error: {e}{Style.RESET_ALL}")
                continue
        
        print(f"{Fore.GREEN}👋 Thank you for using the Grant Funding Assistant. Goodbye!{Style.RESET_ALL}")


    def handle_command(self, command: str):
        """Handle chat commands"""
        command = command.lower().strip()
            
        if command == '/history':
            print(f"\n{Fore.CYAN}📚 Conversation History:{Style.RESET_ALL}")
            summary = self.chat_memory.get_conversation_summary()
            print(summary if summary != "No conversation history." else "No conversation history yet.")
            print()
            
        elif command == '/clear':
            self.chat_memory.clear_history()
            print(f"{Fore.GREEN}✅ Conversation history cleared.{Style.RESET_ALL}\n")
            
        elif command == '/config':
            print(f"\n{Fore.CYAN}⚙️ Current Configuration:{Style.RESET_ALL}")
            print(f"  - Fusion method: {self.config['fusion_method']}")
            print(f"  - Alpha: {self.config['alpha']}")
            print(f"  - RRF k: {self.config['rrf_k']}")
            print(f"  - Max iterations: {self.config.get('max_iterations', 15)}")
            print(f"  - Max time: {self.config.get('max_time_seconds', 60.0)}s")
            print(f"  - Max chat history: {self.config.get('max_chat_history', 20)}")
            print()
            
        elif command == '/stats':
            print(f"\n{Fore.CYAN}📊 Memory Statistics:{Style.RESET_ALL}")
            print(f"  - Total messages: {len(self.chat_memory.messages)}")
            print(f"  - Max history: {self.chat_memory.max_messages}")
            if self.chat_memory.messages:
                print(f"  - Session duration: {(time.time() - self.chat_memory.messages[0].timestamp)/60:.1f} minutes")
            print()
            
        else:
            print(f"{Fore.RED}❌ Unknown command: {command}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Available commands: /exit, /quit, /history, /clear, /config, /stats{Style.RESET_ALL}\n")


def main():
    # Configuration for multi-document setup with memory-safe settings
    config = {
        'source_path': "../preprocessing/cropsAll/",  # Root folder containing multiple documents
        'chunk_dst': "chunks_grants/multi_doc/",  # Output folder for all embeddings
        'dense_index_name': 'grant-dense-multi-doc',
        'sparse_index_name': 'grant-sparse-multi-doc',
        'processing_batch_size': 1, 
        'upsert_batch_size': 10,    
        'sparse_top_k': 1000,       
        'dense_search_top_k': 5,
        'sparse_search_top_k': 5,
        'rerank_top_k': 3,
        'use_reranker': False,  # Set to False to disable the reranking step
        'fusion_method': 'rrf',
        'alpha': 0.7,
        'rrf_k': 60,
        'create_new_embeddings': False,  # Set to True to create new embeddings
        'run_initial_setup': False, # Set to True for first-time setup or to rebuild indices
        # Chat-specific config
        'max_chat_history': 20,
        'max_iterations': 15,
        'max_time_seconds': 60.0,
    }
    
    # Automatically set dependent flags based on create_new_embeddings
    if config['create_new_embeddings']:
        config['load_existing_chunks'] = False
        config['reset_db'] = True
    else:
        config['load_existing_chunks'] = True
        config['reset_db'] = False
    
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Initialize chat pipeline
    pipeline = LateFusionHybridRAGChatPipeline(config)
    pipeline.load_models()

    # If run_initial_setup is True, perform the one-time setup.
    # Otherwise, just connect to the existing indices.
    if config.get('run_initial_setup', False):
        logger.info("`run_initial_setup` is True. Running full setup: loading/creating embeddings and populating indices...")
        
        # Create or load embeddings from files
        if config['create_new_embeddings']:
            utils.checkPath(config['chunk_dst'])
            dense_embeddings, sparse_embeddings = pipeline.create_embeddings(
                config['source_path'], 
                config['chunk_dst']
            )
        else:
            dense_embeddings, sparse_embeddings = pipeline.load_embeddings(config['chunk_dst'])
        
        if not dense_embeddings or not sparse_embeddings:
            logger.error("No embeddings available for setup. Exiting.")
            exit()
        
        # Initialize and populate indices
        pipeline.initialize_indices(
            config['dense_index_name'], 
            config['sparse_index_name'],
            dense_dimension=1024,
            sparse_dimension=config['sparse_top_k'], 
            reset_db=config['reset_db'],
        )
        pipeline.populate_indices(dense_embeddings, sparse_embeddings)
        logger.info("Initial setup complete.")

    else:
        logger.info("`run_initial_setup` is False. Skipping setup, connecting to existing indices...")
        # Just initialize the connection to the indices without resetting or populating
        pipeline.initialize_indices(
            config['dense_index_name'], 
            config['sparse_index_name'],
            dense_dimension=1024,
            sparse_dimension=config['sparse_top_k'], 
            reset_db=False,
        )
    
    # Build graph
    pipeline.build_graph()
    pipeline.save_graph_visualization("images/rag_system_graph.png")
    
    # Start interactive chat session
    pipeline.start_chat_session()

if __name__ == "__main__":
    main()

