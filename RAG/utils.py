from transformers import AutoTokenizer, AutoModelForCausalLM
from splade.splade.models.transformer_rep import Splade
from transformers import AutoModel, AutoTokenizer
import torch
#from sklearn.metrics.pairwise import cosine_similarity
import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()


def suppress_warnings():
    import warnings
    warnings.filterwarnings("ignore")

def pinecone_login():    
    return Pinecone(os.getenv("PINECONE_API_KEY"))


def loadFile_recursive(ext,path=os.getcwd()):
    

    cfiles = []
    for root, dirs, files in os.walk(path):
      for file in files:
        #print(file)
        for i in ext:
            if file.endswith(i):
                cfiles.append(os.path.join(root, file))
    
    return cfiles

def checkPath(path):
    #path=os.getcwd()+'\\'+'background images'
    if not os.path.exists(path):
        os.makedirs(path)
    return 


def HF_login():
    from huggingface_hub import login
    login(os.getenv("HF_API_KEY"))

def spalde_v3_load(device='cuda'):
    
    HF_login()
    model_name = "naver/splade-v3"
    
    model = Splade(model_name, agg="max").to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #reverse_voc = {v: k for k, v in tokenizer.vocab.items()}


    # Load the SPLADE model (you can change this to the specific SPLADE variant you need)
    # model_name = "naver/splade-v3"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForMaskedLM.from_pretrained(model_name)

    # Input sentence without any [MASK] token
    #text = "This is an example sentence for SPLADE embeddings."
    #text = "I like to go to the gym"

    # # Tokenize the input (returning the input tensor for PyTorch)
    # inputs = tokenizer(text, return_tensors="pt")

    # now compute the document representation
    return model, tokenizer


def load_Alibaba(device='cuda'):
    #model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
    model = AutoModel.from_pretrained('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True,torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True,torch_dtype=torch.float16)
    return model, tokenizer

def load_Qwen(device='cuda'):
    #model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
    model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True,torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True,torch_dtype=torch.float16)
    return model, tokenizer


def load_QwenReranker(device='cuda'):
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").to(device).eval()
    return model, tokenizer


def reranker(model, tokenizer, query, documents, top_k=2):

    def format_instruction(query, doc):
        output = "<Query>: {query}\n<Document>: {doc}".format(query=query, doc=doc)
        return output

    def process_inputs(pairs):
        inputs = tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt")
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(inputs, **kwargs):
        batch_scores = model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    max_length = 8192

    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)


    # Extract the actual document texts
    documents_list = [match['metadata']['context'] for match in documents['matches']]

    all_scores = []
    pairs = [format_instruction(query, doc_text) for doc_text in documents_list]
    inputs = process_inputs(pairs)
    scores = compute_logits(inputs)
    all_scores.append(scores)

    # Example of how to associate scores with documents for a single query

    query_scores = all_scores[0]
    # Update the original document dictionaries with the new reranker scores
    for i, match in enumerate(documents['matches']):
        match['reranker_score'] = query_scores[i] # Add new key for reranker score

    # Sort the documents based on the new reranker_score
    # We sort a copy of the list of matches to keep the original documents_data intact if needed
    # or you can sort documents_data['matches'] in place
    ranked_matches = sorted(documents['matches'], key=lambda x: x['reranker_score'], reverse=True)

    print(f"\nRanked documents for query '{query}':")
    for match in ranked_matches:
        print(f"ID: {match['id']} - Score: {match['reranker_score']:.4f}")

    # Get the top 2 reranked documents
    top_k_reranked_matches = ranked_matches[:top_k]

    # Construct the output dictionary in the desired format
    output_documents_data = {
        'matches': top_k_reranked_matches,
        'namespace': documents.get('namespace', ''), # Preserve namespace if it exists
        'usage': documents.get('usage', {}) # Preserve usage if it exists
    }

    return output_documents_data


def runDense(model, tokenizer, sentence, device='cuda'):

    # Tokenize and get embeddings
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to(device)
    embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Averaging over token embeddings to get sentence embeddings
    #print(embeddings[0].shape)
    

    # Compute cosine similarity between the embeddings
    # cosine_similarity = util.cos_sim(embeddings[0], embeddings[1])
    # print(cosine_similarity)
    
    return embeddings.cpu().tolist()


#################################### Pinecone ####################################


def initialize_pinecone(index_name, dimension=384, metric='cosine', cloud='aws', region='us-east-1'):
    
    """Initialize Pinecone and connect to index."""
    pc = pinecone_login()

    # existing_indexes = [
    #     index_info["name"] for index_info in pc.list_indexes()
    # ]

    # Check if the index exists, if not, create it
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, 
            dimension=dimension,  # embedding model size
            metric=metric,
            spec=ServerlessSpec(
                cloud=cloud,
                region=region
            )
        )
    
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    # Connect to the index
    return pc, pc.Index(index_name)
