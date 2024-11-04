from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
import torch

class ModelLoader:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self, 
                  model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                  max_tokens: int = 512,
                  temperature: float = 0.1, 
                  top_p: float = 0.95,
                  repetition_penalty: float = 1.15):

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
                device_map="auto"
            )
                        
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                # return_full_text=False, 
            )
            
            return HuggingFacePipeline(pipeline=pipe)
            
        except Exception as e:
            raise Exception(f"Error loading model {model_name}: {str(e)}")
    
    def load_embedding_model(self, 
                           model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': self.device},
                encode_kwargs={'device': self.device, 'batch_size': 32}
            )
        except Exception as e:
            raise Exception(f"Error loading embedding model {model_name}: {str(e)}")