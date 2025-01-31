import requests
from typing import List, Dict
import re
from backend.ingestion import retrieve_similar
import time
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_API_BASE = "http://localhost:11434"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_RETRIES = 3  # Number of retries for API calls
RETRY_DELAY = 2  # Delay between retries in seconds
RESPONSE_MODEL = "gpt-4o-mini"  # OpenAI's GPT-4o-mini model

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def list_models() -> List[str]:
    """Get list of available models."""
    try:
        # Get Ollama models for reasoning
        response = requests.get(f"{OLLAMA_API_BASE}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            deepseek_models = [m for m in model_names if "deepseek" in m.lower()]
            return {
                "deepseek": deepseek_models,
                "gpt": [RESPONSE_MODEL],  # Show GPT-4o-mini as the response model
                "all": model_names + [RESPONSE_MODEL]
            }
        return {"deepseek": [], "gpt": [RESPONSE_MODEL], "all": []}
    except Exception as e:
        print(f"Error listing models: {e}")
        return {"deepseek": [], "gpt": [RESPONSE_MODEL], "all": []}

def make_ollama_call(url: str, payload: Dict, model_name: str) -> Dict:
    """Make Ollama API call with retries."""
    for attempt in range(MAX_RETRIES):
        try:
            print(f"Attempt {attempt + 1} of {MAX_RETRIES} for {model_name}...")
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "response": response.json().get("response", "")
                }
            
            print(f"Error from {model_name} API: {response.status_code}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
                
            return {
                "success": False,
                "error": f"Error: {response.status_code}"
            }
            
        except Exception as e:
            print(f"Error on attempt {attempt + 1} for {model_name}: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
                continue
            return {
                "success": False,
                "error": str(e)
            }

def extract_reasoning(text: str) -> str:
    """Extract only the reasoning content from deepseek response."""
    text = text.strip()
    text = re.sub(r"^(?:Reasoning Process:|Reasoning:)\s*", "", text, flags=re.IGNORECASE)
    text = re.split(r"\n(?:Answer:|Response:|Final Answer:)", text, flags=re.IGNORECASE)[0]
    return text.strip()

def get_deepseek_reasoning(query: str, context: str, model_name: str = "deepseek", system_prompt: str = "") -> Dict:
    """Get reasoning from DeepSeek model with RAG context."""
    deepseek_prompt = f"""Context information:
{context}

I want you to ONLY show your reasoning content about how to carry out the task using the given context.
Focus on analyzing the task and breaking down how you would approach it using the context provided.
DO NOT provide any final answer or conclusion.

Task: {query}"""
    
    payload = {
        "model": model_name,
        "prompt": deepseek_prompt,
        "system": system_prompt,
        "stream": False,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 2048
    }
    
    result = make_ollama_call(f"{OLLAMA_API_BASE}/api/generate", payload, model_name)
    if not result["success"]:
        return result
        
    if not result["response"]:
        return {
            "success": False,
            "error": f"Empty response from {model_name}"
        }
        
    reasoning_content = extract_reasoning(result["response"])
    return {
        "success": True,
        "reasoning": reasoning_content
    }

def get_gpt_response(model_name: str, original_prompt: str, reasoning: str) -> Dict:
    """Get the response from GPT-4o-mini model based on the reasoning."""
    try:
        chain_prompt = f"""Original task: {original_prompt}

Using ONLY the following reasoning process produced by a different model, provide your answer to the original task.
Base your answer solely on these logical steps and thought process.

Make sure to create a detailed but concise answer to the task.

Reasoning steps:
{reasoning}

Provide your direct answer to given task:"""

        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model=RESPONSE_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise and accurate answers based on given reasoning steps."},
                {"role": "user", "content": chain_prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        
        if not response.choices:
            return {
                "success": False,
                "error": "Empty response from OpenAI API"
            }
            
        return {
            "success": True,
            "response": response.choices[0].message.content
        }
        
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

def process_query(query: str, reasoning_model: str = "deepseek", response_model: str = None, k: int = 4) -> Dict:
    """Process a query through the complete RAG + DeepSeek + GPT chain."""
    try:
        print(f"Processing query: {query}")
        print(f"Using models - Reasoning: {reasoning_model}, Response: {RESPONSE_MODEL}")
        
        # Step 1: Get relevant context using RAG
        print("Getting relevant context...")
        docs = retrieve_similar(query, k)
        context = "\n\n".join(doc.page_content for doc in docs)
        print("Retrieved context")
        
        # Step 2: Get reasoning from DeepSeek using context
        print(f"Getting reasoning from {reasoning_model}...")
        reasoning_result = get_deepseek_reasoning(query, context, model_name=reasoning_model)
        if not reasoning_result["success"]:
            print(f"{reasoning_model} reasoning failed: {reasoning_result['error']}")
            return reasoning_result
        print(f"Got reasoning from {reasoning_model}")
        
        # Step 3: Get final response from GPT-4o-mini using reasoning
        print(f"Getting response from {RESPONSE_MODEL}...")
        final_result = get_gpt_response(RESPONSE_MODEL, query, reasoning_result["reasoning"])
        if not final_result["success"]:
            print(f"{RESPONSE_MODEL} response failed: {final_result['error']}")
            return final_result
        print(f"Got response from {RESPONSE_MODEL}")
        
        return {
            "success": True,
            "context": context,
            "reasoning": reasoning_result["reasoning"],
            "response": final_result["response"]
        }
    except Exception as e:
        print(f"Error in process_query: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        } 