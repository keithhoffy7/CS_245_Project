"""
Gemini LLM Wrapper for WebSocietySimulator
"""

import os
from typing import Dict, List, Optional, Union
import google.generativeai as genai
from websocietysimulator.llm import LLMBase
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
logger = logging.getLogger("websocietysimulator")

class GeminiLLM(LLMBase):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        """
        Initialize Gemini LLM
        
        Args:
            api_key: Google Gemini API key
            model: Model name, defaults to gemini-2.0-flash
        """
        super().__init__(model)
        genai.configure(api_key=api_key)
        self.api_key = api_key
        self.model_name = model
        self.client = genai.GenerativeModel(model)
    
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=10, max=300),  # Wait time starts at 10 seconds, exponential backoff, max 300 seconds
        stop=stop_after_attempt(10)  # Retry up to 10 times
    )
    def __call__(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 500, stop_strs: Optional[List[str]] = None, n: int = 1) -> Union[str, List[str]]:
        """
        Call Gemini API to get response with rate limit handling
        
        Args:
            messages: List of input messages, each message is a dict containing role and content
            model: Optional model override
            temperature: Temperature for generation, defaults to 0.0
            max_tokens: Maximum tokens in response, defaults to 500
            stop_strs: Optional list of stop strings
            n: Number of responses to generate, defaults to 1
            
        Returns:
            Union[str, List[str]]: Response text from LLM, either a single string or list of strings
        """
        # Use model override if provided, otherwise use default
        model_to_use = model or self.model
        client = genai.GenerativeModel(model_to_use)
        
        # Convert OpenAI-style messages to Gemini format
        # Gemini expects history + current message
        
        # Simple concatenation for now as Gemini's chat history format is different
        # This is a robust way to handle the prompt
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant: "
        
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                stop_sequences=stop_strs,
                candidate_count=1  # Gemini currently supports 1 candidate per request via this API
            )
            
            responses = []
            for _ in range(n):
                response = client.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                responses.append(response.text)
            
            if n == 1:
                return responses[0]
            else:
                return responses
                
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower() or "quota" in str(e).lower():
                logger.warning(f"Rate limit exceeded, will retry: {e}")
            else:
                logger.error(f"Gemini API Error: {e}")
            raise e
    
    def get_embedding_model(self):
        # Return None - embeddings are handled directly with genai.embed_content
        return None