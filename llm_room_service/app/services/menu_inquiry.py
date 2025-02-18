import numpy as np
from openai import OpenAI
from loguru import logger

from ..config import MENU_ITEMS, OPENAI_CONFIG
from .menu_embeddings import menu_embedding_service

class MenuInquirySystem:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_CONFIG["api_key"])

    async def answer_inquiry(self, query: str) -> str:
        """Answer a menu-related inquiry using relevant context."""
        logger.info(f"Processing menu inquiry: {query}")
        
        # Find relevant menu items using the shared embedding service
        relevant_items = menu_embedding_service.find_similar_items(query, threshold=0.7)
        
        if not relevant_items:
            return "I apologize, but I couldn't find specific information about that in our menu. Could you please rephrase your question?"
            
        # Create context from relevant items
        context = "Menu Information:\n"
        for item_name, _, data in relevant_items[:3]:
            details = data["details"]
            context += f"""
            Item: {item_name}
            Category: {data['category']}
            Price: ${details['price']}
            Description: {details['description']}
            Allergens: {', '.join(details['allergens'])}
            Modifications: {', '.join(details['available_modifications'])}
            Preparation Time: {details['preparation_time']} minutes
            """
        
        # Generate response using GPT
        try:
            messages = [
                {"role": "system", "content": f"""You are a helpful room service assistant. 
Answer questions about menu items using ONLY the information provided in the context below.
If you're not sure or the information isn't in the context, say so.
Be concise but friendly.

{context}"""},
                {"role": "user", "content": query}
            ]
            
            response = self.client.chat.completions.create(
                model=OPENAI_CONFIG["model"],
                messages=messages,
                temperature=0.3  # Keep it factual
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response. Please try again."

# Initialize menu inquiry system at module level
menu_inquiry_system = MenuInquirySystem() 
