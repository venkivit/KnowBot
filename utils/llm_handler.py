from groq import Groq
from typing import List

class LLMHandler:
    def __init__(self):
        self.client = Groq()
        self.model = "mixtral-8x7b-32768"  # Using Mixtral model
        
    def generate_response(self, 
                         query: str, 
                         context: List[str], 
                         chat_history: List[dict]) -> str:
        """Generate a response using the Groq LLM"""
        try:
            # Format chat history
            formatted_history = "\n".join([
                f"User: {msg['user']}\nAssistant: {msg['assistant']}"
                for msg in chat_history[-5:]  # Keep last 5 messages for context
            ])
            
            # Construct prompt
            prompt = f"""Based on the following context and chat history, 
            please provide a helpful response to the user's question.
            
            Context:
            {' '.join(context)}
            
            Chat History:
            {formatted_history}
            
            User Question: {query}
            
            Assistant:"""
            
            # Generate response
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1024
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
