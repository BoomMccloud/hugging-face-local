import torch
from transformers import pipeline

def setup_chat_model(model_path="./model"):
    """
    Set up the text generation model using CPU with proper truncation settings
    """
    try:
        device = "cpu"
        print("Using CPU for inference")
        
        # Create pipeline with truncation settings
        generator = pipeline(
            "text-generation",
            model=model_path,
            device=device,
            max_length=200,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
            truncation=True,  # Enable truncation
            padding=True      # Enable padding
        )
        
        print(f"Successfully loaded model from {model_path}")
        return generator
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def generate_response(generator, messages):
    """
    Generate a response based on the conversation history
    """
    try:
        # Convert messages to prompt format
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        # Generate response with truncation handling
        response = generator(
            prompt,
            max_new_tokens=100,    # Control the length of generated text
            pad_token_id=generator.tokenizer.pad_token_id,
            truncation=True
        )
        return response[0]['generated_text']
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error generating a response."

def interactive_chat():
    """
    Run an interactive chat session with the model
    """
    print("Initializing chat model from local folder...")
    generator = setup_chat_model()
    
    if not generator:
        print("Failed to initialize the model.")
        return
    
    print("\nChat initialized! Type 'quit' to exit.")
    print("Type your message and press Enter to chat.")
    
    messages = []
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not user_input:
            print("Please enter a message.")
            continue
            
        # Add user message to history
        messages.append({"role": "user", "content": user_input})
        
        # Generate and print response
        response = generate_response(generator, messages)
        print("\nAssistant:", response)
        
        # Add assistant response to history
        messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    interactive_chat()