import torch
from transformers import pipeline
import gc

def clean_memory():
    """
    Clean up memory for both MPS and CPU
    """
    gc.collect()
    if hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()

def setup_chat_model(model_path="./model", force_cpu=False):
    """
    Set up the text generation model with MPS optimization when possible
    """
    try:
        # Check for MPS availability
        if not force_cpu and torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple Silicon (MPS) for acceleration")
            torch_dtype = torch.float16  # Use half precision for MPS
        else:
            device = "cpu"
            print("Using CPU for inference")
            torch_dtype = torch.float32
            
        # Create pipeline with optimized settings
        generator = pipeline(
            "text-generation",
            model=model_path,
            device=device,
            torch_dtype=torch_dtype,
            max_length=200,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
            truncation=True,
            padding=True
        )
        
        print(f"Successfully loaded model from {model_path}")
        return generator
    
    except Exception as e:
        if "MPS backend out of memory" in str(e):
            print("MPS out of memory, falling back to CPU...")
            return setup_chat_model(model_path, force_cpu=True)
        print(f"Error loading model: {str(e)}")
        return None

def generate_response(generator, messages, max_history=5):
    """
    Generate a response with memory management
    """
    try:
        # Limit conversation history for memory management
        recent_messages = messages[-max_history:] if len(messages) > max_history else messages
        
        # Convert messages to prompt format
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
        
        # Clean memory before generation
        clean_memory()
        
        # Generate response
        with torch.inference_mode():  # Use inference mode for better memory usage
            response = generator(
                prompt,
                max_new_tokens=100,
                pad_token_id=generator.tokenizer.pad_token_id,
                truncation=True,
                do_sample=True,
                num_return_sequences=1
            )
        
        # Clean memory after generation
        clean_memory()
        
        return response[0]['generated_text']
    except Exception as e:
        if "MPS backend out of memory" in str(e):
            print("MPS out of memory during generation, trying CPU...")
            # Temporarily move to CPU for this generation
            original_device = generator.device
            generator.device = "cpu"
            result = generate_response(generator, recent_messages)
            generator.device = original_device
            return result
        print(f"Error generating response: {str(e)}")
        return "Sorry, I encountered an error generating a response."

def interactive_chat():
    """
    Run an interactive chat session with the model
    """
    print("Initializing chat model...")
    generator = setup_chat_model()
    
    if not generator:
        print("Failed to initialize the model.")
        return
    
    print("\nChat initialized! Type 'quit' to exit.")
    print("Type your message and press Enter to chat.")
    
    messages = []
    
    while True:
        # Clean memory at the start of each loop
        clean_memory()
        
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