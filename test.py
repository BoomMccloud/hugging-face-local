import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def setup_local_optimized_model(model_path="./model", task="sentiment-analysis"):
    """
    Set up a local model optimized for MacBook with Apple Silicon
    """
    # Check if MPS (Metal Performance Shaders) is available
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using Apple Silicon (MPS) for acceleration")
    else:
        device = "cpu"
        print("MPS not available, falling back to CPU")
    
    try:
        # Load local tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Create pipeline with local model and tokenizer
        classifier = pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        print(f"Successfully loaded model from {model_path}")
        return classifier
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def test_performance(model):
    """
    Run a simple performance test
    """
    import time
    
    test_texts = [
        "This is a test sentence for performance measurement.",
    ] * 10  # Run 10 predictions to warm up
    
    # Warm up
    _ = model(test_texts)
    
    # Actual timing
    start_time = time.time()
    results = model(test_texts)
    end_time = time.time()
    
    print(f"\nProcessed {len(test_texts)} predictions in {end_time - start_time:.3f} seconds")
    print(f"Average time per prediction: {(end_time - start_time) / len(test_texts):.3f} seconds")

if __name__ == "__main__":
    # Setup model from local path
    model_path = "./model"  # Change this to your model's path if different
    classifier = setup_local_optimized_model(model_path)
    
    if classifier:
        # Test it
        test_sentence = "I really love this new optimization!"
        result = classifier(test_sentence)
        print(f"\nTest prediction: {result}")
        
        # Run performance test
        test_performance(classifier)