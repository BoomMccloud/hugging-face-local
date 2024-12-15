from transformers import pipeline
import torch

# Set up the model (using the same sentiment analysis model as before)
classifier = pipeline(
    task="sentiment-analysis",
    model="aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def analyze_sentiment(text):
    """
    Analyze the sentiment of given text and return formatted results
    """
    result = classifier(text)
    return f"Text: '{text}'\nSentiment: {result[0]['label']}\nConfidence: {result[0]['score']:.3f}"

# Interactive loop for continuous interaction
def interactive_sentiment_analysis():
    print("Enter text to analyze sentiment (type 'quit' to exit):")
    
    while True:
        user_input = input("\nEnter text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if user_input:
            print("\n" + analyze_sentiment(user_input))
        else:
            print("Please enter some text to analyze.")

if __name__ == "__main__":
    interactive_sentiment_analysis()