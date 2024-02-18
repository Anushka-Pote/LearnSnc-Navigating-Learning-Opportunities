import openai

class HealthChatbot:
    def __init__(self, api_key, model="text-davinci-003"):
        openai.api_key = api_key
        self.model = model

    def generate_response(self, query):
        prompt = f"Health question: {query}"
        response = openai.Completion.create(
            engine=self.model,
            prompt=prompt,
            temperature=0.7,
            max_tokens=150,
            n=1,
            stop=None
        )
        return response.choices[0].text.strip()

# Example usage
if __name__ == "__main__":
    api_key = "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"  # Replace with your actual OpenAI API key
    chatbot = HealthChatbot(api_key)

    queries = [
        "How to prevent COVID?",
        "What are the symptoms of flu?",
        "Tell me about a healthy diet.",
        "Explain vaccination to me.",
        "Can you cure cancer?"
    ]

    for query in queries:
        response = chatbot.generate_response(query)
        print(f"User: {query}")
        print(f"Chatbot: {response}\n")
