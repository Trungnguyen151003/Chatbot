import time
from llama_cpp import Llama

class LocalModelProcessor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.conversation_history = []
        
        self.model = Llama(
            model_path,
            n_ctx=131072,
            n_threads=10,
            n_gpu_layers=-1,
            verbose=False
        )

    def clear_thinking(self, ai_response):
        lines = ai_response.strip().split('\n')
        cleaned_lines = []
        thinking = False
        
        for line in lines:
            line = line.strip()
            if line == "<think>":
                thinking = True
                continue
            elif line == "</think>":
                thinking = False
                continue
            
            if not thinking and not any(
                line.startswith(prefix) for prefix in [
                    "Okay, so", "Let me start by thinking", "But I'm not 100% clear", "I think", "I'm trying to understand"
                ]
            ):
                cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines).strip()

    def generate_response(self, prompt, history=None):
        history = history or []
        messages = [{"role": "user" if h.startswith("User:") else "assistant", "content": h.split(":", 1)[1].strip()} for h in history]
        messages.append({"role": "user", "content": prompt})
        
        try:
            response_stream = self.model.create_chat_completion(
                messages=messages, temperature=0.8, max_tokens=1024, stop=["<|im_end|>"], stream=True
            )
            
            response_text = ""
            for chunk in response_stream:
                text = chunk['choices'][0]['delta'].get('content', '')
                print(text, end='', flush=True)
                response_text += text
            
            return response_text.strip()
        except Exception as e:
            print(f"Error: {e}")
            return None

    def chat(self, user_input):
        response = self.generate_response(user_input, self.conversation_history)
        if response:
            self.conversation_history.extend([f"User: {user_input}", f"AI: {response}"])
        else:
            print("AI: Sorry, I could not generate a response.")

if __name__ == '__main__':
    process = LocalModelProcessor("E:/Phi-3.5-mini-instruct-IQ2_M.gguf")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break
        start_time = time.time()
        process.chat(user_input)
        print(f"[Response time: {time.time() - start_time:.2f} seconds]")
