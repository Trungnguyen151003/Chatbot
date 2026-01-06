import time

from llama_cpp import Llama

model_local_path = "E:/DeepSeek-R1-Distill-Qwen-1.5B-f16.gguf"


class LocalModelProcessor:
    def __init__(self, model_path):
        super().__init__()
        self.thinking = None
        self.messages = None
        self.cleaned_lines = None
        self.readable_response = None
        self.lines = None
        self.ai_response = None
        self.user_input = None
        self.response = None
        self.response_data = None
        self.content = None
        self.role = None
        self.history = None
        self.prompt = None

        self.model_path: str = model_path

        self.n_gpu_layers = -1
        self.n_ctx: int = 131072
        self.n_batch: int = 1024
        self.n_threads: int = 10

        self.max_tokens: int = 1024
        self.temperature: float = 0.8
        self.top_p: float = 0.95
        self.min_p: float = 0.05
        self.typical_p: float = 1.0
        self.frequency_penalty: float = 0.0
        self.presence_penalty: float = 0.0
        self.repeat_penalty: float = 1.0
        self.top_k: int = 40
        self.tfs_z: float = 1.0
        self.mirostat_mode: int = 0
        self.mirostat_tau: float = 5.0
        self.mirostat_eta: float = 0.1

        self.verbose: bool = False
        self.stream: bool = True

        self.conversation_history = []

        self.stop = ["<|im_end|>"]  #["</s>", "###"]

        self.model = Llama(self.model_path,
                           n_ctx=self.n_ctx,
                           n_threads=self.n_threads,
                           n_gpu_layers=self.n_gpu_layers,
                           verbose=self.verbose)

    def clear_thinking(self, ai_response):
        self.ai_response = ai_response
        self.lines = ai_response.strip().split('\n')
        self.cleaned_lines = []
        self.thinking = False
        for line in self.lines:
            line = line.strip()
            if line == "<think>":
                self.thinking = True
                continue
            elif line == "</think>":  # If you expect closing tag - though your example didn't have one
                self.thinking = False
                continue
            if not self.thinking \
                    and not line.startswith("Okay, so") \
                    and not line.startswith("Let me start by thinking") \
                    and not line.startswith("But I'm not 100% clear") \
                    and not line.startswith("I think") \
                    and not line.startswith("I'm trying to understand"):
                self.cleaned_lines.append(line)

        return "\n".join(self.cleaned_lines).strip()

    def generate_response_with_history(self, prompt, history=None):
        self.messages = []
        if history is None:
            history = []
        self.prompt = prompt
        self.history = history
        for turn in history:
            self.role = "user" if turn.startswith("User:") else "assistant"  # Infer role from prefix
            self.content = turn[len(self.role) + 2:].strip()  # Remove "User: " or "AI: " prefix and strip whitespace
            self.messages.append({"role": self.role, "content": self.content})
        self.messages.append({"role": "user", "content": self.prompt})  # Add the current user prompt

        try:
            self.response_data = self.model.create_chat_completion(messages=self.messages,
                                                                   temperature=self.temperature,
                                                                   max_tokens=self.max_tokens,
                                                                   stop=self.stop)  # Stop sequences
            self.response = self.response_data['choices'][0]['message']['content']  # Extract response content
            return self.response.strip()
        except Exception as e:
            print(f"**Error during create_chat_completion:** {e}")
            return None

    def generate_response_with_new(self, user_input):
        self.user_input = user_input
        self.ai_response = process.generate_response_with_history(self.user_input, self.conversation_history)
        if self.ai_response:
            self.conversation_history.append(f"User: {self.user_input}")
            self.conversation_history.append(f"AI: {self.ai_response}")
            return self.ai_response
        else:
            print("AI: Sorry, I could not generate a response.")


process = LocalModelProcessor(model_local_path)


def main():
    while True:
        start_time = time.time()
        user_input = input("User: ")
        if user_input.lower() == "quit":
            break
        mess = process.generate_response_with_new(user_input)
        print(mess)
        #clean thingking
        #print(process.clear_thinking(mess))

        print("[time response: ", time.time()-start_time, "]")


if __name__ == '__main__':
    main()