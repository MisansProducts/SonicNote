import ollama
import os
import textwrap
import socket
import subprocess
import time
import signal

HOST = "127.0.0.1"
PORT = 11434

def is_port_open(host=HOST, port=PORT, timeout=0.2) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0

class OllamaServer:
    def __init__(self, host=HOST, port=PORT, startup_timeout_s=15):
        self.host = host
        self.port = port
        self.startup_timeout_s = startup_timeout_s
        self.proc = None
        self.started_by_me = False

    def __enter__(self):
        print(f"[ollama] Checking server on {self.host}:{self.port} ...")
        if is_port_open(self.host, self.port):
            print("[ollama] Server already running; will reuse it (won't stop it on exit).")
            return self

        print("[ollama] Server not running; starting `ollama serve` ...")

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        self.proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        self.started_by_me = True
        print(f"[ollama] Started process pid={self.proc.pid}. Waiting for server to become ready...")

        deadline = time.time() + self.startup_timeout_s
        last_log = 0.0
        while time.time() < deadline:
            if is_port_open(self.host, self.port):
                print(f"[ollama] Server is up on {self.host}:{self.port}.")
                return self

            if self.proc.poll() is not None:
                raise RuntimeError("[ollama] `ollama serve` exited early (failed to start).")

            # Prints a heartbeat about once per second
            now = time.time()
            if now - last_log >= 1.0:
                remaining = int(deadline - now)
                print(f"[ollama] ...still starting ({remaining}s left)")
                last_log = now

            time.sleep(0.2)

        raise TimeoutError("[ollama] Timed out waiting for Ollama server to start.")

    def __exit__(self, exc_type, exc, tb):
        if not (self.started_by_me and self.proc):
            return

        print("[ollama] Stopping `ollama serve` (started by this script) ...")
        try:
            if os.name == "nt":
                self.proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self.proc.send_signal(signal.SIGINT)
            self.proc.wait(timeout=5)
            print("[ollama] Server stopped cleanly.")
        except Exception:
            print("[ollama] Graceful stop failed; terminating process ...")
            try:
                self.proc.terminate()
                self.proc.wait(timeout=5)
                print("[ollama] Server terminated.")
            except Exception:
                print("[ollama] Terminate failed; killing process ...")
                self.proc.kill()
                print("[ollama] Server killed.")

def summarize_file(file_path, model='qwen2.5'):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return
    
    # Reads the file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Construct the prompt with XML tags and "Sandwich" strategy
    system_prompt = "You are an expert meeting summarizer. You output structured summaries in Markdown."
    user_instruction = textwrap.dedent(f"""
        <transcript>
        {text_content}
        </transcript>

        INSTRUCTIONS:
        The text above is a meeting transcript.
        1. Ignore speaker labels (e.g., SPEAKER_04) and timestamps; do not mention them.
        2. Ignore filler words or casual conversation.
        3. Do not write as if you are an observer (i.e., "The conversation revolves around..."); only summarize.
        4. Write a summary of the key discussion points.
        5. Use the following format:
            ### Summary
            [One to two paragraphs in a direct, neutral tone explaining what was discussed and the outcomes]

            ### Key Topics
            - [Topic 1]
            - [Topic 2]
    """).strip()

    # Calls the model and prints the output to the terminal (stream=True)
    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_instruction},
        ],
        stream=True
    )

    # Saves the output to summary_output.txt
    output_filename = os.path.join("text", "summary_output.txt")
    print(f"\n--- SUMMARY (Saving to {output_filename}) ---\n")

    with open(output_filename, 'w', encoding='utf-8') as f_out:
        for chunk in response:
            content = chunk['message']['content']
            print(content, end='', flush=True)
            f_out.write(content)

    print("\n\n--- END ---")

if __name__ == "__main__":
    input_file = os.path.join("text", "transcription_output.txt")
    with OllamaServer():
        summarize_file(input_file, model="qwen2.5")
