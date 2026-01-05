import ollama
import os
import textwrap

def summarize_file(file_path, model='qwen2.5'):
    """
    Reads a text file and summarizes it using a local Ollama model.
    """
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

    print(f"Read {len(text_content)} characters from {file_path}...")
    print(f"Sending to {model} for summarization (this may take a moment)...")

    # Constructs the prompt
    # Construct the prompt with XML tags and "Sandwich" strategy
    system_prompt = "You are an expert meeting summarizer. You output structured summaries in Markdown."
    
    # We wrap the content in <transcript> tags and put the instruction AFTER the text.
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
    # Calls the model
    try:
        response = ollama.chat(
            model='qwen2.5',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_instruction},
            ],
            stream=True
        )

        output_filename = "summary_output.txt"
        print(f"\n--- SUMMARY (Saving to {output_filename}) ---\n")

        with open(output_filename, 'w', encoding='utf-8') as f_out:
            for chunk in response:
                content = chunk['message']['content']
                
                print(content, end='', flush=True)
                f_out.write(content)

        print("\n\n--- END ---")

    except Exception as e:
        print(f"\nError calling Ollama: {e}")
        print("Make sure you pulled the correct model:\nollama pull qwen2.5")

if __name__ == "__main__":
    input_file = "transcription_results.txt"
    summarize_file(input_file)
