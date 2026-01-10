from openai import OpenAI
import os 
import json
import argparse
import pandas as pd

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

ROLE = "You are an expert annotator. For the given claim and verdict, produce Yes if claim and verdict match, else produce No. If the verdict is Half True, produce Half True."
PROMPT = """Instructions: 
Input: You will be given a claim and a verdict.
Core Task: Determine if the claim and verdict match.
Output Format: Your output must be either "Yes" or "No". Do not add any explanations, introductions, or other text.

Example 1: 
Claim: Copenhagen is a territory of India
Verdict: India 
Output: Yes 

Example 2:
Claim: New Delhi is a territory of India 
Verdict: Denmark 
Output: No

Example 3:
Claim: Lisbon is a territory of Portugal 
Verdict: Portugal 
Output: Yes 

Claim: {claim}
Verdict: {verdict}
Output: 
"""


def load_csv(file_path):
    return pd.read_csv(file_path)

def process_chunk(chunk):
    """Process a chunk of items and return the processed items with ground truth annotations.
    
    This function preserves all original fields from the CSV and adds a new 'ground_truth' field.
    """
    processed_items = []
    for item in chunk:
        try:
            claim = item['claim']
            verdict = item['factcheck_verdict']
            prompt = PROMPT.format(claim=claim, verdict=verdict)
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": ROLE}, {"role": "user", "content": prompt}],
                stream=False
            )
            item['ground_truth'] = response.choices[0].message.content.strip()
            processed_items.append(item)
        except Exception as e:
            print(f"Error processing item: {e}")
            # Keep original item with ERROR ground_truth if processing fails
            item['ground_truth'] = "ERROR"
            processed_items.append(item)
    return processed_items


def save_jsonl(data, file_path, append=True):
    """Save data to JSONL file, appending if specified."""
    mode = 'a' if append else 'w'
    with open(file_path, mode) as file:
        for item in data:
            file.write(json.dumps(item) + '\n')


def main(input_file, output_file, test, chunk_size):
    # Load the data
    print(f"Loading data from {input_file}...")
    data = load_csv(input_file)
    data = data.to_dict(orient='records')

    # Limit to last 20 samples if test mode
    if test:
        data = data[:20] if len(data) >= 20 else data
        print(f"Test mode: Processing last {len(data)} samples")
    
    print(f"Total items to process: {len(data)}")
    
    # Process in chunks
    total_chunks = (len(data) + chunk_size - 1) // chunk_size
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunk_num = (i // chunk_size) + 1
        
        print(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} items)...")
        
        # Process the chunk
        processed_chunk = process_chunk(chunk)
        
        # Save the processed chunk (append to output file)
        save_jsonl(processed_chunk, output_file, append=True)
        
        # Verify fields are preserved (only for first chunk)
        if chunk_num == 1 and processed_chunk:
            print(f"Sample output fields: {list(processed_chunk[0].keys())}")
            print(f"Total fields in output: {len(processed_chunk[0].keys())}")
        
        print(f"Chunk {chunk_num} completed and saved")
    
    print(f"Processing complete! Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Annotate Druid ground truth annotations')
    parser.add_argument('--input_file', default='druid/borderlines.csv', help='Path to input CSV file')
    parser.add_argument('--output_file', default='druid/borderlines_ground_truth.jsonl', help='Path to output JSONL file')
    parser.add_argument('--test', action='store_true', help='Process only last 20 samples for testing')
    parser.add_argument('--chunk_size', type=int, default=100, help='Number of items to process in each chunk (default: 10)')
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.test, args.chunk_size)