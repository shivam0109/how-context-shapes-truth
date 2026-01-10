from openai import OpenAI
import os 
import json
import argparse
import pandas as pd 

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

ROLE = "You are an expert annotator for converting questions into claims."
PROMPT = """Instructions: 
Convert the following question into claims. 
Do not answer the question, just rephrase it as a neutral claim. 

[Question]: {question}
""" 

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def process_chunk(chunk):
    """Process a chunk of items and return the processed items with rephrased questions."""
    processed_items = []
    for item in chunk:
        try:
            question = item['question']
            prompt = PROMPT.format(question=question)
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": ROLE}, {"role": "user", "content": prompt}],
                stream=False
            )
            item['claim'] = response.choices[0].message.content.strip()
            processed_items.append(item)
        except Exception as e:
            print(f"Error processing item: {e}")
            # Keep original item without rephrased_claim if processing fails
            item['claim'] = "ERROR"
            processed_items.append(item)
    return processed_items


def save_jsonl(data, file_path):
    """Save data to JSONL file."""
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')


def main(input_file, output_file, test, chunk_size):
    # Load the data
    print(f"Loading data from {input_file}...")
    df = load_data(input_file)
    
    # Limit to last 20 samples if test mode
    if test:
        df = df.tail(20)
        print(f"Test mode: Processing last {len(df)} samples")
    
    print(f"Total items to process: {len(df)}")
    
    # Convert dataframe to list of dicts for processing
    data = df.to_dict('records')
    
    # Process in chunks
    total_chunks = (len(data) + chunk_size - 1) // chunk_size
    all_processed_items = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunk_num = (i // chunk_size) + 1
        
        print(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} items)...")
        
        # Process the chunk
        processed_chunk = process_chunk(chunk)
        all_processed_items.extend(processed_chunk)
        
        print(f"Chunk {chunk_num} completed")
    
    # Save all processed items to JSONL
    save_jsonl(all_processed_items, output_file)
    
    print(f"Processing complete! Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rephrase questions in ConflictQA dataset')
    parser.add_argument('--input_file', default='conflictqa/conflictqa-raw.csv', help='Path to input CSV file')
    parser.add_argument('--output_file', default='conflictqa/conflictqa_rephrased_claims.jsonl', help='Path to output JSONL file')
    parser.add_argument('--test', action='store_true', help='Process only last 20 samples for testing')
    parser.add_argument('--chunk_size', type=int, default=100, help='Number of items to process in each chunk (default: 100)')
    
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.test, args.chunk_size)
