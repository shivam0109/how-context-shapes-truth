import json

def count_unique_company_values(jsonl_file: str):
    """
    Count the number of true and false values for unique_company field.
    """
    unique_true_count = 0
    unique_false_count = 0
    total_entries = 0
    
    print(f"Counting unique_company values in {jsonl_file}...")
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                total_entries += 1
                
                unique_company = data.get('unique_company')
                if unique_company is True:
                    unique_true_count += 1
                elif unique_company is False:
                    unique_false_count += 1
                else:
                    print(f"  Warning: Line {line_num} has unexpected unique_company value: {unique_company}")
                    
            except json.JSONDecodeError as e:
                print(f"  Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"  Error processing line {line_num}: {e}")
                continue
    
    print(f"\n✓ Count complete!")
    print(f"  - Total entries: {total_entries}")
    print(f"  - unique_company = true: {unique_true_count}")
    print(f"  - unique_company = false: {unique_false_count}")
    print(f"  - Percentage true: {(unique_true_count/total_entries)*100:.1f}%")
    print(f"  - Percentage false: {(unique_false_count/total_entries)*100:.1f}%")

if __name__ == "__main__":
    count_unique_company_values("legalbench/privacyqa_company_name_processed.jsonl")
