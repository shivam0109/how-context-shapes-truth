"""
Generate random data controlled for length 
1. Random on a character level
2. No syntactic meaning 
3. Syntactically correct sentences with no semantic meaning
4. Semantically correct sentences from Wikipedia
"""

import random
import string
import nltk
import wikipedia # Added for the new Wikipedia extraction mode
import argparse
import pandas as pd
import logging
import os
from datetime import datetime


# --- NLTK Setup ---
# The first time you run this, it will download the necessary NLTK data.
# This provides access to large vocabularies and corpora.
try:
    nltk.data.find('corpora/words')
    nltk.data.find('corpora/brown')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logging.info("Downloading necessary NLTK data...")
    nltk.download('words')
    nltk.download('brown')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    logging.info("Download complete.")


# --- Logging Setup ---
def setup_logging(log_file_path=None):
    """
    Setup logging to both console and file.
    
    Args:
        log_file_path: Path to log file. If None, uses a fixed log file name.
    """
    if log_file_path is None:
        log_file_path = "generate_random_data.log"
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file_path) if os.path.dirname(log_file_path) else "custom_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Full path to log file
    full_log_path = os.path.join(log_dir, os.path.basename(log_file_path))
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(full_log_path, mode='a'),  # Append mode to reuse same file
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    return full_log_path


# --- Dynamic Word Bank Initialization ---
# This will be populated from the NLTK corpus instead of being hard-coded.
WORD_BANK = {}
ALL_ENGLISH_WORDS = []

def _initialize_word_banks():
    """
    Populates the WORD_BANK with words from the NLTK Brown corpus,
    sorted by part of speech, for the 'salad' mode. Also populates a
    general word list for the 'word' mode.
    """
    global WORD_BANK, ALL_ENGLISH_WORDS
    
    # Populate the general word list for the 'word' mode
    ALL_ENGLISH_WORDS = nltk.corpus.words.words()

    # Define parts of speech and their corresponding tags in the Brown corpus
    pos_map = {
        "nouns": {'NN', 'NNS', 'NP', 'NPS'},
        "verbs": {'VB', 'VBD', 'VBG', 'VBN', 'VBZ'},
        "adjectives": {'JJ', 'JJR', 'JJS'},
        "adverbs": {'RB', 'RBR', 'RBS', 'WRB'},
        "pronouns": {'PP', 'PPO', 'PPS', 'PPSS'},
        "articles": {'AT'},
        "conjunctions": {'CC'},
    }

    # Initialize empty lists in the WORD_BANK
    for pos in pos_map:
        WORD_BANK[pos] = []

    # Categorize words from the Brown corpus into the WORD_BANK
    for word, tag in nltk.corpus.brown.tagged_words():
        # Clean the word: lowercase and ensure it's alphabetic
        if not word.isalpha():
            continue
        word = word.lower()

        for pos, tags in pos_map.items():
            if tag.upper() in tags:
                WORD_BANK[pos].append(word)
                break
    
    logging.info("Word banks initialized from NLTK corpus.")


# --- Mode 1: Character-Level Paragraph Generator ---

def _generate_char_paragraph(word_count: int) -> str:
    """
    Generates a paragraph with made-up words at a character level.
    """
    words = []
    for _ in range(word_count):
        word_len = random.randint(2, 10)
        random_word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_len))
        words.append(random_word)
    paragraph = ' '.join(words)
    return paragraph.capitalize() + '.'

# --- Mode 2: Word-Level (No Sense) Paragraph Generator ---

def _generate_word_paragraph(word_count: int) -> str:
    """
    Generates a paragraph using real English words in a random order.
    """
    words = random.choices(ALL_ENGLISH_WORDS, k=word_count)
    paragraph = ' '.join(words)
    return paragraph.capitalize() + '.'

# --- Mode 3: Word Salad (Semantic Nonsense) Paragraph Generator ---

def _generate_salad_paragraph(word_count: int) -> str:
    """
    Generates a "word salad" paragraph with some grammatical structure.
    """
    sentence_structures = [
        ["article", "adjective", "noun", "verb", "adverb"],
        ["pronoun", "verb", "article", "noun"],
        ["noun", "verb", "conjunction", "noun", "verb"],
        ["article", "noun", "adverb", "verb", "adjective", "noun"],
    ]

    words = []
    while len(words) < word_count:
        structure = random.choice(sentence_structures)
        sentence = []
        for part_of_speech in structure:
            key = part_of_speech + "s"
            if key in WORD_BANK and WORD_BANK[key]:
                word = random.choice(WORD_BANK[key])
                sentence.append(word)
            else:
                sentence.append("word") # Fallback

        if sentence:
            sentence[0] = sentence[0].capitalize()
            sentence[-1] += "."
            words.extend(sentence)

    return ' '.join(words)

# --- Mode 4: Wikipedia Paragraph Extractor ---

def _extract_wiki_paragraph(word_count: int, tolerance: float = 0.2, max_retries: int = 10, seed: int = None) -> str:
    """
    Extracts a paragraph or combines multiple paragraphs from a random Wikipedia article 
    to reach the desired word count.

    Args:
        word_count: The target number of words for the paragraph.
        tolerance: The acceptable percentage deviation from the word_count (e.g., 0.2 for 20%).
        max_retries: The number of random articles to try before giving up.
        seed: Random seed for reproducibility. If None, uses current time.

    Returns:
        A paragraph or combined paragraphs from Wikipedia, or an error message if one could not be found.
    """
    # Set random seed for reproducibility
    if seed is not None:
        random.seed(seed)
    
    current_tolerance = tolerance
    total_attempts = 0
    max_total_attempts = max_retries * 3  # Allow for tolerance increases
    
    while total_attempts < max_total_attempts:
        for attempt in range(max_retries):
            total_attempts += 1
            
            try:
                # Get a random Wikipedia page title
                random_title = wikipedia.random(pages=1)
                # Fetch the page content
                page = wikipedia.page(random_title, auto_suggest=False)
                content = page.content
                
                # Split content into paragraphs
                paragraphs = [p.strip() for p in content.split('\n') if len(p.strip().split()) > 10]
                
                # First, try to find a single paragraph that matches
                suitable_paragraphs = []
                for p in paragraphs:
                    p_word_count = len(p.split())
                    if abs(p_word_count - word_count) <= word_count * current_tolerance:
                        suitable_paragraphs.append(p)
                
                if suitable_paragraphs:
                    return random.choice(suitable_paragraphs)
                
                # If no single paragraph works, try combining multiple paragraphs
                combined_result = _combine_paragraphs_to_target(paragraphs, word_count, current_tolerance)
                if combined_result:
                    return combined_result

            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError):
                # If we get a disambiguation page or other error, just try another random page
                continue
            except Exception as e:
                # Catch other potential errors (e.g., network issues)
                logging.error(f"An unexpected error occurred: {e}")
                continue
        
        # If we've exhausted max_retries with current tolerance, increase tolerance
        if total_attempts < max_total_attempts:
            current_tolerance += 0.1
            logging.info(f"Increasing tolerance to {current_tolerance:.1f} for word count {word_count}")
    
    return f"Could not find a suitable paragraph with ~{word_count} words after {total_attempts} attempts (max tolerance: {current_tolerance:.1f})."


def _combine_paragraphs_to_target(paragraphs, target_word_count, tolerance):
    """
    Combines continuous paragraphs to reach the target word count.
    
    Args:
        paragraphs: List of paragraph strings
        target_word_count: Target number of words
        tolerance: Acceptable percentage deviation
        
    Returns:
        Combined paragraph string or None if no suitable combination found
    """
    if not paragraphs:
        return None
    
    min_words = max(1, int(target_word_count * (1 - tolerance)))
    max_words = int(target_word_count * (1 + tolerance))
    
    # Strategy 1: Try continuous paragraph combinations (2-4 paragraphs)
    for start_idx in range(len(paragraphs)):
        for num_paras in range(2, min(5, len(paragraphs) - start_idx + 1)):
            end_idx = start_idx + num_paras
            
            # Get continuous paragraphs
            continuous_paras = paragraphs[start_idx:end_idx]
            total_words = sum(len(p.split()) for p in continuous_paras)
            
            if min_words <= total_words <= max_words:
                # Found a good continuous combination
                combined_text = ' '.join(continuous_paras)
                return combined_text
    
    # Strategy 2: Find the closest single paragraph and extend with adjacent paragraphs
    best_single = None
    best_diff = float('inf')
    
    for idx, para in enumerate(paragraphs):
        word_count = len(para.split())
        diff = abs(word_count - target_word_count)
        if diff < best_diff:
            best_diff = diff
            best_single = idx
    
    if best_single is not None:
        single_word_count = len(paragraphs[best_single].split())
        
        if single_word_count < min_words:
            # Need to add adjacent paragraphs
            # Try extending forward first
            current_words = single_word_count
            end_idx = best_single + 1
            
            while end_idx < len(paragraphs) and current_words < max_words:
                next_words = len(paragraphs[end_idx].split())
                if current_words + next_words <= max_words:
                    current_words += next_words
                    end_idx += 1
                else:
                    break
            
            if min_words <= current_words <= max_words:
                combined_text = ' '.join(paragraphs[best_single:end_idx])
                return combined_text
            
            # Try extending backward if forward didn't work
            current_words = single_word_count
            start_idx = best_single - 1
            
            while start_idx >= 0 and current_words < max_words:
                prev_words = len(paragraphs[start_idx].split())
                if current_words + prev_words <= max_words:
                    current_words += prev_words
                    start_idx -= 1
                else:
                    break
            
            if min_words <= current_words <= max_words:
                combined_text = ' '.join(paragraphs[start_idx + 1:best_single + 1])
                return combined_text
        
        elif single_word_count > max_words:
            # Paragraph is too long, try to truncate it
            words = paragraphs[best_single].split()
            if len(words) > min_words:
                truncated = ' '.join(words[:target_word_count])
                return truncated
    
    # Strategy 3: Greedy approach with continuous paragraphs
    for start_idx in range(len(paragraphs)):
        current_words = 0
        end_idx = start_idx
        
        while end_idx < len(paragraphs):
            next_words = len(paragraphs[end_idx].split())
            if current_words + next_words <= max_words:
                current_words += next_words
                end_idx += 1
                
                if current_words >= min_words:
                    # Found a suitable continuous range
                    combined_text = ' '.join(paragraphs[start_idx:end_idx])
                    return combined_text
            else:
                break
    
    return None


def _shuffle_contexts_no_match(contexts: list[str], claims: list[str], seed: int = None) -> list[str]:
    """
    Shuffles contexts such that no claim is paired with any of its original contexts.
    Ensures:
    1. No context is in its original position (derangement: shuffled_indices[i] != i)
    2. For each claim, the shuffled context is not one of the contexts it was originally paired with
    
    Args:
        contexts: List of original context strings
        claims: List of claim strings (used to ensure no matching)
        seed: Random seed for reproducibility
        
    Returns:
        A list of shuffled context strings where no claim is paired with any of its original contexts
    """
    if seed is not None:
        random.seed(seed)
    
    n = len(contexts)
    if n == 0:
        return []
    if n == 1:
        # Can't shuffle a single item, but we can still return it
        return contexts.copy()
    
    # Build a dictionary mapping each claim to all contexts it was originally paired with
    claim_to_contexts = {}
    for claim, context in zip(claims, contexts):
        if claim not in claim_to_contexts:
            claim_to_contexts[claim] = set()
        claim_to_contexts[claim].add(context)
    
    # Create indices
    shuffled_indices = list(range(n))
    
    # Perform derangement: shuffle until:
    # 1. No context is in its original position (shuffled_indices[i] != i)
    # 2. For each claim, the shuffled context is not in its original context set
    max_attempts = 1000
    attempts = 0
    
    while attempts < max_attempts:
        random.shuffle(shuffled_indices)
        
        # Check if it's a valid shuffle
        valid = True
        for i in range(n):
            # Check if context is in its original position
            if shuffled_indices[i] == i:
                valid = False
                break
            # Check if the shuffled context was originally paired with this claim
            claim = claims[i]
            shuffled_context = contexts[shuffled_indices[i]]
            if claim in claim_to_contexts and shuffled_context in claim_to_contexts[claim]:
                valid = False
                break
        
        if valid:
            break
        
        attempts += 1
    
    if attempts >= max_attempts:
        logging.warning(f"Could not find perfect derangement after {max_attempts} attempts. "
                       f"Using best-effort shuffle.")
        # Fallback: try to minimize matches
        best_shuffle = shuffled_indices.copy()
        best_matches = sum(1 for i in range(n) 
                          if shuffled_indices[i] == i or 
                          (claims[i] in claim_to_contexts and 
                           contexts[shuffled_indices[i]] in claim_to_contexts[claims[i]]))
        
        for _ in range(100):
            random.shuffle(shuffled_indices)
            matches = sum(1 for i in range(n) 
                         if shuffled_indices[i] == i or 
                         (claims[i] in claim_to_contexts and 
                          contexts[shuffled_indices[i]] in claim_to_contexts[claims[i]]))
            if matches < best_matches:
                best_matches = matches
                best_shuffle = shuffled_indices.copy()
                if best_matches == 0:
                    break
        
        shuffled_indices = best_shuffle
    
    # Return shuffled contexts
    return [contexts[i] for i in shuffled_indices]


# --- Main Function ---

def generate_paragraphs(paragraph_word_counts: list[int], mode: str = 'salad', seed: int = None) -> list[str]:
    """
    Generates a list of random paragraphs based on specified word counts and mode.

    Args:
        paragraph_word_counts: A list of integers for desired word counts.
        mode: The generation mode: 'char', 'word', 'salad', or 'wiki'.
        seed: Random seed for reproducibility. If None, uses current time.

    Returns:
        A list of strings, where each string is a generated paragraph.
    """
    generation_functions = {
        'char': _generate_char_paragraph,
        'word': _generate_word_paragraph,
        'salad': _generate_salad_paragraph,
        'wiki': _extract_wiki_paragraph
    }

    if mode not in generation_functions:
        raise ValueError(f"Invalid mode '{mode}'. Available modes are: {list(generation_functions.keys())}")

    selected_function = generation_functions[mode]
    
    # Set random seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
    
    paragraphs = []
    for i, count in enumerate(paragraph_word_counts):
        logging.info(f"Generating paragraph {i+1} with {count} words")
        if count > 0:
            if mode == 'wiki':
                # For wiki mode, use a different seed for each paragraph to ensure variety
                paragraph_seed = seed + i if seed is not None else None
                paragraphs.append(selected_function(count, seed=paragraph_seed))
            else:
                paragraphs.append(selected_function(count))
        
    return paragraphs


def get_counts(input_path, context_col):
    df = pd.read_csv(input_path)
    df[context_col + '_length'] = df[context_col].apply(lambda x: len(x.split()))
    return df, df[context_col + '_length'].tolist()


def main(input_path, output_path, merge_df_path, context_col, seed: int = None, method: str = 'all'):
    _initialize_word_banks()
    
    df, counts = get_counts(input_path, context_col)

    # Generate methods based on the method argument
    if method in ['all', 'char']:
        logging.info("\n--- 1. Character-Level Randomness ---")
        char_paragraphs = generate_paragraphs(counts, mode='char', seed=seed)
        df['random_char'] = char_paragraphs

    if method in ['all', 'word']:
        logging.info("\n--- 2. Word-Level Randomness (No Sense) ---")
        word_paragraphs = generate_paragraphs(counts, mode='word', seed=seed)
        df['random_word'] = word_paragraphs
    
    if method in ['all', 'salad']:
        logging.info("\n--- 3. Word Salad (Semantic Nonsense) ---")
        salad_paragraphs = generate_paragraphs(counts, mode='salad', seed=seed)
        df['random_salad'] = salad_paragraphs
    
    if method in ['all', 'wiki']:
        logging.info("\n--- 4. Real Paragraphs from Wikipedia ---")
        
        # Note: Wikipedia extraction can be slower due to network requests
        wiki_paragraphs = generate_paragraphs(counts, mode='wiki', seed=seed)
        df['random_wiki'] = wiki_paragraphs
        
        # Count how many Wikipedia paragraphs failed to extract
        failed_wiki_count = sum(1 for paragraph in wiki_paragraphs if paragraph.startswith("Could not find a suitable paragraph"))
        total_wiki_attempts = len(wiki_paragraphs)
        successful_wiki_count = total_wiki_attempts - failed_wiki_count
        
        logging.info(f"\n--- Wikipedia Extraction Summary ---")
        logging.info(f"Total attempts: {total_wiki_attempts}")
        logging.info(f"Successful extractions: {successful_wiki_count}")
        logging.info(f"Failed extractions: {failed_wiki_count}")
        logging.info(f"Success rate: {(successful_wiki_count/total_wiki_attempts)*100:.1f}%")
    
    if method in ['all', 'shuffle']:
        logging.info("\n--- 5. Shuffled Context ---")
        # Get original contexts and claims for shuffling
        original_contexts = df[context_col].tolist()
        original_claims = df['claim'].tolist()
        shuffled_contexts = _shuffle_contexts_no_match(original_contexts, original_claims, seed=seed)
        df['random_shuffled_context'] = shuffled_contexts
    
    df_orig = pd.read_csv(merge_df_path)
    df_orig['claim'] = df_orig['claim'].apply(lambda x: x.strip())
    df_orig[context_col] = df_orig[context_col].apply(lambda x: x.strip())
    logging.info("Shape of original dataframe: " + str(df_orig.shape))
    df['claim'] = df['claim'].apply(lambda x: x.strip())
    df[context_col] = df[context_col].apply(lambda x: x.strip())
    logging.info("Shape of random dataframe: " + str(df.shape))
    if 'index' in df_orig.columns:
        df_merged = df.merge(df_orig, on=['claim', context_col, 'index'], how='inner')
        logging.info("Shape of merged dataframe: " + str(df_merged.shape))
        df_merged = df_merged.drop_duplicates(subset=['claim', context_col, 'index'])
        logging.info("Shape of merged dataframe after dropping duplicates: " + str(df_merged.shape))
    else:
        df_merged = df.merge(df_orig, on=['claim', context_col], how='inner')
        logging.info("Shape of merged dataframe: " + str(df_merged.shape))
        df_merged = df_merged.drop_duplicates(subset=['claim', context_col])
        logging.info("Shape of merged dataframe after dropping duplicates: " + str(df_merged.shape))
    df_merged.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate random paragraphs")
    parser.add_argument("--input_path", type=str, default="../get_datasets/legalbench/corporate_lobbying/corporate_lobbying.csv", help="Path to input file")
    parser.add_argument("--output_path", type=str, default="../get_datasets/legalbench/corporate_lobbying/df_corporate_lobbying_company_random_data_v2.csv", help="Path to output file")
    parser.add_argument("--merge_df_path", type=str, default="../get_datasets/legalbench/corporate_lobbying/corporate_lobbying.csv", help="path to dataframe to merge with")
    parser.add_argument("--context_col", type=str, default="company_description", help="context column name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log_file", type=str, default=None, help="Path to log file (optional, defaults to timestamped file)")
    parser.add_argument("--method", type=str, default="all", choices=['all', 'char', 'word', 'salad', 'wiki', 'shuffle'],
                       help="Generation method to use: 'all' (default), 'char', 'word', 'salad', 'wiki', or 'shuffle'")
    args = parser.parse_args()

    # Setup logging
    log_file_path = setup_logging(args.log_file)
    logging.info(f"Logging to file: {log_file_path}")
    logging.info(f"Starting random data generation with seed: {args.seed} and method: {args.method}")

    main(args.input_path, args.output_path, args.merge_df_path, args.context_col, args.seed, args.method)
