"""
Code to get the MF2 dataset.
"""

from datasets import load_dataset, Dataset

MOVIES_DICT = {
    1: {"title": "The last chance", "year": 1945},
    2: {"title": "They Made Me a Criminal", "year": 1939},
    3: {"title": "Tokyo After Dark", "year": 1959},
    4: {"title": "The Sadist", "year": 1963},
    5: {"title": "Suddenly", "year": 1954},
    6: {"title": "Sabotage (Hitchcock)", "year": 1936},
    7: {"title": "Murder By Contract", "year": 1958},
    8: {"title": "Pushover", "year": 1954},
    9: {"title": "Go for Broke", "year": 1951},
    10: {"title": "Meet John Doe", "year": 1941},
    11: {"title": "Scarlet Street", "year": 1945},
    12: {"title": "Little Lord Fauntleroy", "year": 1936},
    13: {"title": "Deadline - U.S.A.", "year": 1952},
    14: {"title": "My Favorite Brunette", "year": 1947},
    15: {"title": "Woman in the Moon", "year": 1929},
    16: {"title": "Lonely Wives", "year": 1931},
    17: {"title": "Nothing Sacred", "year": 1937},
    18: {"title": "Fingerman", "year": 1955},
    19: {"title": "Borderline", "year": 1950},
    20: {"title": "Babes in Toyland", "year": 1934},
    21: {"title": "The Man From Utah", "year": 1934},
    22: {"title": "The Man With The Golden Arm", "year": 1955},
    23: {"title": "A Star Is Born", "year": 1937},
    24: {"title": "Africa Screams", "year": 1949},
    25: {"title": "Dementia 13", "year": 1963},
    26: {"title": "Fear and Desire", "year": 1952},
    27: {"title": "The Little Princess", "year": 1939},
    28: {"title": "Father's Little Dividend", "year": 1951},
    29: {"title": "Kansas City Confidential", "year": 1952},
    30: {"title": "Of Human Bondage", "year": 1934},
    31: {"title": "Half Shot at Sunrise", "year": 1930},
    32: {"title": "Bowery at Midnight", "year": 1942},
    33: {"title": "The Emperor Jones", "year": 1933},
    34: {"title": "The Deadly Companions", "year": 1961},
    35: {"title": "The Red House", "year": 1947},
    36: {"title": "Trapped", "year": 1949},
    37: {"title": "City of Fear", "year": 1959},
    38: {"title": "Kid Monk Baroni", "year": 1952},
    39: {"title": "Tight Spot", "year": 1955},
    40: {"title": "Captain Kidd", "year": 1945},
    41: {"title": "Algiers", "year": 1938},
    42: {"title": "The Front Page", "year": 1931},
    43: {"title": "The Hitch-Hiker", "year": 1953},
    44: {"title": "Obsession", "year": 1949},
    45: {"title": "Thunderbolt", "year": 1929},
    46: {"title": "Cyrano de Bergerac", "year": 1950},
    47: {"title": "Scandal Sheet", "year": 1952},
    48: {"title": "Ladies in Retirement", "year": 1941},
    49: {"title": "Detour", "year": 1945},
    50: {"title": "The Crooked Way", "year": 1949},
    51: {"title": "A Bucket of Blood", "year": 1959},
    52: {"title": "Love Affair", "year": 1939},
    53: {"title": "The Jackie Robinson Story", "year": 1950},
    54: {"title": "The Last Time I Saw Paris", "year": 1954}
}

def add_movie_info_helper(example):
    """
    Add movie title and year to a single example.
    """
    movie_id = example['movie_id']
    
    # Get movie info from MOVIES_DICT
    if movie_id in MOVIES_DICT:
        movie_info = MOVIES_DICT[movie_id]
        example['title'] = movie_info['title']
        example['year'] = movie_info['year']
    else:
        # Handle missing movie_id
        example['title'] = "Unknown"
        example['year'] = None
        
    return example

def add_movie_info(regular_dataset):
    """
    Add movie title and year fields to the dataset based on movie_id.
    
    Args:
        regular_dataset: Hugging Face Dataset with movie_id field
        
    Returns:
        Dataset with added title and year fields
    """
    # Apply the function to all examples in the dataset
    enhanced_dataset = regular_dataset.map(add_movie_info_helper)
    
    print(f"Added title and year fields to {len(enhanced_dataset)} examples")
    
    return enhanced_dataset


def get_mf2_dataset():
    """
    Get the MF2 dataset.
    """
    dataset = load_dataset("sardinelab/MF2", split="test", streaming=True)
    dataset = dataset.remove_columns('video')

    # Convert streaming dataset to regular dataset by iterating through it
    print("Loading MF2 dataset...")
    
    # Collect all data from the streaming dataset
    data_list = []
    for i, example in enumerate(dataset):
        data_list.append(example)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} examples...")
    
    print(f"Total examples loaded: {len(data_list)}")
    
    # Create a regular Dataset from the collected data
    regular_dataset = Dataset.from_list(data_list)

    # Add movie title and year fields to the dataset
    regular_dataset = add_movie_info(regular_dataset)
    
    # Save the dataset
    output_path = "mf2/mf2_dataset.json"
    regular_dataset.to_json(output_path)
    print(f"Dataset saved to: {output_path}")
    
    return regular_dataset

if __name__ == "__main__":
    get_mf2_dataset()