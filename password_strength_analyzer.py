import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import math
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

!pip install pyahocorasick

import ahocorasick

max_seq_length = 61

vocab_set = pd.read_csv("/kaggle/input/english-word-frequency-list/ngram_freq.csv")

# token_table = pd.read_csv('/kaggle/input/lookup/token_lookup.csv')

# token_table = token_table[token_table['Key'].str.len() > 1]

vocab_df = vocab_set.copy()

vocab_df = vocab_df[vocab_df['word'].str.len() > 1 ] 

def classify_words_by_quantiles(df):
    actual_words = df.copy()
    # Calculate quantile thresholds
    q_25, q_50, q_75, q_90 = actual_words['count'].quantile([0.25, 0.5, 0.75, 0.90])

    # Classify based on quantiles using pd.cut
    bins = [-float('inf'), q_25, q_50, q_75, q_90, float('inf')]
    labels = ['very_low', 'low', 'medium', 'high', 'very_high']

    actual_words['Occurance'] = pd.cut(actual_words['count'], bins=bins, labels=labels)

    # Create the new DataFrame directly
    new_vocab = actual_words[['word', 'count', 'Occurance']].copy()

    return new_vocab

def calculate_entropy(password):
    # A rough estimation using Shannon entropy.
    if not password:
        return 0
    freq = {}
    for char in password:
        freq[char] = freq.get(char, 0) + 1
    entropy = 0.0
    for count in freq.values():
        p = count / len(password)
        entropy -= p * math.log2(p)
    return entropy

new_vocab_df = classify_words_by_quantiles(vocab_df)

new_vocab_df.drop(columns = 'count' , inplace =  True)

new_vocab_df.to_csv('/kaggle/working/vocab_tier.csv')

# Convert to dictionary for fast lookup
vocab_tiers = dict(zip(new_vocab_df['word'], new_vocab_df['Occurance']))

# Define tier priority
tier_priority = {'very_low': 1, 'low': 2, 'medium': 3 , 'high':4 , 'very_high':5}  # Higher number = higher priority

# Initialize the automaton
automaton = ahocorasick.Automaton()

# Add words to automaton with tier info
for word, tier in vocab_tiers.items():
    automaton.add_word(word.lower(), (word.lower(), tier))

# Finalize automaton for fast searching
automaton.make_automaton()

def check_password_debug(password):
    # Normalize the password to lowercase.
    text = str(password).lower()
    matched_words = set()  # Using a set to keep unique matched words.
    highest_tier = "none"    # Default tier if no words are found.
    highest_priority = 0

    # Iterate over all matches in the password using the automaton.
    for end_index, (word, tier) in automaton.iter(text):
        matched_words.add(word)  # Add the word to our set.
        current_priority = tier_priority.get(tier, 0)
        # Update the highest tier if this word's tier has a higher priority.
        if current_priority > highest_priority:
            highest_priority = current_priority
            highest_tier = tier

    # Return a sorted list (optional, for easier debugging) and the highest tier.
    return  highest_tier



model_path = '/kaggle/input/password-strength-checking-model/tensorflow1/default/1/model.h5'  # Adjust if your .h5 file is elsewhere
model = load_model(model_path)

from tokenizers import ByteLevelBPETokenizer

# Define the paths for vocab and merges files
vocab_path = "/kaggle/input/tokenizer/other/default/1/Tokenizer/vocab.json"       # Update with the correct local path
merges_path = "/kaggle/input/tokenizer/other/default/1/Tokenizer/merges.txt"      # Update with the correct local path

# Initialize the ByteLevel BPE tokenizer
my_tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)

def feature_extract(password: str):
    features = {}
    password = str(password)

    # If empty
    if len(password) == 0:
        features['num_upper'] = features['num_lower'] = features['num_digits'] = features['num_special'] = 0
        features['upper_ratio'] = features['lower_ratio'] = features['digit_ratio'] = features['special_ratio'] = 0
    else:
        features['length']= len(password)
        features['uppercase']= sum(1 for char in password if char.isupper())
        features['lowercase']= sum(1 for char in password if char.islower())
        features['digits']= sum(1 for char in password if char.isdigit())
        features['special_chars']= sum(1 for char in password if not char.isalnum())
        features['vocab_tier'] = check_password_debug(password)
        features['num_upper'] = sum(1 for c in password if c.isupper())
        features['num_lower'] = sum(1 for c in password if c.islower())
        features['num_digits'] = sum(1 for c in password if c.isdigit())
        features['num_special'] = len(password) - (
        features['num_upper'] + features['num_lower'] + features['num_digits']
        )

        features['upper_ratio'] = features['num_upper'] / len(password)
        features['lower_ratio'] = features['num_lower'] / len(password)
        features['digit_ratio'] = features['num_digits'] / len(password)
        features['special_ratio'] = features['num_special'] / len(password)
        features['entropy'] = calculate_entropy(password)

    # Example: If your model expects 15 features total, fill out the rest
    # with additional logic. For demonstration, let's say we have 8 features:
    # [num_upper, num_lower, num_digits, num_special, upper_ratio, lower_ratio, digit_ratio, special_ratio]
    return [
        features['length'],
        features['uppercase'],
        features['lowercase'],
        features['digits'],
        features['special_chars'],
        features['vocab_tier'],
        features['num_upper'],
        features['num_lower'],
        features['num_digits'],
        features['num_special'],
        features['upper_ratio'],
        features['lower_ratio'],
        features['digit_ratio'],
        features['special_ratio'],
        features['entropy']
    ]

def process_sequences(sequences, max_seq_length, padding_value=0):
    # Use pad_sequences to pad or truncate the sequences
    processed_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=max_seq_length,
        padding='post',  # You can change to 'pre' if needed
        truncating='post',  # 'post' truncates from the end
        value=padding_value
    )
    
    return processed_sequences

def predict_password_strength(password_seq , features_seq, model , max_seq_length=61):
    seq_padded = password_seq
    eng_features =features_seq
    preds = model.predict({'input_seq': seq_padded, 'input_eng': eng_features}, verbose=0)
    predicted_class = np.argmax(preds, axis=1)[0]

    # d) Map numeric class -> label
    strength_labels = {0: 'Weak', 1: 'Medium', 2: 'Strong'}
    return strength_labels.get(predicted_class, 'Unknown')


# token_table.drop(columns = 'Unnamed: 0' , inplace = True)

# token_to_index = token_table.to_dict()

# nested_dict = token_to_index['Key']
# flat_token_to_index = {token: index for index, token in nested_dict.items()}

# def tokens_to_indices(token_list, token_to_index):
#     # For each token, get its index (if not found, default to 0)
#     return [token_to_index.get(token, 0) for token in token_list]

sample_text = '95vjo5jvi3ivnh!T'
testing = feature_extract(sample_text)

mapping = {'very_high':4 , 'high':3 , 'medium':2 , 'none':1}
testing = [mapping[item] if item in mapping else item for item in testing]

password = str(sample_text)
seq_tokens = (my_tokenizer.encode(password)).tokens
indices = tokens_to_indices(seq_tokens, flat_token_to_index)
padded_seq = process_sequences([indices], max_seq_length)
padded_seq 

padded_seq = np.array(padded_seq)
testing = np.array(testing).reshape(1,-1)

predict_password_strength(padded_seq , testing , model)

def password_to_score(password_class:str):
    tier = password_class
    maps = {'Weak': 0 , 'Medium':5 , 'Strong':10}
    return maps[password_class]

password_to_score(predict_password_strength(padded_seq , testing , model))

def rules1(features):
    flag = True
    features_list = features.tolist()
    features_list = [item for sublist in features_list for item in sublist]
    if features_list[0] < 8 : 
        return [not flag , 'Password should have minimum length of 8 ']
    for i in range(1,5):
        if features_list[i] < 1 : 
            return [not flag , 'Password should contain 1 characters form all 4 letters group']
    else :
        return flag 
         
    
    

rules1(testing)

def rules2(password: str) -> bool:
    """
    Rule 6: Prohibit Sequential Characters
    Returns True if the password does NOT contain sequences like 'abc', '123'.
    (Basic check: look for ascending sequences of length 3.)
    """
    # Simple approach: check each triplet
    for i in range(len(password) - 2):
        c1, c2, c3 = password[i], password[i+1], password[i+2]
        # Check ascending sequence in ASCII
        if ord(c2) == ord(c1) + 1 and ord(c3) == ord(c2) + 1:
            return True
    return False

rules2(password)

def rules3(password: str, max_repeats: int = 2) -> bool:
    """
    Rule 7: Prohibit Repetitive Characters
    Returns True if no character repeats more than 'max_repeats' times in a row.
    """
    count = 1
    for i in range(1, len(password)):
        if password[i] == password[i-1]:
            count += 1
            if count > max_repeats:
                return True
        else:
            count = 1
    return False

rules3('aaa')

common = vocab_set[:10000]

common = common[common['word'].str.len() > 1]

common_eng_vocab = common.to_dict()

nested_vocab = common_eng_vocab['word']
flat_vocab_dict = {token: index for index, token in nested_vocab.items()}

import json

my_dict = flat_vocab_dict

json_output = json.dumps(my_dict, indent=4)  # indent for pretty printing
print(json_output)

# To write to a file:
with open('top10000common.json', 'w') as json_file:
    json.dump(my_dict, json_file, indent=4)
print(json_output)

# To write to a file:
with open('/kaggle/working/', 'w') as json_file:
    json.dump(my_dict, json_file, indent=4)

with open("/kaggle/input/top10000-common-eng-vocab/top10000common.json" , "r",encoding = "utf-8")as file:
    common_words = json.load(file)

def rules4(password: str, dictionary_list=None) -> bool:
    if dictionary_list is None:
        dictionary_list = common_words
    lower_pass = password.lower()
    for word in dictionary_list:
        if word in lower_pass:
            return [True , word]
    return False

def rules5(password: str, dictionary_list=None) -> bool:
    if dictionary_list is None:
        dictionary_list = common_words
    # Very simple substitution map
    subs = str.maketrans({
    "@": "a",
    "0": "o",
    "1": "l",
    "3": "e",
    "$": "s",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "9": "g",
    "|": "l",
    "!": "i",
    "(": "c",
    ")": "d",
    "{": "c",
    "}": "c",
    "[": "c",
    "]": "c",
    "+": "t",
    "²": "2",
    "6": "b",
    "&": "and",
    "¥": "y",
    "€": "e",
    "#": "h",
    "%": "x",
    "^": "v",
    "<": "c",
    ">": "r",
    "÷": "/",
    "×": "x"
})
    normalized = password.lower().translate(subs)
    for word in dictionary_list:
        if word in normalized:
            return True
    return False

common_password = pd.read_csv('/kaggle/input/common-password-list-rockyoutxt/rockyou.txt' ,encoding='latin-1',  on_bad_lines = 'skip')

common_password[:10000].to_json('/kaggle/working/top10000password.json')

common_password[:10000]

common_password_df = pd.read_json('/kaggle/input/top-10000-common-password-form-rockyou-txt/top10000password (2).json')

common_password_list = common_password_df.values.tolist()

common_password_list = common_password_df[123456].values.tolist()

def rules6(password: str, common_passwords=None) -> bool:
    if common_passwords is None:
        common_passwords = common_password_list
    return password.lower() in common_passwords

def rules7(password: str, personal_data=None) -> bool:
    if personal_data is None:
        personal_data = {"john", "doe", "2023", "1990"}  # Example placeholders
    lower_pass = password.lower()
    for info in personal_data:
        if info in lower_pass:
            return True
    return False

def rules8(password: str, username: str = "") -> bool:
    return username.lower() in password.lower()

KEYBOARD_LAYOUT = [
    # Row 0
    ['`','1','2','3','4','5','6','7','8','9','0','-','='],
    # Row 1
    ['Q','W','E','R','T','Y','U','I','O','P','[',']','\\'],
    # Row 2
    ['A','S','D','F','G','H','J','K','L',';','\''],
    # Row 3
    ['Z','X','C','V','B','N','M',',','.','/']
]

def build_key_coords(layout=KEYBOARD_LAYOUT):
    key_coords = {}
    for r, row_keys in enumerate(layout):
        for c, key in enumerate(row_keys):
            key_coords[key.upper()] = (r, c)   
            key_coords[key.lower()] = (r, c)    
    return key_coords

KEY_COORDS = build_key_coords()


def build_adjacency_map(key_coords):
    adjacency_map = {}
    all_keys = list(key_coords.keys())
    
    for k in all_keys:
        (r1, c1) = key_coords[k]
        neighbors = set()
        for k2 in all_keys:
            if k2 == k:
                continue
            (r2, c2) = key_coords[k2]
            dist = math.dist((r1, c1), (r2, c2))
            if dist < math.sqrt(2):
                neighbors.add(k2)
        adjacency_map[k] = neighbors
    
    return adjacency_map

ADJACENCY_MAP = build_adjacency_map(KEY_COORDS)

def slope_between(k1, k2, key_coords):
    (r1, c1) = key_coords[k1]
    (r2, c2) = key_coords[k2]
    dr = r2 - r1
    dc = c2 - c1
    if dr == 0 and dc == 0:
        return (0, 0)
    # reduce to gcd
    g = math.gcd(dr, dc)
    dr //= g
    dc //= g
    return (dr, dc)

def is_single_line_parallel(password, key_coords):
    if len(password) < 2:
        return True 
    first, second = password[0], password[1]
    if first not in key_coords or second not in key_coords:
        return False    
    base_slope = slope_between(first, second, key_coords)
    for i in range(len(password) - 1):
        c1, c2 = password[i], password[i+1]
        if c1 not in key_coords or c2 not in key_coords:
            return False
        if slope_between(c1, c2, key_coords) != base_slope:
            return False
    return True

def is_two_line_parallel(password, key_coords):
    L = len(password)
    if L % 2 != 0:
        return False  
    
    mid = L // 2
    line1 = password[:mid]
    line2 = password[mid:]
    

    if not is_single_line_parallel(line1, key_coords):
        return False
    if not is_single_line_parallel(line2, key_coords):
        return False
    

    if len(line1) < 2:

        return True
    
    slope1 = slope_between(line1[0], line1[1], key_coords)
    slope2 = slope_between(line2[0], line2[1], key_coords)
    return (slope1 == slope2)

def rules9(password: str) -> bool:
    password = password.strip()
    if len(password) < 2:
        return False  # trivial short password won't be considered an AP pattern
    
    # 1) Check adjacency
    #    if every consecutive pair is in adjacency map
    all_adj = True
    for i in range(len(password) - 1):
        c1, c2 = password[i], password[i+1]
        if (c1 not in ADJACENCY_MAP) or (c2 not in ADJACENCY_MAP[c1]):
            all_adj = False
            break
    if all_adj:
        return True
    
    # 2) Check single-line parallel
    if is_single_line_parallel(password, KEY_COORDS):
        return True
    if is_two_line_parallel(password, KEY_COORDS):
        return True
    
    return False

with open("/kaggle/input/keyboard-patterns-new/data (1).json", "r") as file:
    keyboard_pattern_list = json.load(file)

def has_common_substring_of_length_n_or_more(s1: str, s2: str, n: int = 4) -> bool:
    """
    Returns True if s1 and s2 share any substring of length >= n.
    Otherwise False.
    
    Example:
      - s1 = "tgyh"
      - s2 = "tgyhuj"
      => They share "tgyh" (length 4) => returns True
      - s1 = "tgy"
      - s2 = "tgyhuj"
      => Longest common substring is "tgy" (length 3) => returns False
    """
    s1, s2 = s1.lower(), s2.lower()
    len1, len2 = len(s1), len(s2)
    
    # If either string is shorter than n, they can't have a substring of length >= n in common
    if len1 < n or len2 < n:
        return False

    # Naive approach: check all substring lengths from n up to min(len1, len2).
    # Return True as soon as we find a match.
    max_possible = min(len1, len2)
    for length in range(n, max_possible + 1):
        # Check every substring of s1 with this length
        for start in range(len1 - length + 1):
            sub = s1[start:start + length]
            if sub in s2:
                return True
    return False

def rules10(password: str, keyboard_patterns=None, min_length=4) -> bool:
    """
    Rule 10 (Enhanced):
    Returns True if 'password' shares a common substring of length >= min_length
    with any entry in 'keyboard_patterns'.
    
    Example:
      - keyboard_patterns = ["tgyhuj", "qwerty", ...]
      - password = "tgyh" => shares "tgyh" with "tgyhuj" => True
      - password = "tgy"  => shares "tgy" (length 3) => not >= 4 => False
    """
    if keyboard_patterns is None:
        keyboard_patterns = ["qwerty", "asdf", "zxcv", "tgyhuj"]

    for pattern in keyboard_patterns:
        if has_common_substring_of_length_n_or_more(password, pattern, n=min_length):
            return True
    return False


def rules11(password: str, dictionary_list=None) -> bool:
    """
    Rule 14: Reversed Dictionary Words
    Returns True if password does NOT contain reversed dictionary words.
    """
    if dictionary_list is None:
        dictionary_list = common_words
    lower_pass = password.lower()
    for word in dictionary_list:
        if word[::-1] in lower_pass:
            return True
    return False

Rules12 : data recognition

def rules12(password: str) -> bool:
    """
    Rule 15: Year/Date Patterns
    Returns True if password does NOT contain typical 4-digit year patterns (e.g., 1990-2025).
    (Simple placeholder check.)
    """
    for year in range(1900, 2030):
        if str(year) in password:
            return True
    return False

def rules13(password: str) -> bool:

    half_len = len(password) // 2
    # Check if the first half is repeated in the second half
    if len(password) % 2 == 0:  # even length
        if password[:half_len] == password[half_len:]:
            return True
    return False

def rules14(password: str, entropy_threshold: float = 3.0) -> bool:
    """
    Rule 23: Entropy Estimation
    Returns True if estimated Shannon entropy >= entropy_threshold.
    (Simple placeholder calculation.)
    """
    if not password:
        return False
    freq = {}
    for char in password:
        freq[char] = freq.get(char, 0) + 1
    entropy = 0.0
    length = len(password)
    for count in freq.values():
        p = count / length
        entropy -= p * math.log2(p)
    return entropy <= entropy_threshold

