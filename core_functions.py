import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import requests
from bs4 import BeautifulSoup
import time
import re
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
import random
import traceback

# Add Selenium imports
from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Add a logging function to display detailed information in the terminal
def log_to_terminal(message, log_type="INFO", delay=0.5):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{log_type}] {message}")
    time.sleep(delay)  # Add delay after each log message

# Initialize Firebase Admin
def init_firebase():
    try:
        print("\n=== Initializing Firebase ===")
        time.sleep(1)  # Delay before starting
        
        log_to_terminal("Loading credentials...", "SETUP")
        cred = credentials.Certificate('special-issues-project-firebase-adminsdk-fbsvc-0b2060ce4b.json')
        
        log_to_terminal("Initializing Firebase app...", "SETUP")
        firebase_admin.initialize_app(cred)
        
        log_to_terminal("Firebase initialized successfully!", "SUCCESS", 1)  # Longer delay for success message
        return firestore.client()
    except Exception as e:
        log_to_terminal(f"Error initializing Firebase: {e}", "ERROR", 1)
        return None

# Database functions
def store_special_issue(title, url, keywords, db):
    special_issues_ref = db.collection('special_issues')
    
    try:
        # Clean up data before storing
        if not title or not url:
            log_to_terminal("Missing title or URL, skipping...", "WARNING")
            return False
            
        # Filter out empty or None keywords
        clean_keywords = [k for k in keywords if k]
            
        # Check if document with this URL already exists
        existing_docs = special_issues_ref.where('url', '==', url).limit(1).stream()
        exists = False
        for _ in existing_docs:
            exists = True
            break
            
        if exists:
            log_to_terminal(f"Issue already exists: {title}", "INFO")
            return False
        
        # Create a new document with an auto-generated ID
        doc_data = {
            'title': title,
            'url': url,
            'keywords': clean_keywords,
            'date_added': firestore.SERVER_TIMESTAMP
        }
        
        # Log detailed information about the issue being added
        log_to_terminal(f"Adding new issue to Firestore: {title}", "INFO")
        log_to_terminal(f"URL: {url}", "DEBUG")
        log_to_terminal(f"Keywords ({len(clean_keywords)}): {', '.join(clean_keywords[:5])}{', ...' if len(clean_keywords) > 5 else ''}", "DEBUG")
        
        special_issues_ref.add(doc_data)
        return True
    except Exception as e:
        log_to_terminal(f"Error storing special issue: {e}", "ERROR")
        return False

def load_data(db):
    try:
        print("\n=== Loading Data from Firestore ===")
        time.sleep(1)  # Delay before starting
        
        log_to_terminal("Connecting to Firestore...", "DATA")
        issues_ref = db.collection('special_issues').stream()
        
        special_issues = []
        count = 0
        
        log_to_terminal("Starting to fetch special issues...", "DATA")
        for doc in issues_ref:
            count += 1
            if count % 50 == 0:  # Show progress every 50 items
                log_to_terminal(f"Fetched {count} issues so far...", "PROGRESS", 0.2)
            
            doc_data = doc.to_dict()
            special_issues.append({
                'title': doc_data.get('title', ''),
                'url': doc_data.get('url', ''),
                'keywords': doc_data.get('keywords', [])
            })
        
        df = pd.DataFrame(special_issues)
        log_to_terminal(f"Successfully loaded {len(df)} special issues from Firestore!", "SUCCESS", 1)
        
        # Display a sample of loaded issues
        print("\n=== Sample of Loaded Special Issues ===")
        time.sleep(1)
        
        # Get 5 random issues to show as examples
        sample_size = min(5, len(df))
        sample_indices = np.random.choice(len(df), sample_size, replace=False)
        sample_issues = df.iloc[sample_indices]
        
        for idx, (_, issue) in enumerate(sample_issues.iterrows(), 1):
            print(f"\nIssue {idx}:")
            print(f"Title: {issue['title']}")
            print(f"URL: {issue['url']}")
            print(f"Keywords: {', '.join(issue['keywords'][:5])}")
            if len(issue['keywords']) > 5:
                print(f"... and {len(issue['keywords']) - 5} more keywords")
            time.sleep(0.5)  # Delay between each issue
        
        print("\n" + "="*50)  # Visual separator
        return df
    except Exception as e:
        log_to_terminal(f"Error loading data: {e}", "ERROR", 1)
        return pd.DataFrame(columns=['title', 'url', 'keywords'])

# Recommendation functions
def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def preprocess_keywords(keywords_list):
    return [keyword.lower().strip() for keyword in keywords_list if isinstance(keyword, str)]

def recommend_special_issues_jaccard(user_interests, special_issues_df, top_n=5):
    print("\n=== Generating Jaccard Similarity Recommendations ===")
    time.sleep(1)  # Delay before starting
    
    user_interests_set = set(preprocess_keywords(user_interests))
    log_to_terminal(f"Processing user interests: {', '.join(user_interests)}", "PROCESS")
    
    similarities = []
    total_issues = len(special_issues_df)
    
    log_to_terminal(f"Calculating Jaccard similarity for {total_issues} special issues...", "PROCESS")
    for idx, row in special_issues_df.iterrows():
        if (idx + 1) % 50 == 0:  # Show progress every 50 items
            log_to_terminal(f"Processed {idx + 1}/{total_issues} issues...", "PROGRESS", 0.1)
            
        keywords_set = set(preprocess_keywords(row['keywords']))
        similarity = jaccard_similarity(user_interests_set, keywords_set)
        similarities.append({
            'title': row['title'],
            'url': row['url'],
            'keywords': row['keywords'],
            'similarity_score': similarity,
        })
    
    log_to_terminal("Sorting results by similarity score...", "PROCESS")
    recommendations_df = pd.DataFrame(similarities)
    return recommendations_df.sort_values(by='similarity_score', ascending=False).head(top_n)

def recommend_special_issues_cosine(user_interests, special_issues_df, top_n=5):
    print("\n=== Generating Cosine Similarity Recommendations ===")
    time.sleep(1)  # Delay before starting
    
    log_to_terminal(f"Processing user interests: {', '.join(user_interests)}", "PROCESS")
    
    df = special_issues_df.copy()
    log_to_terminal("Preprocessing keywords...", "PROCESS")
    df['keywords_text'] = df['keywords'].apply(lambda x: ' '.join(preprocess_keywords(x)))
    user_interests_text = ' '.join(preprocess_keywords(user_interests))
    
    log_to_terminal("Creating document vectors...", "PROCESS")
    corpus = [user_interests_text] + df['keywords_text'].tolist()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    
    log_to_terminal("Calculating cosine similarities...", "PROCESS")
    similarities = cosine_similarity(X[0:1], X[1:]).flatten()
    
    log_to_terminal("Sorting results by similarity score...", "PROCESS")
    df['similarity_score'] = similarities
    return df[['title', 'url', 'keywords', 'similarity_score']].sort_values(by='similarity_score', ascending=False).head(top_n)

def display_recommendations(recommendations, method_name):
    print(f"\n=== Top {len(recommendations)} Recommendations using {method_name} ===")
    time.sleep(1)  # Delay before showing results
    
    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
        print(f"\n{idx}. {row['title']}")
        time.sleep(0.5)  # Delay between each recommendation
        print(f"   Score: {row['similarity_score']:.2%}")
        print(f"   URL: {row['url']}")
        print(f"   Keywords: {', '.join(row['keywords'][:5])}")
        if len(row['keywords']) > 5:
            print(f"   ... and {len(row['keywords']) - 5} more keywords")
        time.sleep(0.5)  # Additional delay after each recommendation

def get_valid_input(prompt, validation_func, error_message, default_value=None):
    while True:
        user_input = input(prompt).strip()
        if not user_input and default_value is not None:
            return default_value
        if validation_func(user_input):
            return user_input
        print(error_message)

def validate_yes_no(input_str):
    return input_str.lower() in ['y', 'n', 'yes', 'no']

def validate_number(input_str, min_val=1, max_val=None):
    try:
        num = int(input_str)
        if min_val is not None and num < min_val:
            return False
        if max_val is not None and num > max_val:
            return False
        return True
    except ValueError:
        return False

def validate_keywords(input_str):
    return bool(input_str.strip())

def validate_algorithm(input_str):
    return input_str.lower() in ['1', '2', 'jaccard', 'cosine']

def main():
    print("\n=== Special Issues Recommendation System ===")
    print("==========================================")
    time.sleep(1)
    
    # Initialize Firebase
    db = init_firebase()
    if not db:
        log_to_terminal("Failed to initialize Firebase. Exiting...", "ERROR", 1)
        return
    
    # Ask if user wants to scrape new data
    print("\nWould you like to scrape new special issues?")
    print("This will update the database with the latest special issues.")
    print("Note: This process may take some time.")
    scrape_choice = get_valid_input(
        "Enter 'y' to scrape or 'n' to use existing data (y/n): ",
        validate_yes_no,
        "Please enter 'y' or 'n'",
        "n"
    )
    
    if scrape_choice.lower() in ['y', 'yes']:
        print("\n=== Scraping Configuration ===")
        print("1. The scraper will start from where it left off by default")
        print("2. You can force it to start from the beginning")
        print("3. You can limit the number of pages to scrape")
        print("4. You can choose to scrape all available pages")
        
        force_restart = get_valid_input(
            "Force restart from page 1? (y/n): ",
            validate_yes_no,
            "Please enter 'y' or 'n'",
            "n"
        )
        
        all_pages = get_valid_input(
            "Scrape all available pages? (y/n): ",
            validate_yes_no,
            "Please enter 'y' or 'n'",
            "n"
        )
        
        max_pages = 9999 if all_pages.lower() in ['y', 'yes'] else get_valid_input(
            "Enter number of pages to scrape (1-50): ",
            lambda x: validate_number(x, 1, 50),
            "Please enter a number between 1 and 50",
            "3"
        )
        
        # Start scraping process
        print("\nStarting scraping process...")
        # Add your scraping logic here
        # For now, we'll just load existing data
        df = load_data(db)
    else:
        print("\nLoading existing data...")
        df = load_data(db)
    
    if len(df) > 0:
        print("\n=== Enter Your Research Interests ===")
        print("1. Enter keywords separated by commas")
        print("2. Each keyword should be relevant to your research area")
        print("3. Example: machine learning, artificial intelligence, deep learning")
        
        keywords_input = get_valid_input(
            "Enter your research interests: ",
            validate_keywords,
            "Please enter at least one keyword"
        )
        user_interests = [k.strip() for k in keywords_input.split(',')]
        
        print("\n=== Choose Recommendation Algorithm ===")
        print("1. Jaccard Similarity - Good for exact keyword matches")
        print("2. Cosine Similarity - Better for semantic similarity")
        
        algorithm_choice = get_valid_input(
            "Enter algorithm number (1 or 2): ",
            validate_algorithm,
            "Please enter 1 or 2",
            "1"
        )
        
        print("\n=== Number of Recommendations ===")
        print("Enter how many recommendations you want to see")
        print("Default is 5 recommendations")
        
        top_n = int(get_valid_input(
            "Enter number of recommendations (1-20): ",
            lambda x: validate_number(x, 1, 20),
            "Please enter a number between 1 and 20",
            "5"
        ))
        
        # Generate recommendations
        if algorithm_choice in ['1', 'jaccard']:
            recommendations = recommend_special_issues_jaccard(user_interests, df, top_n)
            method_name = "Jaccard Similarity"
        else:
            recommendations = recommend_special_issues_cosine(user_interests, df, top_n)
            method_name = "Cosine Similarity"
        
        # Display recommendations
        print(f"\n=== Top {top_n} Recommendations using {method_name} ===")
        for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
            print(f"\n{idx}. {row['title']}")
            print(f"   Score: {row['similarity_score']:.2%}")
            print(f"   URL: {row['url']}")
            print(f"   Keywords: {', '.join(row['keywords'][:5])}")
            if len(row['keywords']) > 5:
                print(f"   ... and {len(row['keywords']) - 5} more keywords")
            time.sleep(0.5)
        
        print("\n=== Recommendation Process Completed ===")
        time.sleep(1)
    else:
        log_to_terminal("No data available for recommendations", "WARNING", 1)

if __name__ == "__main__":
    main() 