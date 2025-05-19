from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import sqlite3
import json
import threading
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
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Add a logging function to display detailed information in the terminal
def log_to_terminal(message, log_type="INFO"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{log_type}] {message}")

app = Flask(__name__)
app.secret_key = 'special_issues_recommender_key'

# Database setup
def setup_database():
    # No schema setup needed for Firestore
    # Collections and documents are created on demand
    pass

# Initialize Firebase Admin
cred = credentials.Certificate('special-issues-project-firebase-adminsdk-fbsvc-0b2060ce4b.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Call setup_database once
setup_database()


def get_last_scraped_page():
    try:
        # First try to get from the meta document (faster)
        last_scrape_ref = db.collection('meta').document('last_scrape')
        last_scrape = last_scrape_ref.get()
        
        if last_scrape.exists:
            last_scrape_data = last_scrape.to_dict()
            return last_scrape_data.get('last_page_scraped', 0)
        
        # If meta document doesn't exist, check the logs
        scrape_logs = db.collection('scrape_log').order_by('date_scraped', direction=firestore.Query.DESCENDING).limit(1).stream()
        for log in scrape_logs:
            log_data = log.to_dict()
            return log_data.get('last_page_scraped', 0)
        
        return 0
    except Exception as e:
        log_to_terminal(f"Error getting last scraped page: {e}", "ERROR")
        return 0


def store_special_issue(title, url, keywords):
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


def log_scrape_session(pages_scraped, issues_added, last_page_scraped):
    try:
        # Get current timestamp
        timestamp = firestore.SERVER_TIMESTAMP
        
        # Store detailed scraping log
        db.collection('scrape_log').add({
            'date_scraped': timestamp,
            'pages_scraped': pages_scraped,
            'issues_added': issues_added,
            'last_page_scraped': last_page_scraped,
            'scraping_method': 'edge_webdriver',  # Track which method was used
            'status': 'completed'
        })
        
        # Update the last scrape record for quick access
        last_scrape_ref = db.collection('meta').document('last_scrape')
        last_scrape_ref.set({
            'date_scraped': timestamp,
            'pages_scraped': pages_scraped,
            'issues_added': issues_added,
            'last_page_scraped': last_page_scraped
        }, merge=True)
        
        log_to_terminal(f"Scraping session completed: {pages_scraped} pages processed, {issues_added} new issues added, last page: {last_page_scraped}", "SUCCESS")
    except Exception as e:
        log_to_terminal(f"Error logging scrape session: {e}", "ERROR")


# Global flag to control scraping cancellation
scrape_status = {
    "is_running": False,
    "message": "",
    "timestamp": "",
    "should_cancel": False,
    "current_thread": None
}


def update_status(message):
    scrape_status["message"] = message
    scrape_status["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_terminal(f"Status update: {message}", "STATUS")


def extract_special_issues(start_page=1, max_pages=3, issues_per_page=None, status_callback=None):
    global scrape_status
    base_url = "https://www.mdpi.com/journal/applsci/special_issues"
    
    # Create a session for cookies and state
    session = requests.Session()
    
    # More realistic browser headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Cache-Control": "max-age=0",
        "Sec-Ch-Ua": "\"Google Chrome\";v=\"123\", \"Not:A-Brand\";v=\"8\"",
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": "\"Windows\"",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "Referer": "https://www.mdpi.com/"
    }
    
    # Add random delay function to look more human
    def random_delay():
        time.sleep(random.uniform(1.5, 3.5))
    
    stats = {
        "pages_scraped": 0,
        "issues_found": 0,
        "issues_added": 0,
        "last_page_scraped": start_page - 1
    }
    
    try:
        # Initial request to set cookies and get total pages
        if status_callback:
            status_callback(f"Initializing scraping from page {start_page}...")
        
        log_to_terminal(f"Starting scrape process from page {start_page}", "SCRAPE")
        initial_response = session.get(base_url, headers=headers)
        
        if initial_response.status_code != 200:
            error_msg = f"Failed to access the website. Status code: {initial_response.status_code}"
            log_to_terminal(error_msg, "ERROR")
            if status_callback:
                status_callback(error_msg)
            return stats
        
        # Parse total pages
        soup = BeautifulSoup(initial_response.text, 'html.parser')
        total_pages = 30  # Default fallback
        
        pagination = soup.find('div', class_='pagination')
        if pagination:
            page_links = pagination.find_all('a')
            page_numbers = [int(link.text) for link in page_links if link.text.strip().isdigit()]
            if page_numbers:
                total_pages = max(page_numbers)
        
        end_page = min(total_pages, start_page + max_pages - 1)
        
        log_to_terminal(f"Found {total_pages} total pages. Will scrape from page {start_page} to {end_page}", "SCRAPE")
        if status_callback:
            status_callback(f"Found {total_pages} total pages. Will scrape from page {start_page} to {end_page}.")
        
        # Process each page
        for page_no in range(start_page, end_page + 1):
            if scrape_status["should_cancel"]:
                break
            
            if status_callback:
                status_callback(f"Scraping page {page_no}/{end_page}...")
            
            # Add random delay between page requests to look human
            if page_no > start_page:
                random_delay()
            
            page_url = base_url if page_no == 1 else f"{base_url}?page_no={page_no}"
            
            # Rotate user agent slightly for even more human-like behavior
            if page_no % 2 == 0:
                headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0"
            
            page_response = session.get(page_url, headers=headers)
            
            if page_response.status_code != 200:
                if status_callback:
                    status_callback(f"Failed to get page {page_no}: {page_response.status_code}")
                continue
            
            page_soup = BeautifulSoup(page_response.text, 'html.parser')
            
            # Find all special issues using the specific HTML structure
            special_issue_items = page_soup.find_all('div', class_='generic-item article-item')
            
            if status_callback:
                status_callback(f"Found {len(special_issue_items)} special issues on page {page_no}")
            
            page_issues_added = 0
            
            for item in special_issue_items:
                if scrape_status["should_cancel"]:
                    break
                
                # Extract title and URL
                title_link = item.select_one('a.title-link')
                if not title_link:
                    continue
                
                title = title_link.text.strip()
                url = "https://www.mdpi.com" + title_link['href'] if not title_link['href'].startswith('http') else title_link['href']
                
                # Extract keywords directly from this item if available
                keywords = []
                keywords_div = item.select_one('div:-soup-contains("Keywords:")')
                if keywords_div:
                    keywords_text = keywords_div.text.replace('Keywords:', '').strip()
                    keywords = [k.strip() for k in keywords_text.split(';') if k.strip()]
                
                if status_callback:
                    status_callback(f"Processing: {title[:50]}...")
                
                # Check if we found keywords directly
                if keywords:
                    added = store_special_issue(title, url, keywords)
                    if added:
                        page_issues_added += 1
                        stats["issues_added"] += 1
                        if status_callback:
                            status_callback(f"Added issue: {title[:40]}... with {len(keywords)} keywords")
                        log_to_terminal(f"Added: {title[:60]}{' ...' if len(title) > 60 else ''}", "SCRAPE")
                else:
                    # Need to visit the special issue page to get keywords
                    # Add a random delay to appear more human-like
                    random_delay()
                    
                    try:
                        # Slight variation in headers for each request
                        issue_headers = headers.copy()
                        issue_headers["Referer"] = page_url
                        
                        issue_response = session.get(url, headers=issue_headers)
                        
                        if issue_response.status_code == 200:
                            issue_soup = BeautifulSoup(issue_response.text, 'html.parser')
                            
                            # Look for keywords section
                            keywords_section = issue_soup.select_one('div:-soup-contains("Keywords:")')
                            if keywords_section:
                                keywords_text = keywords_section.text.replace('Keywords:', '').strip()
                                keywords = [k.strip() for k in keywords_text.split(';') if k.strip()]
                                log_to_terminal(f"Found {len(keywords)} keywords using primary method", "DETAIL")
                            
                            # If still no keywords, try alternative methods
                            if not keywords:
                                for tag in ['h2', 'h3', 'h4', 'strong']:
                                    headers = issue_soup.find_all(tag)
                                    for header in headers:
                                        if "keyword" in header.text.lower():
                                            next_text = header.find_next_sibling()
                                            if next_text:
                                                text_content = next_text.text.strip()
                                                keywords = [k.strip() for k in text_content.split(';') if k.strip()]
                                                if keywords:
                                                    log_to_terminal(f"Found {len(keywords)} keywords using alternative method", "DETAIL")
                                                    break
                            
                            # Store the issue
                            added = store_special_issue(title, url, keywords)
                            if added:
                                page_issues_added += 1
                                stats["issues_added"] += 1
                                if status_callback:
                                    status_callback(f"Added issue: {title[:40]}... with {len(keywords)} keywords")
                                log_to_terminal(f"Added: {title[:60]}{' ...' if len(title) > 60 else ''}", "SCRAPE")
                            else:
                                log_to_terminal(f"Skipped (already exists): {title[:60]}{' ...' if len(title) > 60 else ''}", "DETAIL")
                    except Exception as e:
                        print(f"Error processing issue {title}: {e}")
                        if status_callback:
                            status_callback(f"Error processing issue: {title[:40]}...")
            
            if status_callback:
                status_callback(f"Page {page_no} complete. Added {page_issues_added} new issues.")
            
            stats["issues_found"] += len(special_issue_items)
            stats["pages_scraped"] += 1
            stats["last_page_scraped"] = page_no
    
    except Exception as e:
        print(f"Error during scraping: {e}")
        if status_callback:
            status_callback(f"Error during scraping: {str(e)}")
    
    if status_callback:
        status_callback(f"Scraping completed. Added {stats['issues_added']} new issues.")
    
    log_scrape_session(stats["pages_scraped"], stats["issues_added"], stats["last_page_scraped"])
    return stats


def load_data():
    try:
        special_issues = []
        issues_ref = db.collection('special_issues').stream()
        
        for doc in issues_ref:
            doc_data = doc.to_dict()
            special_issues.append({
                'title': doc_data.get('title', ''),
                'url': doc_data.get('url', ''),
                'keywords': doc_data.get('keywords', [])
            })
        
        df = pd.DataFrame(special_issues)
        log_to_terminal(f"Loaded {len(df)} special issues from Firestore", "DATA")
        return df
    except Exception as e:
        log_to_terminal(f"Error loading data: {e}", "ERROR")
        return pd.DataFrame(columns=['title', 'url', 'keywords'])


# Recommendation functions (unchanged)
def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0


def preprocess_keywords(keywords_list):
    return [keyword.lower().strip() for keyword in keywords_list if isinstance(keyword, str)]


def recommend_special_issues_jaccard(user_interests, special_issues_df, top_n=5):
    user_interests_set = set(preprocess_keywords(user_interests))
    similarities = []
    for idx, row in special_issues_df.iterrows():
        keywords_set = set(preprocess_keywords(row['keywords']))
        similarity = jaccard_similarity(user_interests_set, keywords_set)
        similarities.append({
            'title': row['title'],
            'url': row['url'],
            'keywords': row['keywords'],
            'similarity_score': similarity,
        })
    recommendations_df = pd.DataFrame(similarities)
    return recommendations_df.sort_values(by='similarity_score', ascending=False).head(top_n)


def recommend_special_issues_cosine(user_interests, special_issues_df, top_n=5):
    df = special_issues_df.copy()
    df['keywords_text'] = df['keywords'].apply(lambda x: ' '.join(preprocess_keywords(x)))
    user_interests_text = ' '.join(preprocess_keywords(user_interests))
    corpus = [user_interests_text] + df['keywords_text'].tolist()
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(X[0:1], X[1:]).flatten()
    df['similarity_score'] = similarities
    return df[['title', 'url', 'keywords', 'similarity_score']].sort_values(by='similarity_score',
                                                                            ascending=False).head(top_n)


def start_scraping_background(start_page=1, max_pages=3):
    global scrape_status
    if scrape_status["is_running"]:
        return {"success": False, "message": "Scraping is already in progress"}

    scrape_status["is_running"] = True
    scrape_status["should_cancel"] = False
    update_status("Starting scraping process...")

    def scrape_thread():
        try:
            stats = extract_special_issues(start_page, max_pages, status_callback=update_status)
            if not scrape_status["should_cancel"]:
                update_status(
                    f"Scraping completed. Processed {stats['pages_scraped']} pages, added {stats['issues_added']} new issues.")
        except Exception as e:
            update_status(f"Error during scraping: {str(e)}")
        finally:
            scrape_status["is_running"] = False
            scrape_status["current_thread"] = None

    thread = threading.Thread(target=scrape_thread)
    thread.daemon = True
    scrape_status["current_thread"] = thread
    thread.start()
    return {"success": True, "message": "Scraping started in background"}


# Routes
@app.route('/')
def index():
    try:
        # Count special issues
        special_issues = list(db.collection('special_issues').limit(1000).stream())
        issue_count = len(special_issues)
        
        # Get last scrape info
        last_scrape_date = "Never"
        last_page = 0
        
        # Try to get from meta document first (faster)
        last_scrape_ref = db.collection('meta').document('last_scrape')
        last_scrape = last_scrape_ref.get()
        
        if last_scrape.exists:
            last_scrape_data = last_scrape.to_dict()
            if 'date_scraped' in last_scrape_data:
                if isinstance(last_scrape_data['date_scraped'], datetime.datetime):
                    last_scrape_date = last_scrape_data['date_scraped'].strftime("%Y-%m-%d %H:%M:%S")
                else:
                    last_scrape_date = str(last_scrape_data['date_scraped'])
            last_page = last_scrape_data.get('last_page_scraped', 0)
        else:
            # Fall back to scrape logs
            scrape_logs = db.collection('scrape_log').order_by('date_scraped', direction=firestore.Query.DESCENDING).limit(1).stream()
            for log in scrape_logs:
                log_data = log.to_dict()
                if 'date_scraped' in log_data:
                    if isinstance(log_data['date_scraped'], datetime.datetime):
                        last_scrape_date = log_data['date_scraped'].strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        last_scrape_date = str(log_data['date_scraped'])
                last_page = log_data.get('last_page_scraped', 0)
                break
        
        return render_template('index.html',
                            issue_count=issue_count,
                            last_scrape_date=last_scrape_date,
                            last_page=last_page,
                            scrape_running=scrape_status["is_running"])
    except Exception as e:
        print(f"Error loading index page: {e}")
        return render_template('index.html',
                            issue_count=0,
                            last_scrape_date="Error",
                            last_page=0,
                            scrape_running=scrape_status["is_running"])


@app.route('/recommend', methods=['POST'])
def recommend():
    interests = request.form.get('interests', '')
    method = request.form.get('method', 'jaccard')
    top_n = int(request.form.get('top_n', '5'))
    interests_list = [interest.strip() for interest in interests.split(',') if interest.strip()]
    
    # Log the keyword search
    log_to_terminal(f"Keyword search initiated with: {', '.join(interests_list)}", "SEARCH")
    log_to_terminal(f"Using {method} method, requesting top {top_n} results", "SEARCH")
    
    special_issues_df = load_data()

    if len(special_issues_df) == 0:
        log_to_terminal("No special issues found in database for search operation", "WARNING")
        flash("No special issues found in database. Please run the scraper first.", "warning")
        return redirect(url_for('index'))

    if method == 'cosine':
        recommendations = recommend_special_issues_cosine(interests_list, special_issues_df, top_n)
        method_name = "Cosine Similarity"
    else:
        recommendations = recommend_special_issues_jaccard(interests_list, special_issues_df, top_n)
        method_name = "Jaccard Similarity"

    results = [{
        'title': row['title'],
        'url': row['url'],
        'keywords': row['keywords'][:5] if len(row['keywords']) > 5 else row['keywords'],
        'additional_keywords': len(row['keywords']) - 5 if len(row['keywords']) > 5 else 0,
        'similarity_score': round(row['similarity_score'] * 100, 2)
    } for _, row in recommendations.iterrows()]
    
    # Log the search results
    log_to_terminal(f"Search results: Found {len(results)} matching special issues", "RESULT")
    for idx, result in enumerate(results[:3], 1):  # Log top 3 results
        log_to_terminal(f"  Top {idx}: {result['title']} (score: {result['similarity_score']}%)", "RESULT")
    if len(results) > 3:
        log_to_terminal(f"  ... and {len(results) - 3} more matches", "RESULT")

    return render_template('results.html',
                           results=results,
                           interests=interests_list,
                           method=method_name,
                           top_n=top_n)


@app.route('/scrape', methods=['POST'])
def scrape():
    try:
        print("=== Starting scrape process ===")
        print("Received POST request to /scrape")
        
        # Get the last scraped page
        last_page = get_last_scraped_page()
        print(f"Last scraped page: {last_page}")
        
        # Get parameters from request
        max_pages = int(request.form.get('max_pages', 3))
        force_restart = request.form.get('force_restart', 'false').lower() == 'true'
        use_selenium = request.form.get('use_selenium', 'true').lower() == 'true'
        all_pages = request.form.get('all_pages', 'false').lower() == 'true'
        save_debug_files = request.form.get('save_debug_files', 'false').lower() == 'true'
        
        print(f"Scraping parameters:")
        print(f"- max_pages: {max_pages}")
        print(f"- force_restart: {force_restart}")
        print(f"- use_selenium: {use_selenium}")
        print(f"- all_pages: {all_pages}")
        print(f"- save_debug_files: {save_debug_files}")
        
        # Determine start page
        if force_restart:
            start_page = 1
            print("Force restart requested - starting from page 1")
        else:
            start_page = last_page + 1 if last_page > 0 else 1
            
        print(f"Starting scrape from page {start_page}, max pages: {max_pages}, all pages: {all_pages}, save debug files: {save_debug_files}")
        
        # Start the scraping process - using Edge WebDriver
        if use_selenium:
            print("Using Edge WebDriver for scraping")
            # We'll run this in a background thread since it can take time
            def edge_scrape_thread():
                global scrape_status
                try:
                    scrape_status["is_running"] = True
                    scrape_status["should_cancel"] = False
                    update_status(f"Starting Edge WebDriver scraping from page {start_page}...")
                    print(f"Starting Edge WebDriver scraping from page {start_page}...")
                    
                    # Set up Edge WebDriver
                    driver = None
                    try:
                        print("Setting up Edge WebDriver...")
                        # Set up Edge options
                        edge_options = EdgeOptions()
                        edge_options.add_argument("--headless")
                        edge_options.add_argument("--no-sandbox") 
                        edge_options.add_argument("--disable-dev-shm-usage")
                        edge_options.add_argument("--disable-gpu")
                        edge_options.add_argument("--window-size=1920,1080")
                        edge_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0")
                        
                        # Set up driver with explicit path to msedgedriver.exe
                        driver_path = os.path.join(os.getcwd(), "msedgedriver.exe")
                        print(f"Using Edge driver at: {driver_path}")
                        service = EdgeService(executable_path=driver_path)
                        driver = webdriver.Edge(service=service, options=edge_options)
                        print("Edge WebDriver initialized successfully")
                        
                        current_page = int(start_page)
                        end_page = 9999 if all_pages else (current_page + max_pages - 1)
                        
                        # If scraping all pages, determine the total number of pages
                        if all_pages:
                            try:
                                print("Determining total number of pages...")
                                # Navigate to first page
                                driver.get("https://www.mdpi.com/journal/applsci/special_issues")
                                
                                # Wait for the page to load
                                WebDriverWait(driver, 15).until(
                                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.pagination"))
                                )
                                
                                # Get the maximum page number
                                pagination = driver.find_element(By.CSS_SELECTOR, "div.pagination")
                                page_links = pagination.find_elements(By.TAG_NAME, "a")
                                page_numbers = [int(link.text) for link in page_links if link.text.strip().isdigit()]
                                
                                if page_numbers:
                                    max_page = max(page_numbers)
                                    end_page = max_page
                                    print(f"Found {max_page} total pages. Will scrape from page {current_page} to {end_page}.")
                                    update_status(f"Found {max_page} total pages. Will scrape from page {current_page} to {end_page}.")
                            except Exception as e:
                                print(f"Error determining total pages: {str(e)}")
                                update_status(f"Error determining total pages: {str(e)}. Will use default max pages.")
                                end_page = current_page + 10  # Default to 10 pages if we can't determine the total
                        
                        pages_processed = 0
                        issues_found = 0
                        issues_added = 0
                        
                        while current_page <= end_page and not scrape_status["should_cancel"]:
                            print(f"\nProcessing page {current_page} of {end_page}...")
                            update_status(f"Scraping page {current_page} of {end_page}...")
                            
                            # Construct URL
                            url = f"https://www.mdpi.com/journal/applsci/special_issues?page_no={current_page}" if current_page > 1 else "https://www.mdpi.com/journal/applsci/special_issues"
                            print(f"Navigating to URL: {url}")
                            
                            # Navigate to page
                            driver.get(url)
                            
                            # Wait for content to load
                            try:
                                print("Waiting for page content to load...")
                                WebDriverWait(driver, 15).until(
                                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.generic-item.article-item"))
                                )
                                print("Page content loaded successfully")
                            except Exception as e:
                                print(f"Timeout waiting for page content on page {current_page}: {str(e)}")
                                update_status(f"Timeout waiting for page content on page {current_page}: {str(e)}")
                                # Check if we've reached the last page
                                try:
                                    next_button = driver.find_elements(By.XPATH, "//a[contains(text(), 'next')]")
                                    if not next_button:
                                        print(f"Reached the last page ({current_page-1}). No more pages available.")
                                        update_status(f"Reached the last page ({current_page-1}). No more pages available.")
                                        break
                                except:
                                    pass
                                break
                            
                            # Save page source for debugging (only if enabled)
                            if save_debug_files:
                                print(f"Saving page source to scrape_page_{current_page}.html")
                                with open(f"scrape_page_{current_page}.html", "w", encoding="utf-8") as f:
                                    f.write(driver.page_source)
                            
                            # Find all special issue elements
                            issue_elements = driver.find_elements(By.CSS_SELECTOR, "div.generic-item.article-item")
                            issues_found += len(issue_elements)
                            print(f"Found {len(issue_elements)} special issues on page {current_page}")
                            update_status(f"Found {len(issue_elements)} special issues on page {current_page}")
                            
                            page_issues_added = 0
                            
                            # Process each special issue
                            for idx, element in enumerate(issue_elements):
                                if scrape_status["should_cancel"]:
                                    print("Scraping cancelled by user")
                                    break
                                
                                try:
                                    # Extract title and URL
                                    title_link = element.find_element(By.CSS_SELECTOR, "a.title-link")
                                    title = title_link.text.strip()
                                    url = title_link.get_attribute("href")
                                    
                                    print(f"Processing issue {idx+1}/{len(issue_elements)}: {title[:40]}...")
                                    update_status(f"Processing issue {idx+1}/{len(issue_elements)} on page {current_page}: {title[:40]}...")
                                    
                                    # Extract keywords from the listing if available
                                    keywords = []
                                    try:
                                        keywords_div = element.find_elements(By.XPATH, ".//div[contains(text(), 'Keywords:')]")
                                        if keywords_div:
                                            keywords_text = keywords_div[0].text.replace("Keywords:", "").strip()
                                            keywords = [k.strip() for k in keywords_text.split(';') if k.strip()]
                                    except Exception as e:
                                        print(f"Error extracting keywords from listing: {e}")
                                    
                                    # If no keywords in listing, visit the issue page to get them
                                    if not keywords:
                                        try:
                                            # Add a delay to avoid being blocked
                                            time.sleep(random.uniform(1.0, 2.0))
                                            
                                            # Open the special issue page in a new tab
                                            driver.execute_script("window.open('');")
                                            driver.switch_to.window(driver.window_handles[1])
                                            driver.get(url)
                                            
                                            # Wait for the page to load
                                            WebDriverWait(driver, 15).until(
                                                EC.presence_of_element_located((By.TAG_NAME, "h1"))
                                            )
                                            
                                            # Save the issue page for debugging (only if enabled)
                                            if save_debug_files:
                                                with open(f"edge_issue_page_{current_page}_{idx+1}.html", "w", encoding="utf-8") as f:
                                                    f.write(driver.page_source)
                                            
                                            # Method 1: Look for the keywords section in the specified format
                                            try:
                                                # Find h2 with name="keywords"
                                                keywords_h2 = driver.find_elements(By.XPATH, "//h2[@name='keywords' or contains(text(), 'Keywords')]")
                                                
                                                if keywords_h2:
                                                    # Find the container div that might contain the ul
                                                    content_container = None
                                                    
                                                    # Try different approaches to find the keywords section
                                                    # 1. Direct parent-child relationship
                                                    container_divs = driver.find_elements(
                                                        By.XPATH, 
                                                        "//h2[@name='keywords' or contains(text(), 'Keywords')]/following-sibling::div[1]"
                                                    )
                                                    
                                                    if container_divs:
                                                        content_container = container_divs[0]
                                                    
                                                    # 2. With class name
                                                    if not content_container:
                                                        container_divs = driver.find_elements(
                                                            By.XPATH, 
                                                            "//div[contains(@class, 'content__container')]"
                                                        )
                                                        for div in container_divs:
                                                            prev_h2 = div.find_elements(By.XPATH, "./preceding-sibling::h2[1]")
                                                            if prev_h2 and ("keywords" in prev_h2[0].text.lower() or 
                                                                           prev_h2[0].get_attribute("name") == "keywords"):
                                                                content_container = div
                                                                break
                                                    
                                                    # If we found the container, extract the keywords
                                                    if content_container:
                                                        # Look for list items in the container
                                                        keyword_items = content_container.find_elements(By.CSS_SELECTOR, "ul li")
                                                        if keyword_items:
                                                            keywords = [item.text.strip() for item in keyword_items if item.text.strip()]
                                            except Exception as e:
                                                print(f"Error extracting keywords using method 1: {e}")
                                            
                                            # Method 2: If method 1 fails, try alternative approaches
                                            if not keywords:
                                                try:
                                                    # Look for keywords in any section
                                                    keywords_sections = driver.find_elements(
                                                        By.XPATH, 
                                                        "//*[self::h2 or self::h3 or self::h4][contains(text(), 'Keywords') or contains(text(), 'Key words')]"
                                                    )
                                                    
                                                    if keywords_sections:
                                                        for section in keywords_sections:
                                                            # Look for a list after this heading
                                                            next_list = section.find_elements(By.XPATH, "./following-sibling::ul[1]")
                                                            if next_list:
                                                                keyword_items = next_list[0].find_elements(By.TAG_NAME, "li")
                                                                if keyword_items:
                                                                    keywords = [item.text.strip() for item in keyword_items if item.text.strip()]
                                                                    break
                                                            
                                                            # If no list, look for a div containing text
                                                            next_div = section.find_elements(By.XPATH, "./following-sibling::div[1]")
                                                            if next_div:
                                                                # Check if the div contains a list
                                                                div_list = next_div[0].find_elements(By.TAG_NAME, "ul")
                                                                if div_list:
                                                                    keyword_items = div_list[0].find_elements(By.TAG_NAME, "li")
                                                                    if keyword_items:
                                                                        keywords = [item.text.strip() for item in keyword_items if item.text.strip()]
                                                                        break
                                                                
                                                                # If no list, check if it contains semicolon separated keywords
                                                                div_text = next_div[0].text.strip()
                                                                if div_text and ';' in div_text:
                                                                    keywords = [k.strip() for k in div_text.split(';') if k.strip()]
                                                                    break
                                                                elif div_text and ',' in div_text:
                                                                    keywords = [k.strip() for k in div_text.split(',') if k.strip()]
                                                                    break
                                                except Exception as e:
                                                    print(f"Error extracting keywords using method 2: {e}")
                                            
                                            # Close the special issue tab and switch back to the main tab
                                            driver.close()
                                            driver.switch_to.window(driver.window_handles[0])
                                        
                                        except Exception as e:
                                            print(f"Error visiting special issue page: {e}")
                                            # Make sure we're back to the main tab
                                            if len(driver.window_handles) > 1:
                                                driver.close()
                                                driver.switch_to.window(driver.window_handles[0])
                                    
                                    # Store in database
                                    added = store_special_issue(title, url, keywords)
                                    if added:
                                        page_issues_added += 1
                                        issues_added += 1
                                        update_status(f"Added issue with {len(keywords)} keywords: {title[:40]}...")
                                    else:
                                        update_status(f"Issue already exists: {title[:40]}...")
                                        
                                except Exception as e:
                                    update_status(f"Error processing issue: {str(e)}")
                            
                            update_status(f"Page {current_page} complete. Added {page_issues_added} new issues.")
                            
                            pages_processed += 1
                            
                            # Move to next page
                            current_page += 1
                            
                            # Check if we've reached the end (for all_pages option)
                            if all_pages:
                                try:
                                    next_buttons = driver.find_elements(By.XPATH, "//a[contains(text(), 'next') or contains(@class, 'next')]")
                                    if not next_buttons:
                                        update_status(f"Reached the last page. No more pages available.")
                                        break
                                except Exception:
                                    pass
                            
                            # Add a random delay between pages to be less bot-like
                            time.sleep(random.uniform(2.0, 4.0))
                        
                        # Log the scraping session
                        log_scrape_session(pages_processed, issues_added, current_page - 1)
                        
                        if scrape_status["should_cancel"]:
                            print(f"\nScraping cancelled. Processed {pages_processed} pages, added {issues_added} issues.")
                            update_status(f"Scraping cancelled. Processed {pages_processed} pages, added {issues_added} issues.")
                        else:
                            print(f"\nScraping completed successfully!")
                            print(f"Processed {pages_processed} pages")
                            print(f"Found {issues_found} issues")
                            print(f"Added {issues_added} new issues")
                            update_status(f"Scraping completed. Processed {pages_processed} pages, added {issues_added} issues.")
                    
                    except Exception as e:
                        print(f"\nError during Edge scraping: {str(e)}")
                        print("Traceback:")
                        traceback.print_exc()
                        update_status(f"Error during Edge scraping: {str(e)}")
                    
                    finally:
                        # Make sure to close the driver
                        if driver:
                            try:
                                print("Closing Edge WebDriver...")
                                driver.quit()
                                print("Edge WebDriver closed successfully")
                            except Exception as e:
                                print(f"Error closing Edge WebDriver: {str(e)}")
                
                finally:
                    print("Cleaning up scrape status...")
                    scrape_status["is_running"] = False
                    scrape_status["current_thread"] = None
                    print("Scrape status cleaned up")
            
            # Start the thread
            print("Starting scraping thread...")
            thread = threading.Thread(target=edge_scrape_thread)
            thread.daemon = True
            scrape_status["current_thread"] = thread
            thread.start()
            print("Scraping thread started successfully")
            
            return jsonify({"message": "Edge WebDriver scraping started in background", "success": True})
        else:
            print("Using BeautifulSoup scraping method")
            # Fall back to the original BeautifulSoup scraping method
            result = start_scraping_background(start_page, max_pages)
            return jsonify({"message": result["message"], "success": result["success"]})
    
    except Exception as e:
        print(f"\nError starting scrape: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return jsonify({"message": f"Error starting scrape: {str(e)}", "success": False})


@app.route('/cancel-scrape', methods=['POST'])
def cancel_scrape():
    global scrape_status
    if scrape_status["is_running"]:
        scrape_status["should_cancel"] = True
        update_status("Cancelling scraping process...")
        return jsonify({"success": True, "message": "Scraping cancellation requested"})
    return jsonify({"success": False, "message": "No active scraping to cancel"})


@app.route('/scrape-status')
def scrape_status_route():
    return jsonify({
        "running": scrape_status["is_running"],
        "message": scrape_status["message"],
        "timestamp": scrape_status["timestamp"]
    })


@app.route('/database-stats')
def database_stats():
    try:
        # Count special issues
        special_issues = list(db.collection('special_issues').limit(1000).stream())
        issue_count = len(special_issues)
        
        # Get last scrape info from meta document
        last_scrape_ref = db.collection('meta').document('last_scrape')
        last_scrape = last_scrape_ref.get()
        
        last_scrape_date = None
        pages_scraped = 0
        issues_added = 0
        last_page = 0
        
        if last_scrape.exists:
            last_scrape_data = last_scrape.to_dict()
            if 'date_scraped' in last_scrape_data:
                last_scrape_date = last_scrape_data['date_scraped']
            pages_scraped = last_scrape_data.get('pages_scraped', 0)
            issues_added = last_scrape_data.get('issues_added', 0)
            last_page = last_scrape_data.get('last_page_scraped', 0)
        else:
            # Fall back to scrape logs
            scrape_logs = db.collection('scrape_log').order_by('date_scraped', direction=firestore.Query.DESCENDING).limit(1).stream()
            for log in scrape_logs:
                log_data = log.to_dict()
                if 'date_scraped' in log_data:
                    last_scrape_date = log_data['date_scraped']
                pages_scraped = log_data.get('pages_scraped', 0)
                issues_added = log_data.get('issues_added', 0)
                last_page = log_data.get('last_page_scraped', 0)
                break
        
        # Calculate keyword stats
        with_keywords = 0
        total_keywords = 0
        
        for issue in special_issues:
            issue_data = issue.to_dict()
            keywords = issue_data.get('keywords', [])
            if keywords and len(keywords) > 0:
                with_keywords += 1
                total_keywords += len(keywords)
        
        avg_keywords = round(total_keywords / with_keywords, 1) if with_keywords > 0 else 0
        
        # Format date for JSON response
        date_str = None
        if last_scrape_date:
            if isinstance(last_scrape_date, datetime.datetime):
                date_str = last_scrape_date.strftime("%Y-%m-%d %H:%M:%S")
            else:
                date_str = str(last_scrape_date)
        
        stats = {
            "issue_count": issue_count,
            "last_scrape": {
                "date": date_str,
                "pages_scraped": pages_scraped,
                "issues_added": issues_added,
                "last_page": last_page
            },
            "keyword_stats": {
                "with_keywords": with_keywords,
                "avg_keywords": avg_keywords
            }
        }
        return jsonify(stats)
    except Exception as e:
        print(f"Error getting database stats: {e}")
        return jsonify({
            "error": str(e),
            "issue_count": 0
        }), 500


@app.route('/scrape-history')
def scrape_history():
    """
    Get the history of scraping sessions to track progress over time.
    """
    try:
        # Get the scraping history from logs
        scrape_logs = db.collection('scrape_log').order_by('date_scraped', direction=firestore.Query.DESCENDING).limit(20).stream()
        
        history = []
        for log in scrape_logs:
            log_data = log.to_dict()
            date_str = None
            if 'date_scraped' in log_data:
                if isinstance(log_data['date_scraped'], datetime.datetime):
                    date_str = log_data['date_scraped'].strftime("%Y-%m-%d %H:%M:%S")
                else:
                    date_str = str(log_data['date_scraped'])
                    
            history.append({
                "date": date_str,
                "pages_scraped": log_data.get('pages_scraped', 0),
                "issues_added": log_data.get('issues_added', 0),
                "last_page": log_data.get('last_page_scraped', 0),
                "method": log_data.get('scraping_method', 'unknown'),
                "status": log_data.get('status', 'unknown')
            })
        
        return jsonify({
            "success": True,
            "history": history
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/edge-scrape', methods=['GET'])
def edge_scrape():
    """
    Uses Edge WebDriver to scrape MDPI special issues.
    This uses the local msedgedriver.exe in your project folder.
    Returns JSON data without storing in database.
    """
    driver = None
    try:
        # Get parameters
        page = request.args.get('page', '1')
        save_debug_files = request.args.get('save_debug_files', 'false').lower() == 'true'
        
        # Set up Edge options
        edge_options = EdgeOptions()
        edge_options.add_argument("--headless")
        edge_options.add_argument("--no-sandbox")
        edge_options.add_argument("--disable-dev-shm-usage")
        edge_options.add_argument("--disable-gpu")
        edge_options.add_argument("--window-size=1920,1080")
        
        # Add realistic user agent
        edge_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0")
        
        # Set up the Edge driver with local msedgedriver.exe
        driver_path = os.path.join(os.getcwd(), "msedgedriver.exe")
        service = EdgeService(executable_path=driver_path)
        driver = webdriver.Edge(service=service, options=edge_options)
        
        # Page to scrape - get from query params or use default
        url = f"https://www.mdpi.com/journal/applsci/special_issues?page_no={page}" if page != '1' else "https://www.mdpi.com/journal/applsci/special_issues"
        
        # Navigate to the URL
        driver.get(url)
        
        # Wait for the page to load (wait for special issue items to appear)
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.generic-item.article-item"))
            )
        except Exception as e:
            # Take screenshot if page load times out
            if save_debug_files:
                driver.save_screenshot(f"edge_timeout_screenshot_{page}.png")
                
                # Save page source for analysis
                with open(f"edge_timeout_page_{page}.html", "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
                    
            return jsonify({
                "success": False,
                "error": f"Timeout waiting for page elements: {str(e)}",
                "page_source_saved": f"edge_timeout_page_{page}.html" if save_debug_files else "disabled",
                "screenshot_saved": f"edge_timeout_screenshot_{page}.png" if save_debug_files else "disabled"
            })
        
        # Take a screenshot for debugging
        if save_debug_files:
            driver.save_screenshot(f"edge_screenshot_{page}.png")
        
        # Parse the special issues
        special_issues = []
        
        # Find all special issue elements
        issue_elements = driver.find_elements(By.CSS_SELECTOR, "div.generic-item.article-item")
        
        for idx, element in enumerate(issue_elements):
            issue_data = {}
            
            # Get the title and URL
            try:
                title_link = element.find_element(By.CSS_SELECTOR, "a.title-link")
                issue_data["title"] = title_link.text.strip()
                issue_data["url"] = title_link.get_attribute("href")
                
                # Look for keywords
                try:
                    # Try to find keywords directly in the listing
                    keywords_div = element.find_elements(By.XPATH, ".//div[contains(text(), 'Keywords:')]")
                    if keywords_div:
                        keywords_text = keywords_div[0].text.replace("Keywords:", "").strip()
                        issue_data["keywords"] = [k.strip() for k in keywords_text.split(';') if k.strip()]
                except Exception:
                    # If keywords aren't in the listing, we'd need to visit the issue page
                    issue_data["keywords"] = []
                    
                # Get deadline if available
                try:
                    deadline_elem = element.find_elements(By.XPATH, ".//div[contains(text(), 'deadline')]")
                    if deadline_elem:
                        deadline_text = deadline_elem[0].text
                        deadline_match = re.search(r'deadline\s+(\d+\s+\w+\s+\d+)', deadline_text)
                        if deadline_match:
                            issue_data["deadline"] = deadline_match.group(1)
                except Exception:
                    pass
                
                # Check if we want to get keywords from the issue page
                if not issue_data.get("keywords") and idx < 5:  # Limit to first 5 issues to avoid long response times
                    try:
                        # Open the special issue page in a new tab
                        driver.execute_script("window.open('');")
                        driver.switch_to.window(driver.window_handles[1])
                        driver.get(issue_data["url"])
                        
                        # Wait for the page to load
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.TAG_NAME, "h1"))
                        )
                        
                        # Save issue page HTML only if debug files are enabled
                        if save_debug_files:
                            with open(f"edge_issue_page_{page}_{idx+1}.html", "w", encoding="utf-8") as f:
                                f.write(driver.page_source)
                        
                        # Look for h2 with name="keywords"
                        keywords_h2 = driver.find_elements(By.XPATH, "//h2[@name='keywords' or contains(text(), 'Keywords')]")
                        if keywords_h2:
                            # Find the container div
                            container_divs = driver.find_elements(
                                By.XPATH, 
                                "//h2[@name='keywords' or contains(text(), 'Keywords')]/following-sibling::div[1]"
                            )
                            
                            if container_divs:
                                content_container = container_divs[0]
                                
                                # Look for list items
                                keyword_items = content_container.find_elements(By.CSS_SELECTOR, "ul li")
                                if keyword_items:
                                    issue_data["keywords"] = [item.text.strip() for item in keyword_items if item.text.strip()]
                        
                        # Close the tab and switch back
                        driver.close()
                        driver.switch_to.window(driver.window_handles[0])
                    except Exception as e:
                        print(f"Error getting keywords for issue {idx}: {e}")
                        # Make sure we're back to the main window
                        if len(driver.window_handles) > 1:
                            driver.close()
                            driver.switch_to.window(driver.window_handles[0])
                
                special_issues.append(issue_data)
            except Exception as e:
                print(f"Error extracting issue data: {e}")
        
        # Get pagination information
        pagination_info = {}
        try:
            pagination = driver.find_element(By.CSS_SELECTOR, "div.pagination")
            page_links = pagination.find_elements(By.TAG_NAME, "a")
            page_numbers = [int(link.text) for link in page_links if link.text.strip().isdigit()]
            if page_numbers:
                pagination_info["max_page"] = max(page_numbers)
                pagination_info["current_page"] = int(page)
        except Exception as e:
            pagination_info["error"] = str(e)
        
        # Close the driver
        driver.quit()
        driver = None
        
        # Return the data
        return jsonify({
            "success": True,
            "special_issues_found": len(special_issues),
            "special_issues": special_issues,
            "pagination": pagination_info
        })
        
    except Exception as e:
        # Make sure to close the driver in case of error
        if driver:
            try:
                driver.quit()
            except:
                pass
        
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@app.route('/scrape-direct', methods=['GET'])
def scrape_direct():
    """
    A simplified version of the scraper that returns JSON data directly,
    designed to be extremely simple and browser-like.
    """
    try:
        # Create a browser-like session
        session = requests.Session()
        
        # Set extremely standard browser headers
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        # First visit the main MDPI site to get cookies
        session.get("https://www.mdpi.com/", headers=headers)
        
        # Then navigate to the special issues page like a normal browser would
        url = "https://www.mdpi.com/journal/applsci/special_issues"
        
        # Get the page content
        response = session.get(url, headers=headers)
        
        # Save the raw HTML for inspection
        with open("scrape_direct.html", "w", encoding="utf-8") as f:
            f.write(response.text)
        
        if response.status_code == 200:
            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Find all special issue items
            issues = []
            
            # Look specifically for the div structure from the example
            special_issue_items = soup.find_all('div', class_='generic-item article-item')
            
            for item in special_issue_items:
                issue_data = {}
                
                # Get the title and URL
                title_link = item.select_one('a.title-link')
                if title_link:
                    issue_data["title"] = title_link.text.strip()
                    issue_data["url"] = "https://www.mdpi.com" + title_link["href"] if not title_link["href"].startswith("http") else title_link["href"]
                    
                    # Extract keywords if available
                    keywords_div = item.select_one('div:-soup-contains("Keywords:")')
                    if keywords_div:
                        keywords_text = keywords_div.text.replace("Keywords:", "").strip()
                        issue_data["keywords"] = [k.strip() for k in keywords_text.split(';') if k.strip()]
                    
                    issues.append(issue_data)
            
            return jsonify({
                "status_code": response.status_code,
                "special_issues_found": len(issues),
                "data": issues[:10]  # Return first 10 for simplicity
            })
        else:
            return jsonify({
                "status_code": response.status_code,
                "error": f"Failed to retrieve the page: HTTP {response.status_code}"
            })
            
    except Exception as e:
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@app.route('/selenium-scrape', methods=['GET'])
def selenium_scrape():
    """
    Uses Selenium WebDriver to scrape the MDPI special issues page,
    which should bypass any anti-scraping mechanisms by acting like a real browser.
    """
    driver = None
    try:
        # Set up Chrome options
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless")  # Run in headless mode (no GUI)
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Add realistic user agent
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
        
        # Set up the Chrome driver
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
        
        # Page to scrape - get from query params or use default
        page = request.args.get('page', '1')
        url = f"https://www.mdpi.com/journal/applsci/special_issues?page_no={page}" if page != '1' else "https://www.mdpi.com/journal/applsci/special_issues"
        
        # Navigate to the URL
        driver.get(url)
        
        # Wait for the page to load (wait for special issue items to appear)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.generic-item.article-item"))
        )
        
        # Get the page source after JavaScript has executed
        page_source = driver.page_source
        
        # Save the HTML for inspection
        with open(f"selenium_page_{page}.html", "w", encoding="utf-8") as f:
            f.write(page_source)
        
        # Parse the special issues
        special_issues = []
        
        # Find all special issue elements
        issue_elements = driver.find_elements(By.CSS_SELECTOR, "div.generic-item.article-item")
        
        for element in issue_elements:
            issue_data = {}
            
            # Get the title and URL
            try:
                title_link = element.find_element(By.CSS_SELECTOR, "a.title-link")
                issue_data["title"] = title_link.text.strip()
                issue_data["url"] = title_link.get_attribute("href")
                
                # Look for keywords
                try:
                    # Try to find keywords directly in the listing
                    keywords_div = element.find_elements(By.XPATH, ".//div[contains(text(), 'Keywords:')]")
                    if keywords_div:
                        keywords_text = keywords_div[0].text.replace("Keywords:", "").strip()
                        issue_data["keywords"] = [k.strip() for k in keywords_text.split(';') if k.strip()]
                except Exception as ke:
                    # If keywords aren't in the listing, we'd need to visit the issue page
                    # We'll skip this for now to keep the response time reasonable
                    issue_data["keywords"] = []
                    issue_data["keywords_error"] = str(ke)
                
                # Get deadline if available
                try:
                    deadline_elem = element.find_elements(By.XPATH, ".//div[contains(text(), 'deadline')]")
                    if deadline_elem:
                        deadline_text = deadline_elem[0].text
                        deadline_match = re.search(r'deadline\s+(\d+\s+\w+\s+\d+)', deadline_text)
                        if deadline_match:
                            issue_data["deadline"] = deadline_match.group(1)
                except Exception:
                    pass
                
                special_issues.append(issue_data)
            except Exception as e:
                print(f"Error extracting issue data: {e}")
        
        # Take a screenshot for debugging
        driver.save_screenshot(f"selenium_screenshot_{page}.png")
        
        # Get pagination information
        pagination_info = {}
        try:
            pagination = driver.find_element(By.CSS_SELECTOR, "div.pagination")
            page_links = pagination.find_elements(By.TAG_NAME, "a")
            page_numbers = [int(link.text) for link in page_links if link.text.strip().isdigit()]
            if page_numbers:
                pagination_info["max_page"] = max(page_numbers)
                pagination_info["current_page"] = int(page)
        except Exception as e:
            pagination_info["error"] = str(e)
        
        # Close the driver
        driver.quit()
        driver = None
        
        # Return the data
        return jsonify({
            "success": True,
            "special_issues_found": len(special_issues),
            "special_issues": special_issues,
            "pagination": pagination_info
        })
        
    except Exception as e:
        # Make sure to close the driver in case of error
        if driver:
            try:
                driver.quit()
            except:
                pass
        
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@app.route('/selenium-test', methods=['GET'])
def selenium_test():
    """
    Simple route to test if Selenium is working properly.
    """
    driver = None
    try:
        # Set up Edge options
        edge_options = EdgeOptions()
        edge_options.add_argument("--headless")
        edge_options.add_argument("--no-sandbox")
        edge_options.add_argument("--disable-dev-shm-usage")
        
        # Create driver with explicit path to msedgedriver.exe
        driver_path = os.path.join(os.getcwd(), "msedgedriver.exe")
        service = EdgeService(executable_path=driver_path)
        driver = webdriver.Edge(service=service, options=edge_options)
        
        # Go to Google
        driver.get("https://www.google.com")
        
        # Take screenshot
        driver.save_screenshot("edge_test.png")
        
        # Get title
        title = driver.title
        
        # Close browser
        driver.quit()
        driver = None
        
        return jsonify({
            "success": True,
            "message": "Selenium with Edge is working correctly!",
            "title": title,
            "screenshot": "edge_test.png was saved",
            "driver_path": driver_path
        })
        
    except Exception as e:
        if driver:
            try:
                driver.quit()
            except:
                pass
                
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@app.route('/edge-scrape-store', methods=['GET'])
def edge_scrape_store():
    """
    Uses Edge WebDriver to scrape special issues and stores them in the database.
    This uses the local msedgedriver.exe in your project folder.
    """
    driver = None
    try:
        # Get parameters
        page = request.args.get('page', '1')
        max_pages = request.args.get('max_pages', '1')
        all_pages = request.args.get('all_pages', 'false').lower() == 'true'
        save_debug_files = request.args.get('save_debug_files', 'false').lower() == 'true'
        
        results = {
            "pages_processed": 0,
            "issues_found": 0,
            "issues_added": 0,
            "pages": []
        }
        
        # Set up Edge options
        edge_options = EdgeOptions()
        edge_options.add_argument("--headless")
        edge_options.add_argument("--no-sandbox") 
        edge_options.add_argument("--disable-dev-shm-usage")
        edge_options.add_argument("--disable-gpu")
        edge_options.add_argument("--window-size=1920,1080")
        edge_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0")
        
        # Set up driver with explicit path to msedgedriver.exe
        driver_path = os.path.join(os.getcwd(), "msedgedriver.exe")
        service = EdgeService(executable_path=driver_path)
        driver = webdriver.Edge(service=service, options=edge_options)
        
        current_page = int(page)
        end_page = 9999 if all_pages else (current_page + int(max_pages) - 1)
        
        # First, determine the total number of pages
        if all_pages:
            try:
                # Navigate to first page
                driver.get("https://www.mdpi.com/journal/applsci/special_issues")
                
                # Wait for the page to load
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.pagination"))
                )
                
                # Get the maximum page number
                pagination = driver.find_element(By.CSS_SELECTOR, "div.pagination")
                page_links = pagination.find_elements(By.TAG_NAME, "a")
                page_numbers = [int(link.text) for link in page_links if link.text.strip().isdigit()]
                
                if page_numbers:
                    max_page = max(page_numbers)
                    end_page = max_page
                    update_status(f"Found {max_page} total pages. Will scrape all pages.")
            except Exception as e:
                update_status(f"Error determining total pages: {str(e)}. Will use default max pages.")
                end_page = current_page + 10  # Default to 10 pages if we can't determine the total
        
        while current_page <= end_page:
            page_result = {
                "page": current_page,
                "issues_found": 0,
                "issues_added": 0
            }
            
            # Construct URL
            url = f"https://www.mdpi.com/journal/applsci/special_issues?page_no={current_page}" if current_page > 1 else "https://www.mdpi.com/journal/applsci/special_issues"
            
            # Navigate to page
            driver.get(url)
            
            # Take a screenshot for this page (only if debug files enabled)
            if save_debug_files:
                driver.save_screenshot(f"edge_store_screenshot_{current_page}.png")
            
            # Wait for content to load
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.generic-item.article-item"))
                )
            except Exception as e:
                page_result["error"] = f"Timeout waiting for page content: {str(e)}"
                results["pages"].append(page_result)
                # Check if we've reached the last page by looking for "next" button
                try:
                    next_button = driver.find_elements(By.XPATH, "//a[contains(text(), 'next')]")
                    if not next_button:
                        update_status(f"Reached the last page ({current_page-1}). No more pages available.")
                        break
                except:
                    pass
                break
            
            # Save page source for debugging (only if debug files enabled)
            if save_debug_files:
                with open(f"edge_store_page_{current_page}.html", "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
            
            # Find all special issue elements
            issue_elements = driver.find_elements(By.CSS_SELECTOR, "div.generic-item.article-item")
            page_result["issues_found"] = len(issue_elements)
            results["issues_found"] += len(issue_elements)
            
            update_status(f"Found {len(issue_elements)} special issues on page {current_page}")
            
            # Process each special issue
            for idx, element in enumerate(issue_elements):
                if scrape_status["should_cancel"]:
                    break
                    
                try:
                    # Extract title and URL
                    title_link = element.find_element(By.CSS_SELECTOR, "a.title-link")
                    title = title_link.text.strip()
                    url = title_link.get_attribute("href")
                    
                    update_status(f"Processing issue {idx+1}/{len(issue_elements)} on page {current_page}: {title[:40]}...")
                    
                    # Extract keywords from the listing if available
                    keywords = []
                    try:
                        keywords_div = element.find_elements(By.XPATH, ".//div[contains(text(), 'Keywords:')]")
                        if keywords_div:
                            keywords_text = keywords_div[0].text.replace("Keywords:", "").strip()
                            keywords = [k.strip() for k in keywords_text.split(';') if k.strip()]
                    except Exception as e:
                        print(f"Error extracting keywords from listing: {e}")
                    
                    # If no keywords in listing, visit the issue page to get them
                    if not keywords:
                        try:
                            # Add a delay to avoid being blocked
                            time.sleep(random.uniform(1.0, 2.0))
                            
                            # Open the special issue page in a new tab
                            driver.execute_script("window.open('');")
                            driver.switch_to.window(driver.window_handles[1])
                            driver.get(url)
                            
                            # Wait for the page to load
                            WebDriverWait(driver, 15).until(
                                EC.presence_of_element_located((By.TAG_NAME, "h1"))
                            )
                            
                            # Save the issue page for debugging (only if debug files enabled)
                            if save_debug_files:
                                with open(f"edge_issue_page_{current_page}_{idx+1}.html", "w", encoding="utf-8") as f:
                                    f.write(driver.page_source)
                            
                            # Method 1: Look for the keywords section in the specified format
                            try:
                                # Find h2 with name="keywords"
                                keywords_h2 = driver.find_elements(By.XPATH, "//h2[@name='keywords' or contains(text(), 'Keywords')]")
                                
                                if keywords_h2:
                                    # Find the container div that might contain the ul
                                    content_container = None
                                    
                                    # Try different approaches to find the keywords section
                                    # 1. Direct parent-child relationship
                                    container_divs = driver.find_elements(
                                        By.XPATH, 
                                        "//h2[@name='keywords' or contains(text(), 'Keywords')]/following-sibling::div[1]"
                                    )
                                    
                                    if container_divs:
                                        content_container = container_divs[0]
                                    
                                    # 2. With class name
                                    if not content_container:
                                        container_divs = driver.find_elements(
                                            By.XPATH, 
                                            "//div[contains(@class, 'content__container')]"
                                        )
                                        for div in container_divs:
                                            prev_h2 = div.find_elements(By.XPATH, "./preceding-sibling::h2[1]")
                                            if prev_h2 and ("keywords" in prev_h2[0].text.lower() or 
                                                           prev_h2[0].get_attribute("name") == "keywords"):
                                                content_container = div
                                                break
                                    
                                    # If we found the container, extract the keywords
                                    if content_container:
                                        # Look for list items in the container
                                        keyword_items = content_container.find_elements(By.CSS_SELECTOR, "ul li")
                                        if keyword_items:
                                            keywords = [item.text.strip() for item in keyword_items if item.text.strip()]
                            except Exception as e:
                                print(f"Error extracting keywords using method 1: {e}")
                            
                            # Method 2: If method 1 fails, try alternative approaches
                            if not keywords:
                                try:
                                    # Look for keywords in any section
                                    keywords_sections = driver.find_elements(
                                        By.XPATH, 
                                        "//*[self::h2 or self::h3 or self::h4][contains(text(), 'Keywords') or contains(text(), 'Key words')]"
                                    )
                                    
                                    if keywords_sections:
                                        for section in keywords_sections:
                                            # Look for a list after this heading
                                            next_list = section.find_elements(By.XPATH, "./following-sibling::ul[1]")
                                            if next_list:
                                                keyword_items = next_list[0].find_elements(By.TAG_NAME, "li")
                                                if keyword_items:
                                                    keywords = [item.text.strip() for item in keyword_items if item.text.strip()]
                                                    break
                                            
                                            # If no list, look for a div containing text
                                            next_div = section.find_elements(By.XPATH, "./following-sibling::div[1]")
                                            if next_div:
                                                # Check if the div contains a list
                                                div_list = next_div[0].find_elements(By.TAG_NAME, "ul")
                                                if div_list:
                                                    keyword_items = div_list[0].find_elements(By.TAG_NAME, "li")
                                                    if keyword_items:
                                                        keywords = [item.text.strip() for item in keyword_items if item.text.strip()]
                                                        break
                                                
                                                # If no list, check if it contains semicolon separated keywords
                                                div_text = next_div[0].text.strip()
                                                if div_text and ';' in div_text:
                                                    keywords = [k.strip() for k in div_text.split(';') if k.strip()]
                                                    break
                                                elif div_text and ',' in div_text:
                                                    keywords = [k.strip() for k in div_text.split(',') if k.strip()]
                                                    break
                                except Exception as e:
                                    print(f"Error extracting keywords using method 2: {e}")
                            
                            # Close the special issue tab and switch back to the main tab
                            driver.close()
                            driver.switch_to.window(driver.window_handles[0])
                        
                        except Exception as e:
                            print(f"Error visiting special issue page: {e}")
                            # Make sure we're back to the main tab
                            if len(driver.window_handles) > 1:
                                driver.close()
                                driver.switch_to.window(driver.window_handles[0])
                    
                    # Store in database
                    added = store_special_issue(title, url, keywords)
                    if added:
                        page_result["issues_added"] += 1
                        results["issues_added"] += 1
                        update_status(f"Added issue with {len(keywords)} keywords: {title[:40]}...")
                    else:
                        update_status(f"Issue already exists: {title[:40]}...")
                        
                except Exception as e:
                    print(f"Error processing issue: {e}")
                    update_status(f"Error processing issue: {str(e)}")
            
            # Add page result to overall results
            results["pages"].append(page_result)
            results["pages_processed"] += 1
            
            # Move to next page
            current_page += 1
            
            # Check if we've reached the end (for all_pages option)
            if all_pages:
                try:
                    next_buttons = driver.find_elements(By.XPATH, "//a[contains(text(), 'next') or contains(@class, 'next')]")
                    if not next_buttons:
                        update_status(f"Reached the last page. No more pages available.")
                        break
                except Exception:
                    pass
            
            # Add a random delay between pages
            time.sleep(random.uniform(2.0, 4.0))
        
        # Log the scraping session
        last_page_scraped = results["pages"][-1]["page"] if results["pages"] else int(page)
        log_scrape_session(results["pages_processed"], results["issues_added"], last_page_scraped)
        
        # Close the driver
        driver.quit()
        driver = None
        
        return jsonify({
            "success": True,
            "message": f"Added {results['issues_added']} new issues from {results['pages_processed']} pages",
            "results": results
        })
        
    except Exception as e:
        if driver:
            try:
                driver.quit()
            except:
                pass
                
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@app.route('/selenium-scrape-store', methods=['GET'])
def selenium_scrape_store():
    """
    Uses Selenium to scrape special issues and stores them in the database.
    This is a combination of the selenium scraping with database storage.
    """
    driver = None
    try:
        # Get parameters
        page = request.args.get('page', '1')
        max_pages = int(request.args.get('max_pages', '1'))
        
        results = {
            "pages_processed": 0,
            "issues_found": 0,
            "issues_added": 0,
            "pages": []
        }
        
        # Set up Chrome options
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox") 
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36")
        
        # Set up driver
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
        
        current_page = int(page)
        end_page = current_page + max_pages - 1
        
        while current_page <= end_page:
            page_result = {
                "page": current_page,
                "issues_found": 0,
                "issues_added": 0
            }
            
            # Construct URL
            url = f"https://www.mdpi.com/journal/applsci/special_issues?page_no={current_page}" if current_page > 1 else "https://www.mdpi.com/journal/applsci/special_issues"
            
            # Navigate to page
            driver.get(url)
            
            # Wait for content to load
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.generic-item.article-item"))
                )
            except Exception as e:
                page_result["error"] = f"Timeout waiting for page content: {str(e)}"
                results["pages"].append(page_result)
                break
            
            # Save page source for debugging
            with open(f"selenium_store_page_{current_page}.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            
            # Find all special issue elements
            issue_elements = driver.find_elements(By.CSS_SELECTOR, "div.generic-item.article-item")
            page_result["issues_found"] = len(issue_elements)
            results["issues_found"] += len(issue_elements)
            
            # Process each special issue
            for element in issue_elements:
                try:
                    # Extract title and URL
                    title_link = element.find_element(By.CSS_SELECTOR, "a.title-link")
                    title = title_link.text.strip()
                    url = title_link.get_attribute("href")
                    
                    # Extract keywords if available
                    keywords = []
                    try:
                        keywords_div = element.find_elements(By.XPATH, ".//div[contains(text(), 'Keywords:')]")
                        if keywords_div:
                            keywords_text = keywords_div[0].text.replace("Keywords:", "").strip()
                            keywords = [k.strip() for k in keywords_text.split(';') if k.strip()]
                    except:
                        # If no keywords in listing, we need to visit the issue page
                        # For this example we'll just not get keywords
                        pass
                    
                    # Store in database
                    added = store_special_issue(title, url, keywords)
                    if added:
                        page_result["issues_added"] += 1
                        results["issues_added"] += 1
                        
                except Exception as e:
                    print(f"Error processing issue: {e}")
            
            # Add page result to overall results
            results["pages"].append(page_result)
            results["pages_processed"] += 1
            
            # Move to next page
            current_page += 1
            
            # Add a random delay between pages
            time.sleep(random.uniform(1.5, 3.5))
        
        # Log the scraping session
        last_page_scraped = results["pages"][-1]["page"] if results["pages"] else int(page)
        log_scrape_session(results["pages_processed"], results["issues_added"], last_page_scraped)
        
        # Close the driver
        driver.quit()
        driver = None
        
        return jsonify({
            "success": True,
            "message": f"Added {results['issues_added']} new issues from {results['pages_processed']} pages",
            "results": results
        })
        
    except Exception as e:
        if driver:
            try:
                driver.quit()
            except:
                pass
                
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


if __name__ == '__main__':
    app.run(debug=True)