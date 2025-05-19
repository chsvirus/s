import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import firebase_admin
from firebase_admin import credentials, firestore
import re
from keybert import KeyBERT

def get_recommendations():
    # Initialize
    print("\n=====================================================")
    print("🔄 INITIALIZING RECOMMENDATION SYSTEM...")
    print("=====================================================")
    db = firestore.client()
    kw_model = KeyBERT()
    print("✅ Successfully initialized Firestore DB and KeyBERT model")
    
    # Load data
    print("\n=====================================================")
    print("📊 LOADING DATA FROM FIRESTORE...")
    print("=====================================================")
    df = pd.DataFrame([{
        'title': doc.to_dict().get('title', ''),
        'url': doc.to_dict().get('url', ''),
        'keywords': doc.to_dict().get('keywords', [])
    } for doc in db.collection('special_issues').stream()])
    print(f"✅ Successfully loaded {len(df)} special issues from database")
    
    # Get user input
    print("\n=====================================================")
    print("🔍 COLLECTING USER PREFERENCES...")
    print("=====================================================")
    while True:
        keywords = input("\nEnter research interests (comma-separated): ").strip()
        if keywords and all(re.match(r'^[a-zA-Z0-9\s]+$', k.strip()) for k in keywords.split(',')):
            break
        print("Invalid! Use letters and numbers only")
    
    # Get recommendation count
    num_recs = 10
    try:
        user_input = input("\nEnter number of recommendations (default: 10): ").strip()
        if user_input:
            num_recs = max(1, int(user_input))
    except ValueError:
        pass
    print(f"✅ User requested {num_recs} recommendations")
    
    # Enhance keywords
    print("\n=====================================================")
    print("🧠 ENHANCING USER KEYWORDS...")
    print("=====================================================")
    print(f"📌 Original keywords: {keywords}")
    user_keywords = [k.strip() for k in keywords.split(',')]
    enhanced_user_keywords = user_keywords + [
        kw for kw, _ in kw_model.extract_keywords(' '.join(user_keywords), top_n=3) 
        if kw not in user_keywords
    ]
    print(f"✅ Enhanced search keywords: {', '.join(enhanced_user_keywords)}")
    
    # Enhance document keywords
    print("\n=====================================================")
    print("🔄 ENHANCING DOCUMENT KEYWORDS...")
    print("=====================================================")
    keyword_count_before = sum(len(kw) for kw in df['keywords'])
    print(f"📊 Total keywords before enhancement: {keyword_count_before}")
    
    for i, row in df.iterrows():
        if row['keywords']:
            doc_text = row['title'] + ' ' + ' '.join(row['keywords'])
            new_keywords = [kw for kw, _ in kw_model.extract_keywords(doc_text, top_n=3)
                          if kw not in row['keywords']]
            df.at[i, 'keywords'] = row['keywords'] + new_keywords
    
    keyword_count_after = sum(len(kw) for kw in df['keywords'])
    print(f"✅ Total keywords after enhancement: {keyword_count_after}")
    print(f"✅ Added {keyword_count_after - keyword_count_before} new keywords")
    
    # Calculate similarities
    print("\n=====================================================")
    print("🧮 CALCULATING SIMILARITY SCORES...")
    print("=====================================================")
    df['keywords_text'] = df['keywords'].apply(lambda x: ' '.join(x))
    corpus = [' '.join(enhanced_user_keywords)] + df['keywords_text'].tolist()
    
    print("📊 Creating document vectors...")
    vectorizer = CountVectorizer()
    vector_matrix = vectorizer.fit_transform(corpus)
    print(f"✅ Vector matrix created with shape: {vector_matrix.shape}")
    
    print("📊 Computing cosine similarities...")
    similarity_scores = cosine_similarity(vector_matrix)[0, 1:].flatten()
    df['similarity_score'] = similarity_scores
    print(f"✅ Similarity calculation complete")
    
    # Sort and prepare results
    print("\n=====================================================")
    print("📋 PREPARING FINAL RECOMMENDATIONS...")
    print("=====================================================")
    sorted_results = df.sort_values('similarity_score', ascending=False).head(num_recs)
    print(f"✅ Sorted {len(df)} items by relevance")
    print(f"✅ Selected top {num_recs} matches")
    
    # Display results
    print("\n=====================================================")
    print(f"🏆 TOP {num_recs} RECOMMENDATIONS:")
    print("=====================================================")
    for idx, (_, row) in enumerate(sorted_results.iterrows(), 1):
        keywords_display = ', '.join(row['keywords'][:5])
        if len(row['keywords']) > 5:
            keywords_display += f" (+ {len(row['keywords']) - 5} more)"
        
        print(f"\n{idx}. {row['title']}")
        print(f"   Score: {row['similarity_score']:.2%}")
        print(f"   URL: {row['url']}")
        print(f"   Keywords: {keywords_display}")
    
    print("\n=====================================================")
    print("✅ RECOMMENDATION PROCESS COMPLETE")
    print("=====================================================")

if __name__ == "__main__":
    print("\n=====================================================")
    print("🚀 STARTING SPECIAL ISSUES RECOMMENDER")
    print("=====================================================")
    print("🔑 Initializing Firebase connection...")
    firebase_admin.initialize_app(credentials.Certificate('special-issues-project-firebase-adminsdk-fbsvc-0b2060ce4b.json'))
    print("✅ Firebase connection established")
    get_recommendations()