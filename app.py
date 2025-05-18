from flask import Flask, render_template, request, jsonify
from scraper.reddit_scraper import RedditScraper
from scraper.gemini_analyzer import GeminiAnalyzer  
from scraper.sentiment_analyzer import SentimentAnalyzer
import os
import json
import hashlib
from typing import List, Dict
from datetime import datetime
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

# Check for required API keys
if not os.getenv('REDDIT_CLIENT_ID') or not os.getenv('REDDIT_CLIENT_SECRET'):
    print("WARNING: Reddit API credentials not set. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.")

if not os.getenv('GEMINI_API_KEY'):
    print("WARNING: Gemini API key not set. Set GEMINI_API_KEY environment variable.")

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize components
reddit_scraper = RedditScraper(
    client_id=os.getenv('REDDIT_CLIENT_ID'),
    client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)
gemini_analyzer = GeminiAnalyzer()
sentiment_analyzer = SentimentAnalyzer()

def extract_keywords(results, top_n=5):
    """Extract mental health keywords from posts"""
    mental_health_terms = [
        'anxiety', 'depression', 'therapy', 'stress', 'trauma',
        'coping', 'medication', 'support', 'recovery', 'mental health',
        'therapist', 'counseling', 'self-care', 'diagnosis', 'symptoms',
        'treatment', 'healing', 'crisis', 'panic', 'grief'
    ]
    
    keyword_counts = Counter()
    
    for post in results:
        content = (post.get('title', '') + ' ' + post.get('content', '')).lower()
        for term in mental_health_terms:
            if term in content:
                keyword_counts[term] += 1
    
    return dict(keyword_counts.most_common(top_n))

def filter_mod_posts(posts: List[Dict]) -> List[Dict]:
    """Filter out any posts that might be from moderators with comprehensive checks"""
    mod_keywords = [
        'moderator', 'mod ', 'mods ', 'modding', 'moderation', 
        'rules', 'rule', 'announcement', 'official', 'meta',
        'welcome', 'introduction', 'guideline', 'reminder', 'update'
    ]
    filtered = []
    
    for post in posts:
        # Skip if explicitly marked as mod
        if post.get('is_moderator', False):
            continue
            
        # Check content for mod keywords in title or content
        title = post.get('title', '').lower()
        content = post.get('content', '').lower()
        
        # Skip if title starts with [Mod] or similar
        if title.startswith(('[mod]', '[meta]', '[announcement]')):
            continue
            
        # Check for mod keywords in content
        combined = f"{title}\n{content}"
        if any(keyword in combined for keyword in mod_keywords):
            continue
            
        # Skip posts with very high upvote-to-comment ratios (common for mod posts)
        upvotes = post.get('upvotes', 0)
        comments = post.get('comments', 1)
        if upvotes > 100 and comments < 3:  # Suspicious ratio
            continue
            
        filtered.append(post)
        
    return filtered

def get_cache_key(keywords, limit):
    """Generate a cache key based on search parameters"""
    key_string = f"{','.join(sorted(keywords))}-{limit}"
    return hashlib.md5(key_string.encode()).hexdigest()

def get_cached_results(cache_key):
    """Try to get results from cache with mod post filtering"""
    cache_file = f"data/cache_{cache_key}.json"
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
            return filter_mod_posts(cached_data)  # Always filter mod posts when loading
    return None

def cache_results(cache_key, results):
    """Save results to cache after filtering mod posts"""
    cache_file = f"data/cache_{cache_key}.json"
    filtered_results = filter_mod_posts(results)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_results, f)

def clean_cache(cache_key=None):
    """Clean the cache by filtering out mod posts from cache files"""
    os.makedirs('data', exist_ok=True)
    cache_files = []
    
    if cache_key:
        cache_files = [f"data/cache_{cache_key}.json"]
    else:
        cache_files = [f for f in os.listdir('data') if f.startswith('cache_')]
        cache_files = [os.path.join('data', f) for f in cache_files]
    
    cleaned_count = 0
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                
                original_count = len(cached_data)
                filtered_data = filter_mod_posts(cached_data)
                
                if len(filtered_data) < original_count:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
                    cleaned_count += (original_count - len(filtered_data))
                    
            except Exception as e:
                print(f"Error cleaning cache file {cache_file}: {str(e)}")
                continue
    
    return cleaned_count

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def start_scraping():
    try:
        data = request.get_json()
        keywords = [k.strip() for k in data.get('keywords', '').split(',') if k.strip()]
        limit = int(data.get('limit', 10))
        
        if not keywords:
            return jsonify({'status': 'error', 'message': 'Please enter keywords'}), 400

        # Check cache first
        cache_key = get_cache_key(keywords, limit)
        cached_results = get_cached_results(cache_key)  # This now includes filtering
        
        if cached_results:
            return jsonify({
                'status': 'success',
                'count': len(cached_results),
                'preview': cached_results[:3],
                'file': f"cache_{cache_key}.json",
                'sentiment_counts': get_sentiment_counts(cached_results),
                'cached': True
            })

        # Scrape Reddit
        reddit_results = reddit_scraper.scrape_mentalhealth(keywords, limit=limit)
        filtered_results = filter_mod_posts(reddit_results)
        
        if not filtered_results:
            return jsonify({'status': 'success', 'count': 0, 'message': 'No results found'})

        # Analyze content
        analyzed_results = []
        for post in filtered_results:
            sentiment = sentiment_analyzer.analyze_text(post['content'])
            insight = gemini_analyzer.generate_insight(post['content'])
            
            analyzed_post = {
                **post,
                'sentiment': sentiment['label'],
                'sentiment_score': sentiment['score'],
                'insight': insight
            }
            analyzed_results.append(analyzed_post)

        # Save and cache filtered results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reddit_results_{timestamp}.json"
        save_results(analyzed_results, filename)
        
        # Only cache already-filtered results
        cache_results(cache_key, analyzed_results)
        
        return jsonify({
            'status': 'success',
            'count': len(analyzed_results),
            'preview': analyzed_results[:3],
            'file': filename,
            'sentiment_counts': get_sentiment_counts(analyzed_results)
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'status': 'error', 'message': 'No text provided'}), 400
            
        sentiment = sentiment_analyzer.analyze_text(text)
        insight = gemini_analyzer.generate_insight(text)
        
        return jsonify({
            'status': 'success',
            'sentiment': sentiment,
            'insight': insight
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/clean-cache', methods=['POST'])
def clean_cache_route():
    try:
        data = request.get_json()
        cache_key = data.get('cache_key', None)
        
        cleaned_count = clean_cache(cache_key)
        
        return jsonify({
            'status': 'success',
            'message': f'Cleaned {cleaned_count} cache files',
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/dashboard')
def show_dashboard():
    try:
        data_files = sorted(
            [f for f in os.listdir('data') if f.startswith('reddit_results_')],
            reverse=True
        )
        latest_file = data_files[0] if data_files else None
        
        if latest_file:
            results = load_results(f"data/{latest_file}")
            # Always filter mod posts when displaying results
            filtered_results = filter_mod_posts(results)  
            sentiment_counts = get_sentiment_counts(filtered_results)
            keywords = extract_keywords(filtered_results, top_n=5)
        else:
            filtered_results = []
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            keywords = {}
        
        return render_template('dashboard.html', 
                            results=filtered_results,
                            sentiment_counts=sentiment_counts,
                            keywords=keywords)
    
    except Exception as e:
        return render_template('dashboard.html', 
                            error=str(e),
                            results=[],
                            sentiment_counts={'positive': 0, 'negative': 0, 'neutral': 0},
                            keywords={})

@app.route('/about')
def about():
    return render_template('about.html')

def save_results(data, filename):
    """Save results to file, making sure we only save filtered results"""
    os.makedirs('data', exist_ok=True)
    # Make sure we're saving filtered data
    filtered_data = filter_mod_posts(data)
    with open(f"data/{filename}", 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)

def load_results(filename):
    """Load results from file and filter mod posts"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Always filter mod posts when loading
    return filter_mod_posts(data)  

def get_sentiment_counts(results):
    return {
        'positive': len([r for r in results if r.get('sentiment') == 'POSITIVE']),
        'negative': len([r for r in results if r.get('sentiment') == 'NEGATIVE']),
        'neutral': len([r for r in results if not r.get('sentiment') or r.get('sentiment') not in ['POSITIVE', 'NEGATIVE']])
    }

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    app.run(debug=True)
