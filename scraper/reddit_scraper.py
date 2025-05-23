import praw
import logging
import re
from typing import List, Dict
from datetime import datetime

class RedditScraper:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.logger = logging.getLogger(__name__)
        try:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            self.logger.info("Reddit API initialized successfully")
        except Exception as e:
            self.logger.error(f"Reddit API initialization failed: {str(e)}")
            raise

    def scrape_mentalhealth(self, keywords: List[str], limit: int = 10) -> List[Dict]:
        """Scrape mental health related posts from Reddit based on keywords"""
        if not self.reddit:
            self.logger.warning("Reddit API not initialized - skipping scrape")
            return []

        results = []
        subreddits = ['mentalhealth', 'depression', 'anxiety', 'therapy', 'CPTSD']
        
        try:
            for sub in subreddits:
                subreddit = self.reddit.subreddit(sub)
                moderators = self._get_moderators(subreddit)
                
                for submission in subreddit.hot(limit=limit * 3):
                    if len(results) >= limit:
                        break
                        
                    if not submission.author:
                        continue
                        
                    if self._is_moderator_post(submission, moderators, sub):
                        continue
                        
                    content = f"{submission.title}\n\n{submission.selftext}"
                    if any(kw.lower() in content.lower() for kw in keywords):
                        results.append({
                            'source': 'reddit',
                            'subreddit': sub,
                            'title': submission.title,
                            'content': content,
                            'author': submission.author.name,
                            'date': datetime.utcfromtimestamp(submission.created_utc).isoformat(),
                            'url': f"https://reddit.com{submission.permalink}",
                            'upvotes': submission.score,
                            'comments': submission.num_comments,
                            'is_moderator': False
                        })
                        
        except Exception as e:
            self.logger.error(f"Reddit scraping error: {str(e)}")

        return results
        
    def _is_moderator_post(self, submission, moderators, subreddit_name):
        """Comprehensive check for moderator posts"""
        # Check if author is in moderator list
        if submission.author and submission.author.name in moderators:
            return True
        
        # Check flair for MOD indicators
        if hasattr(submission, 'author_flair_text') and submission.author_flair_text:
            flair_text = submission.author_flair_text.lower()
            if 'mod' in flair_text or 'moderator' in flair_text:
                return True
        
        # Check CSS class for MOD indicators
        if hasattr(submission, 'author_flair_css_class') and submission.author_flair_css_class:
            if 'mod' in submission.author_flair_css_class.lower():
                return True
        
        # Check if post is stickied or distinguished
        if submission.stickied or getattr(submission, 'distinguished', None):
            return True
        
        # Check title and content for MOD-related terms
        mod_keywords = [
            'moderator', 'mod ', 'mods ', 'modding', 'moderation',
            'rules', 'rule', 'announcement', 'official', 'meta',
            'welcome', 'introduction', 'guideline', 'reminder', 'update'
        ]
        content = f"{submission.title}\n{submission.selftext}".lower()
        return any(keyword in content for keyword in mod_keywords)
        
    def _get_moderators(self, subreddit) -> List[str]:
        """Get list of moderator usernames for a subreddit with caching"""
        if not hasattr(self, '_moderator_cache'):
            self._moderator_cache = {}
            
        sub_name = subreddit.display_name
        
        if sub_name not in self._moderator_cache:
            try:
                self._moderator_cache[sub_name] = [mod.name for mod in subreddit.moderator()]
                self.logger.info(f"Retrieved and cached {len(self._moderator_cache[sub_name])} moderators for r/{sub_name}")
            except Exception as e:
                self.logger.error(f"Failed to get moderators for r/{sub_name}: {str(e)}")
                self._moderator_cache[sub_name] = []
        
        return self._moderator_cache[sub_name]