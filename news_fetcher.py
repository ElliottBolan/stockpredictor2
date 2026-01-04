"""
News Fetcher
Fetches real-time news for stocks and cryptocurrencies using NewsAPI
"""
import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class NewsFetcher:
    """Fetches news articles for stocks, cryptocurrencies, and general market news"""
    
    def __init__(self, api_key=None):
        """
        Initialize NewsFetcher
        
        Args:
            api_key (str): NewsAPI key (optional, will use .env if not provided)
        """
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        self.base_url = "https://newsapi.org/v2/everything"
        self.top_headlines_url = "https://newsapi.org/v2/top-headlines"
    
    def fetch_symbol_news(self, symbol, company_name=None, max_results=5):
        """
        Fetch news for a specific stock or cryptocurrency
        
        Args:
            symbol (str): Stock/crypto ticker symbol
            company_name (str): Company name for better search results
            max_results (int): Maximum number of articles to return
        
        Returns:
            list: List of news articles
        """
        if not self.api_key or self.api_key == 'your_newsapi_key_here':
            print("Warning: No valid NewsAPI key found. Please set NEWS_API_KEY in .env file")
            return self._get_mock_news(symbol, company_name)
        
        try:
            # Build search query
            query = symbol
            if company_name and company_name != symbol:
                query = f"{company_name} OR {symbol}"
            
            # Get news from the last 7 days
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'publishedAt',
                'pageSize': max_results,
                'language': 'en',
                'apiKey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'ok':
                return self._format_articles(data['articles'])
            else:
                print(f"Error fetching news: {data.get('message', 'Unknown error')}")
                return []
        
        except Exception as e:
            print(f"Error fetching news for {symbol}: {e}")
            return self._get_mock_news(symbol, company_name)
    
    def fetch_general_market_news(self, max_results=10):
        """
        Fetch general market news when no specific symbol is selected
        
        Args:
            max_results (int): Maximum number of articles to return
        
        Returns:
            list: List of news articles
        """
        if not self.api_key or self.api_key == 'your_newsapi_key_here':
            print("Warning: No valid NewsAPI key found. Please set NEWS_API_KEY in .env file")
            return self._get_mock_general_news()
        
        try:
            params = {
                'category': 'business',
                'pageSize': max_results,
                'language': 'en',
                'apiKey': self.api_key
            }
            
            response = requests.get(self.top_headlines_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'ok':
                return self._format_articles(data['articles'])
            else:
                print(f"Error fetching general news: {data.get('message', 'Unknown error')}")
                return []
        
        except Exception as e:
            print(f"Error fetching general market news: {e}")
            return self._get_mock_general_news()
    
    def _format_articles(self, articles):
        """
        Format articles for display
        
        Args:
            articles (list): Raw articles from API
        
        Returns:
            list: Formatted articles
        """
        formatted = []
        for article in articles:
            formatted.append({
                'title': article.get('title', 'No title'),
                'description': article.get('description', 'No description'),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', '')
            })
        return formatted
    
    def _get_mock_news(self, symbol, company_name):
        """Return mock news when API key is not available"""
        return [
            {
                'title': f'{company_name or symbol} - Real-time news requires API key',
                'description': 'To fetch real-time news, please sign up for a free NewsAPI key at https://newsapi.org/ and add it to your .env file',
                'source': 'System',
                'url': 'https://newsapi.org/',
                'published_at': datetime.now().isoformat()
            }
        ]
    
    def _get_mock_general_news(self):
        """Return mock general news when API key is not available"""
        return [
            {
                'title': 'Real-time market news requires API key',
                'description': 'To fetch real-time market news, please sign up for a free NewsAPI key at https://newsapi.org/ and add it to your .env file',
                'source': 'System',
                'url': 'https://newsapi.org/',
                'published_at': datetime.now().isoformat()
            }
        ]
    
    def display_news(self, articles):
        """
        Display news articles in a formatted way
        
        Args:
            articles (list): List of news articles
        """
        if not articles:
            print("No news articles found.")
            return
        
        print("\n" + "="*80)
        print("NEWS ARTICLES")
        print("="*80 + "\n")
        
        for i, article in enumerate(articles, 1):
            print(f"{i}. {article['title']}")
            print(f"   Source: {article['source']}")
            print(f"   Published: {article['published_at']}")
            if article['description']:
                print(f"   {article['description']}")
            print(f"   URL: {article['url']}")
            print("-" * 80)
