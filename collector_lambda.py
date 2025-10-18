import json
import os
import boto3
import praw
import urllib3
from datetime import datetime

# Initialize clients
s3 = boto3.client('s3')
http = urllib3.PoolManager()

def get_reddit_posts(limit=25):
    """Fetches post data from Reddit."""
    print("Collecting data from Reddit...")
    # This is the full, correct function
    posts_data = [] # Initialize the list
    reddit = praw.Reddit(
        client_id=os.environ['REDDIT_CLIENT_ID'],
        client_secret=os.environ['REDDIT_CLIENT_SECRET'],
        user_agent="MarketMind-Cloud-Agent v0.1",
        check_for_async=False
    )
    subreddit = reddit.subreddit("wallstreetbets")
    for post in subreddit.hot(limit=limit):
        posts_data.append({
            "source": "Reddit",
            "id": post.id,
            "title": post.title,
            "text": post.selftext
        })
    print(f"Successfully fetched {len(posts_data)} posts from Reddit.")
    return posts_data

def get_news_headlines():
    """Fetches financial news headlines from News API."""
    print("Collecting data from News API...")
    try:
        api_key = os.environ['NEWS_API_KEY']
        url = f"https://newsapi.org/v2/top-headlines?country=us&category=business&apiKey={api_key}"
        
        response = http.request('GET', url)
        data = json.loads(response.data.decode('utf-8'))
        
        headlines_data = []
        for article in data.get('articles', []):
            headlines_data.append({
                "source": "NewsAPI",
                "id": article.get('url'),
                "title": article.get('title'),
                "text": article.get('description')
            })
        print(f"Successfully fetched {len(headlines_data)} news headlines.")
        return headlines_data
    except Exception as e:
        print(f"Error fetching from News API: {e}")
        return []

def lambda_handler(event, context):
    """
    Main handler: collects data from Reddit AND News API, then saves to S3.
    """
    print("--- MarketMind Data Collector: Activated ---")
    
    try:
        reddit_posts = get_reddit_posts()
        news_headlines = get_news_headlines()
        
        combined_data = reddit_posts + news_headlines
        
        if not combined_data:
            print("No data collected. Exiting.")
            return {'statusCode': 200, 'body': json.dumps('No data collected.')}

        bucket_name = os.environ['S3_BUCKET_NAME']
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        file_name = f"raw_data_{timestamp}.json"
        
        s3.put_object(
            Bucket=bucket_name,
            Key=f"raw/{file_name}",
            Body=json.dumps(combined_data, indent=4)
        )
        
        print(f"Successfully saved {len(combined_data)} total records to S3.")
        
        return {
            'statusCode': 200,
            'body': json.dumps('Multi-source data collection successful!')
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        raise e