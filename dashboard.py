import streamlit as st
import boto3
import json
import pandas as pd
from textblob import TextBlob
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf

# --- AWS & Page Configuration ---
S3_BUCKET_NAME = "marketmind-raw-data-ramij-2025" 
bedrock = boto3.client('bedrock-runtime')

st.set_page_config(page_title="MarketMind Dashboard", layout="wide")
st.title("🧠 MarketMind: AI Financial Risk & Sentiment Agent")

# --- Helper Functions (All our previous functions) ---

@st.cache_data(ttl=600)
def load_raw_data_with_timestamps(bucket_name):
    # (Same as before)
    s3 = boto3.client('s3')
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix="raw/")
    if 'Contents' not in objects: return pd.DataFrame()
    all_posts = []
    files_to_load = sorted(objects['Contents'], key=lambda x: x['LastModified'], reverse=True)[:20]
    for obj in files_to_load:
        file_key = obj['Key']
        try:
            timestamp_str = file_key.split('_')[-1].replace('.json', '')
            file_timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M-%S')
        except (IndexError, ValueError): file_timestamp = obj['LastModified']
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        posts = json.loads(response['Body'].read().decode('utf-8'))
        for post in posts:
            if isinstance(post, dict):
                post['timestamp'] = file_timestamp
                all_posts.append(post)
    return pd.DataFrame(all_posts)

@st.cache_data(ttl=600)
def load_latest_analysis(bucket_name):
    # (Same as before)
    s3 = boto3.client('s3')
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix="processed/")
    if 'Contents' not in objects: return None
    latest_file = max(objects['Contents'], key=lambda x: x['LastModified'])
    response = s3.get_object(Bucket=bucket_name, Key=latest_file['Key'])
    analysis = json.loads(response['Body'].read().decode('utf-8'))
    return analysis

def analyze_sentiment(text):
    
    if not isinstance(text, str): return "Neutral"
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1: return "Positive"
    elif analysis.sentiment.polarity < -0.1: return "Negative"
    else: return "Neutral"

def get_sentiment_score(text):
    
    if not isinstance(text, str): return 0.0
    return TextBlob(text).sentiment.polarity

def ask_bedrock_question(question, context_data):
    
    prompt = f"""Using the following social media data as context, please answer the user's question. Provide a concise, direct answer.
    <context_data>{context_data}</context_data>
    Question: {question}"""
    modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 512,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    })
    response = bedrock.invoke_model(body=body, modelId=modelId)
    response_body = json.loads(response.get('body').read())
    return response_body.get('content')[0].get('text')

def summarize_stock_news(news_articles):
    
    context_data = ""
    for article in news_articles[:5]:
        title = article.get('title', 'No Title Available')
        link = article.get('link', 'No Link Available')
        context_data += f"Title: {title}\nLink: {link}\n\n"
    if not context_data.strip(): return "No usable news articles found to summarize."
    prompt = f"""
    You are a financial analyst. Based on the following news articles, provide a brief, one-paragraph summary of the current sentiment and key events for this company.
    <news_data>{context_data}</news_data>
    Summary:
    """
    modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 512,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    })
    response = bedrock.invoke_model(body=body, modelId=modelId)
    response_body = json.loads(response.get('body').read())
    return response_body.get('content')[0].get('text')

@st.cache_data(ttl=300)
def get_data_for_ai_comparison(tickers_list):
    
    all_data = ""
    for ticker_str in tickers_list:
        try:
            ticker = yf.Ticker(ticker_str)
            info = ticker.info
            all_data += f"\n--- Data for {ticker_str} ({info.get('longName', 'N/A')}) ---\n"
            beta = info.get('beta', 'N/A')
            pe_ratio = info.get('trailingPE', 'N/A')
            all_data += f"Beta (Volatility vs. Market): {beta}\n"
            all_data += f"P/E Ratio (Valuation): {pe_ratio}\n"
            news = ticker.news
            if news:
                all_data += "Recent News:\n"
                for item in news[:3]: all_data += f"- {item.get('title', 'No Title')}\n"
            else: all_data += "No recent news found.\n"
        except Exception as e: all_data += f"Could not retrieve data for {ticker_str}. Error: {e}\n"
    return all_data


def get_comparative_ai_analysis(context_data):
    """Sends the comparison data to Bedrock for a final analysis."""
    prompt = f"""
    You are an expert financial analyst. Based *only* on the data provided below, compare the following stocks.
    1.  Analyze their stability (using Beta, where >1 is more volatile than the market, <1 is less).
    2.  Analyze their valuation (using P/E Ratio, where higher is more 'expensive').
    3.  Briefly summarize the sentiment from their recent news.
    4.  **Conclude with a summary of the risk/return profile for each stock. For example, which stock might offer more return but with higher risk (higher Beta), and which seems to be the most stable or 'safest' investment and why.**

    <data>
    {context_data}
    </data>
    
    Analysis:
    """
    modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 1024,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    })
    response = bedrock.invoke_model(body=body, modelId=modelId)
    response_body = json.loads(response.get('body').read())
    return response_body.get('content')[0].get('text')

# Main Dashboard Layout 


data_load_state = st.text('Loading agent data from AWS...')
df_reddit = load_raw_data_with_timestamps(S3_BUCKET_NAME)
analysis_data = load_latest_analysis(S3_BUCKET_NAME)
data_load_state.text('Data loaded successfully! ✅')


st.header("Live Stock Analyzer 📈")
with st.form("stock_analyzer_form"):
    stock_symbol = st.text_input("Enter a stock ticker (e.g., TSLA, AAPL, MSFT)", "TSLA")
    
    
    time_period = st.radio(
        "Select time period:",
        ('5d', '1mo', '6mo', '1y', 'max'),
        index=2,
        horizontal=True
    )
    
    submitted = st.form_submit_button("Analyze Stock")

if submitted and stock_symbol:
    try:
        ticker = yf.Ticker(stock_symbol)
        
        
        hist_chart = ticker.history(period=time_period)
        
        
        hist_metric = ticker.history(period="2d") 
        
        if hist_metric.empty:
            raise ValueError("No history found for this ticker. It may be invalid or delisted.")
            
        current_price = hist_metric['Close'].iloc[-1]
        prev_close = hist_metric['Close'].iloc[-2]
        price_delta = current_price - prev_close
        
        
        news = ticker.news
        ai_summary = "No recent news found for this ticker."
        if news:
            with st.spinner("Generating AI news summary..."):
                ai_summary = summarize_stock_news(news)

    
        st.subheader(f"{ticker.info.get('longName', stock_symbol.upper())} Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            
            st.metric(
                label="Current Share Price", 
                value=f"${current_price:,.2f}", 
                delta=f"${price_delta:,.2f} (Today)"
            )
        with col2:
            st.info(f"**🧠 AI News Summary:**\n\n{ai_summary}")
        
        st.subheader(f"Price Chart ({time_period})")
        st.line_chart(hist_chart['Close'], use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not retrieve data for {stock_symbol}. Please check the ticker. Error: {e}")

st.divider()


st.header("Live Stock Comparator 📊")
with st.form("stock_comparator_form"):
    ticker_input = st.text_input("Enter stock tickers (comma-separated, e.g., TSLA, AAPL, MSFT)", "TSLA, GOOG")
    
    time_period_comp = st.radio(
        "Select time period:",
        ('1mo', '6mo', '1y', '5y', 'max'),
        index=2,
        horizontal=True,
        key="comparator_time" 
    )
    
    submitted_compare = st.form_submit_button("Compare Stocks")

if submitted_compare and ticker_input:
    tickers_list = [t.strip().upper() for t in ticker_input.split(',')]
    try:
        hist_comp = yf.download(tickers_list, period=time_period_comp)
        if hist_comp.empty:
            raise ValueError("No data returned. Check ticker symbols.")
            
        st.subheader(f"Price Chart Comparison ({time_period_comp})")
        st.line_chart(hist_comp['Close'], use_container_width=True)
        
        with st.spinner("Generating AI comparison..."):
            comparison_context = get_data_for_ai_comparison(tickers_list)
            
            ai_comparison = get_comparative_ai_analysis(comparison_context)
            st.subheader("🤖 AI Comparative Analysis (Risk vs. Return)")
            st.info(ai_comparison)
            
    except Exception as e:
        st.error(f"Could not retrieve data. Please check ticker symbols. Error: {e}")

st.divider()


st.header("Reddit Sentiment Risk Assessment (Background Agent)")
if analysis_data:
    alert_type = analysis_data.get("alert_type", "N/A")
    color = "red" if alert_type == "High Risk" else "green"
    st.subheader(f"Alert Status: :{color}[{alert_type}]")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Sentiment Trend", value=analysis_data.get("sentiment_trend", "N/A"))
    with col2:
        st.info(f"**🧠 AI Reasoning:**\n\n{analysis_data.get('reasoning', 'No reasoning provided.')}")
else:
    st.warning("No AI analysis found yet.")

st.divider()


st.header("Ask the Agent About Reddit Data")
with st.form("qa_form"):
    user_question = st.text_input("E.g., 'Summarize the most discussed topics on Reddit.'")
    submitted_qa = st.form_submit_button("Ask Agent")
if submitted_qa and user_question and not df_reddit.empty:
    with st.spinner("Thinking..."):
        context_data = "\n".join(df_reddit['title'].dropna().tolist())
        ai_answer = ask_bedrock_question(user_question, context_data[:10000])
        st.info(f"**🤖 Agent's Answer:**\n\n{ai_answer}")

st.divider()


st.header("Reddit Data Visualizations")
if not df_reddit.empty:
    df_reddit["sentiment_label"] = df_reddit["title"].apply(analyze_sentiment)
    df_reddit["sentiment_score"] = df_reddit["title"].apply(get_sentiment_score)
    st.subheader("Sentiment Trend Over Time (Reddit)")
    time_series_df = df_reddit.set_index('timestamp').resample('15Min')['sentiment_score'].mean().dropna()
    st.line_chart(time_series_df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df_reddit['sentiment_label'].value_counts()
        fig_pie = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.subheader("Trending Topics Word Cloud")
        text = " ".join(title for title in df_reddit.title.dropna())
        wordcloud = WordCloud(width=800, height=400, background_color=None, colormap='viridis').generate(text)
        fig_wc, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig_wc)
    st.subheader("Latest Raw Reddit Data")
    st.dataframe(df_reddit)
else:
    st.warning("No raw data found in the S3 bucket.")