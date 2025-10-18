import streamlit as st
import boto3
import json
import pandas as pd
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go # NEW: Import Plotly graph objects
from plotly.subplots import make_subplots # NEW: For dual y-axis
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import os # Ensure os is imported if debugging env vars

# --- AWS & Page Configuration ---
S3_BUCKET_NAME = "marketmind-raw-data-ramij-2025" 
try:
    aws_creds = {
        "aws_access_key_id": st.secrets["aws"]["aws_access_key_id"],
        "aws_secret_access_key": st.secrets["aws"]["aws_secret_access_key"],
        "region_name": st.secrets["aws"]["region"]
    }
    bedrock = boto3.client('bedrock-runtime', **aws_creds)
    # S3 clients will be created inside functions with creds
except KeyError as e:
    st.error(f"Error accessing Streamlit Secrets! Make sure '[aws]' section exists. Missing key: {e}")
    st.stop()

st.set_page_config(page_title="MarketMind Dashboard", layout="wide")
st.title("🧠 MarketMind: AI Financial Risk & Sentiment Agent")

# --- Helper Functions ---
# (load_raw_data_with_timestamps, load_latest_analysis, analyze_sentiment, get_sentiment_score, ask_bedrock_question, summarize_stock_news, get_data_for_ai_comparison, get_comparative_ai_analysis are the same as before)
# ... Make sure all previous helper functions are pasted here ...
@st.cache_data(ttl=600) # Cache S3 data for 10 minutes
def load_raw_data_with_timestamps(bucket_name):
    """Loads raw data and extracts timestamps from S3 filenames."""
    s3 = boto3.client('s3', **aws_creds) # Pass creds
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix="raw/")
    # ... rest of function is the same ...
    if 'Contents' not in objects: return pd.DataFrame()
    all_posts = []
    files_to_load = sorted(objects['Contents'], key=lambda x: x['LastModified'], reverse=True)[:50] # Load more files for trend
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
    """Loads the most recent analysis file from the 'processed/' folder."""
    s3 = boto3.client('s3', **aws_creds) # Pass creds
    # ... rest of function is the same ...
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix="processed/")
    if 'Contents' not in objects: return None
    latest_file = max(objects['Contents'], key=lambda x: x['LastModified'])
    response = s3.get_object(Bucket=bucket_name, Key=latest_file['Key'])
    analysis = json.loads(response['Body'].read().decode('utf-8'))
    return analysis
# ...(Add all other helper functions here)...
def analyze_sentiment(text):
    if not isinstance(text, str): return "Neutral"
    analysis = TextBlob(text); polarity = analysis.sentiment.polarity
    if polarity > 0.1: return "Positive"
    elif polarity < -0.1: return "Negative"
    else: return "Neutral"
def get_sentiment_score(text):
    if not isinstance(text, str): return 0.0
    return TextBlob(text).sentiment.polarity
def ask_bedrock_question(question, context_data):
    prompt = f"Context:\n{context_data}\n\nQuestion: {question}\nAnswer:"
    modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'; body = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 512,"messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]})
    response = bedrock.invoke_model(body=body, modelId=modelId); response_body = json.loads(response.get('body').read())
    return response_body.get('content')[0].get('text')
def summarize_stock_news(news_articles):
    context_data = ""; count = 0
    for article in news_articles:
        if count >= 5: break
        title = article.get('title', ''); link = article.get('link', '')
        if title: context_data += f"Title: {title}\nLink: {link}\n\n"; count += 1
    if not context_data.strip(): return "No usable news articles found."
    prompt = f"You are a financial analyst. Briefly summarize sentiment/key events from these news items:\n<news_data>{context_data}</news_data>\nSummary:"; modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'; body = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 512,"messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]})
    response = bedrock.invoke_model(body=body, modelId=modelId); response_body = json.loads(response.get('body').read())
    return response_body.get('content')[0].get('text')
@st.cache_data(ttl=300)
def get_data_for_ai_comparison(tickers_list):
    all_data = ""
    for ticker_str in tickers_list:
        try:
            ticker = yf.Ticker(ticker_str); info = ticker.info; all_data += f"\n--- {ticker_str} ({info.get('longName', 'N/A')}) ---\n"
            all_data += f"Beta: {info.get('beta', 'N/A')}\nP/E Ratio: {info.get('trailingPE', 'N/A')}\nNews:\n"
            news = ticker.news; count = 0
            if news:
                for item in news:
                    if count >= 3: break
                    title = item.get('title', '')
                    if title: all_data += f"- {title}\n"; count += 1
            if count == 0: all_data += "No recent news found.\n"
        except Exception as e: all_data += f"Could not get info for {ticker_str}. Error: {e}\n"
    return all_data
def get_comparative_ai_analysis(context_data):
    prompt = f"""You are an expert financial analyst. Based *only* on the provided data, compare the stocks. Analyze stability (Beta), valuation (P/E), news sentiment, and conclude on risk/return profile.
    <data>{context_data}</data>
    Analysis:"""
    modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'; body = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 1024,"messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]})
    response = bedrock.invoke_model(body=body, modelId=modelId); response_body = json.loads(response.get('body').read())
    return response_body.get('content')[0].get('text')


# --- Main Dashboard Layout ---

data_load_state = st.text('Loading agent data from AWS...')
df_reddit = load_raw_data_with_timestamps(S3_BUCKET_NAME)
analysis_data = load_latest_analysis(S3_BUCKET_NAME)
data_load_state.text('Data loading complete! ✅')

# --- Section 1: Live Stock Analyzer ---
st.header("Live Stock Analyzer 📈")
with st.form("stock_analyzer_form"):
    stock_symbol = st.text_input("Enter a stock ticker (e.g., TSLA, AAPL, MSFT)", "TSLA")
    time_period = st.radio("Select time period:",('5d', '1mo', '6mo', '1y', 'max'), index=2, horizontal=True, key="analyzer_time")
    submitted = st.form_submit_button("Analyze Stock")

if submitted and stock_symbol:
    try:
        ticker = yf.Ticker(stock_symbol)
        hist_chart = ticker.history(period=time_period) # Data for main chart
        hist_metric = ticker.history(period="2d")       # Data for metric

        if hist_metric.empty or hist_chart.empty: raise ValueError("Ticker may be invalid or delisted.")
        current_price = hist_metric['Close'].iloc[-1]; prev_close = hist_metric['Close'].iloc[-2] if len(hist_metric['Close']) > 1 else current_price; price_delta = current_price - prev_close

        news = ticker.news; ai_summary = "No recent news."
        if news:
            with st.spinner("Generating AI news summary..."): ai_summary = summarize_stock_news(news)

        st.subheader(f"{ticker.info.get('longName', stock_symbol.upper())} Analysis")
        col1, col2 = st.columns(2)
        with col1: st.metric(label="Current Share Price", value=f"${current_price:,.2f}", delta=f"${price_delta:,.2f} (Today)")
        with col2: st.info(f"**🧠 AI News Summary:**\n\n{ai_summary}")

        st.subheader(f"Price vs. Reddit Sentiment ({time_period})")

        # --- NEW: Sentiment Overlay Logic ---
        if not df_reddit.empty:
            # Calculate sentiment scores if not already done
            if "sentiment_score" not in df_reddit.columns:
                 df_reddit["sentiment_score"] = df_reddit["title"].apply(get_sentiment_score)

            # Ensure timestamp is datetime and make it timezone-naive for merging
            df_reddit['timestamp'] = pd.to_datetime(df_reddit['timestamp']).dt.tz_localize(None)
            hist_chart.index = hist_chart.index.tz_localize(None)

            # Resample sentiment to daily average
            daily_sentiment = df_reddit.set_index('timestamp')['sentiment_score'].resample('D').mean().dropna()

            # Combine stock history with daily sentiment
            combined_df = hist_chart.join(daily_sentiment)

            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Add Price trace
            fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df['Close'], name="Price", line=dict(color='blue')), secondary_y=False)

            # Add Sentiment trace
            fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df['sentiment_score'], name="Reddit Sentiment", line=dict(color='orange', dash='dot')), secondary_y=True)

            # Set titles
            fig.update_layout(title_text=f"{stock_symbol.upper()} Price vs. Avg Daily Reddit Sentiment")
            fig.update_yaxes(title_text="Price ($)", secondary_y=False)
            fig.update_yaxes(title_text="Sentiment Score", secondary_y=True, range=[-1, 1]) # Keep sentiment axis fixed

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No Reddit data available to overlay sentiment.")
            st.line_chart(hist_chart['Close'], use_container_width=True) # Show only price chart

    except Exception as e:
        st.error(f"Could not retrieve data for {stock_symbol}. Error: {e}")

st.divider()

# --- Section 2: Live Stock Comparator ---
st.header("Live Stock Comparator 📊")
with st.form("stock_comparator_form"):
    ticker_input = st.text_input("Enter stock tickers (comma-separated, e.g., TSLA, AAPL, MSFT)", "TSLA, GOOG")
    time_period_comp = st.radio("Select time period:", ('1mo', '6mo', '1y', '5y', 'max'), index=2, horizontal=True, key="comparator_time")
    submitted_compare = st.form_submit_button("Compare Stocks")

if submitted_compare and ticker_input:
    tickers_list = [t.strip().upper() for t in ticker_input.split(',')]
    try:
        hist_comp = yf.download(tickers_list, period=time_period_comp)['Close']
        if hist_comp.empty: raise ValueError("No data returned.")
        normalized_hist = (hist_comp / hist_comp.iloc[0] - 1) * 100
        st.subheader(f"Normalized Price Chart Comparison (% Change, {time_period_comp})")
        st.line_chart(normalized_hist, use_container_width=True)
        with st.spinner("Generating AI comparison..."):
            comparison_context = get_data_for_ai_comparison(tickers_list)
            ai_comparison = get_comparative_ai_analysis(comparison_context)
            st.subheader("🤖 AI Comparative Analysis (Risk vs. Return)")
            st.info(ai_comparison)
    except Exception as e: st.error(f"Could not retrieve data for comparison. Error: {e}")

st.divider()

# --- Section 3: AI Risk Assessment (from background agent) ---
st.header("Reddit/News Sentiment Risk Assessment (Background Agent)")
if analysis_data:
    alert_type = analysis_data.get("alert_type", "N/A"); color = "red" if alert_type == "High Risk" else ("orange" if alert_type == "Medium Risk" else "green")
    st.subheader(f"Overall Market Alert Status: :{color}[{alert_type}]")
    col1, col2 = st.columns(2)
    with col1: st.metric(label="Overall Sentiment Trend", value=analysis_data.get("overall_sentiment_trend", "N/A")) # Updated key name
    with col2: st.info(f"**🧠 Agent Reasoning (Reddit/News):**\n\n{analysis_data.get('reasoning', 'No reasoning provided.')}")
else: st.warning("No background agent analysis found yet.")

st.divider()

# --- Section 4: Interactive Q&A (about Reddit/News data) ---
st.header("Ask the Agent About Background Data")
with st.form("qa_form"):
    user_question = st.text_input("E.g., 'Summarize the main topics discussed recently.'")
    submitted_qa = st.form_submit_button("Ask Agent")
if submitted_qa and user_question and not df_reddit.empty:
    with st.spinner("Thinking..."):
        context_data = "\n".join(df_reddit['title'].dropna().astype(str) + ": " + df_reddit['text'].dropna().astype(str))
        ai_answer = ask_bedrock_question(user_question, context_data[:15000])
        st.info(f"**🤖 Agent's Answer:**\n\n{ai_answer}")

st.divider()

# --- Section 5: Background Data Visualizations ---
st.header("Background Data Visualizations (Reddit/News)")
if not df_reddit.empty:
    df_reddit_filtered = df_reddit[df_reddit['title'].apply(lambda x: isinstance(x, str))]
    if not df_reddit_filtered.empty:
        df_reddit_filtered["sentiment_label"] = df_reddit_filtered["title"].apply(analyze_sentiment)
        df_reddit_filtered["sentiment_score"] = df_reddit_filtered["title"].apply(get_sentiment_score)
        st.subheader("Sentiment Trend Over Time (Background Data)")
        df_reddit_filtered['timestamp'] = pd.to_datetime(df_reddit_filtered['timestamp']).dt.tz_localize(None) # Make timezone naive
        time_series_df = df_reddit_filtered.set_index('timestamp')['sentiment_score'].resample('15Min').mean().dropna() # Resample to 15min
        if not time_series_df.empty: st.line_chart(time_series_df)
        else: st.warning("Not enough data for trend.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment Distribution")
            sentiment_counts = df_reddit_filtered['sentiment_label'].value_counts()
            if not sentiment_counts.empty:
                fig_pie = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index)
                st.plotly_chart(fig_pie, use_container_width=True)
            else: st.warning("No sentiment labels.")
        with col2:
            st.subheader("Trending Topics Word Cloud")
            text = " ".join(title for title in df_reddit_filtered.title.dropna())
            if text:
                wordcloud = WordCloud(width=800, height=400, background_color=None, colormap='viridis').generate(text)
                fig_wc, ax = plt.subplots(); ax.imshow(wordcloud, interpolation='bilinear'); ax.axis("off"); st.pyplot(fig_wc)
            else: st.warning("No text for word cloud.")
    else: st.warning("No valid text data for visualization.")
    st.subheader("Latest Raw Background Data")
    st.dataframe(df_reddit)
else: st.warning("No raw data found in S3.")