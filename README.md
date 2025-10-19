🧠 MarketMind: An Autonomous AI Financial Agent

MarketMind is a serverless AI agent that reads the market's emotions. It fuses real-time financial data with public sentiment from Reddit and news headlines, using AWS Bedrock to distill complex information into actionable intelligence and risk alerts.

➤ Live Demo & Core Functionality

🚀 Access the Live Deployed Dashboard Here

(Replace the URL above with your actual Streamlit Cloud app link)

(To create this GIF: Use a free tool like ScreenToGif or Kap to record your dashboard in action. Show the live stock analyzer, the comparator, and the Q&A. Upload it to a demo folder in your GitHub repo and update the link.)

🎯 The Problem: Information Overload vs. Actionable Insight

Modern markets are driven by a constant firehose of data: price action, news headlines, and chaotic social media sentiment. Retail traders and even institutional analysts struggle to separate the signal from the noise, often reacting too late to emotional shifts that drive volatility.

MarketMind solves this by acting as a digital analyst that never sleeps. It perceives, reasons, and acts on market sentiment, providing a quantifiable edge by understanding the psychology behind the price.

✨ Key Features: A Dual-Agent System

MarketMind operates as a sophisticated system of two distinct AI agents, each with a unique purpose:

1. The Sentinel Agent (Proactive, Background Analysis)

This is a serverless, autonomous agent running 24/7 on AWS.

Perceives: Automatically runs on an hourly schedule (EventBridge) to ingest data streams from Reddit and News APIs (Lambda).

Reasons: Uses Amazon Bedrock (Claude 3) to analyze the combined data, identify overall market sentiment, detect specific stock trends, and generate a detailed risk assessment.

Acts: Saves its analysis to a data lake (S3) and fires Telegram alerts for "High" or "Medium" risk events.

2. The Oracle Agent (Reactive, Real-Time Analysis)

This is an interactive, on-demand agent living inside the Streamlit dashboard.

Listens: Waits for a direct command from a user (e.g., "Compare TSLA and GOOG").

Reasons: Fetches live financial data (yfinance) and news for the requested stocks, then uses Amazon Bedrock to perform a deep, comparative analysis of risk, return, stability, and valuation.

Responds: Instantly delivers a custom-tailored report with charts, metrics, and generative AI insights, including price predictions and risk scores.

⚙️ System Architecture

The entire system is orchestrated on a modern, serverless AWS stack, ensuring scalability, reliability, and cost-efficiency.

(Upload your architecture diagram to a demo folder and update this link.)

🔧 Tech Stack

Cloud Platform: AWS

AI Reasoning Engine: Amazon Bedrock (Anthropic Claude 3 Sonnet)

Serverless Compute: AWS Lambda

Data Lake & Storage: Amazon S3

Scheduling: Amazon EventBridge

Permissions: AWS IAM

Dashboard: Streamlit & Plotly

Data Sources: Reddit API, News API, yfinance (Yahoo Finance)

Notifications: Telegram Bot API

Forecasting: Prophet (by Meta)

🚀 Getting Started

To run the dashboard locally, follow these steps:

1. Prerequisites

Python 3.9+

Git

An AWS account with credentials configured locally via the AWS CLI.

API keys for News API and a Telegram Bot Token & Chat ID.

2. Clone & Install

# Clone the repository
git clone [https://github.com/devils-advocate1/MarketMind-AI-Agent.git](https://github.com/devils-advocate1/MarketMind-AI-Agent.git)
cd MarketMind-AI-Agent

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all required libraries
pip install -r requirements.txt

# Download NLP corpora for TextBlob
python -m textblob.download_corpora


3. Run the Dashboard

streamlit run dashboard.py


The application will open in your web browser. Note that for the dashboard to connect to AWS, your local machine must be configured with AWS credentials.

آینده (Future Roadmap)

Hyper-Personalization: Allow users to define their own portfolios and receive alerts only for stocks they own.

Advanced Technical Indicators: Integrate TA-Lib to include technical indicators like RSI, MACD, and Bollinger Bands into the AI's reasoning context.

Multi-Modal Analysis: Incorporate analysis of images and memes from Reddit to capture visual sentiment trends.

👨‍💻 Created By

Ramij Raj

GitHub

LinkedIn (<-- Add your LinkedIn link here)
