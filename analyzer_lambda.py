import json
import os
import boto3
import urllib3


s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime')
http = urllib3.PoolManager()

def send_telegram_message(message):
    """Sends a message to the Telegram bot."""
    print("Sending message to Telegram...")
    token = os.environ['TELEGRAM_BOT_TOKEN']
    chat_id = os.environ['TELEGRAM_CHAT_ID']
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    
    http.request('POST', url, fields=payload)
    print("Message sent successfully.")

def lambda_handler(event, context):
    """
    Main handler: reads S3 file, analyzes with Bedrock, saves the analysis, and sends a Telegram alert if needed.
    """
    print("--- MarketMind Analyzer Agent: Activated ---")

    
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_key = event['Records'][0]['s3']['object']['key']
    print(f"New file detected: s3://{bucket_name}/{file_key}")

    try:
        
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        content = response['Body'].read().decode('utf-8')
        posts = json.loads(content)
        text_to_analyze = "".join([f"Title: {post['title']}\n---\n" for post in posts])
        prompt = f"""
        You are MarketMind... (rest of prompt is the same)
        Respond ONLY with a valid JSON object with three keys: "sentiment_trend", "alert_type", and "reasoning".
        Data: <data>{text_to_analyze[:10000]}</data>
        """

        
        modelId = 'anthropic.claude-3-sonnet-20240229-v1:0'
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31", "max_tokens": 512,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        })
        response = bedrock.invoke_model(body=body, modelId=modelId, accept='application/json', contentType='application/json')
        response_body = json.loads(response.get('body').read())
        ai_response_text = response_body.get('content')[0].get('text')
        
        print("--- AI Analysis Complete ---")
        print(ai_response_text)
        
        ai_decision = json.loads(ai_response_text)
        
        
        
        processed_file_key = file_key.replace("raw/raw_reddit_data_", "processed/analysis_")
        
        s3.put_object(
            Bucket=bucket_name,
            Key=processed_file_key,
            Body=json.dumps(ai_decision, indent=4)
        )
        print(f"AI analysis saved to s3://{bucket_name}/{processed_file_key}")

        
        if ai_decision.get("alert_type") == "High Risk":
            print("High risk detected. Preparing Telegram alert.")
            alert_message = (
                f"📊 *MarketMind Alert: HIGH RISK* 🚨\n\n"
                f"*Sentiment Trend:* {ai_decision.get('sentiment_trend')}\n\n"
                f"*🧠 Reasoning:* {ai_decision.get('reasoning')}"
            )
            send_telegram_message(alert_message)
        else:
            print("Neutral or low risk. No alert sent.")

        return {'statusCode': 200, 'body': json.dumps('Analysis and action phase complete!')}

    except Exception as e:
        print(f"ERROR: {e}")
        raise e