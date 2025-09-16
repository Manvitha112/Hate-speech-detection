import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import emoji
from datetime import datetime
from textblob import TextBlob
import urllib.parse
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model = None
try:
    logger.info('Loading model...')
    # Go up one directory from the backend folder to get to the root
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root_dir, 'hate_speech_model.pkl')
    logger.info(f'Looking for model at: {model_path}')
    
    if not os.path.exists(model_path):
        logger.error(f'Model file not found at: {model_path}. Current working directory: {os.getcwd()}. Files in root: {os.listdir(root_dir)}')
    else:
        logger.info('Model file found, loading...')
        model = joblib.load(model_path)
        logger.info('Model loaded successfully')
        logger.info('Model loaded successfully')
except Exception as e:
    logger.error(f'Error loading model: {str(e)}')

@app.route('/')
def home():
    return "Hate Speech Detection API is running."

def clean_social_media_text(text):
    # Store hashtags
    hashtags = re.findall(r'#\w+', text)
    # Store mentions
    mentions = re.findall(r'@\w+', text)
    # Store URLs
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    
    # Convert emojis to text
    text = emoji.demojize(text)
    
    # Handle repeated characters (e.g., 'hateeeee' -> 'hate')
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # Handle common social media abbreviations
    social_abbr = {
        'af': 'as fuck', 'tbh': 'to be honest', 'imo': 'in my opinion',
        'idk': 'i dont know', 'wtf': 'what the fuck', 'tf': 'the fuck',
        'stfu': 'shut the fuck up', 'gtfo': 'get the fuck out',
        'mf': 'motherfucker', 'kys': 'kill yourself'
    }
    
    for abbr, full in social_abbr.items():
        text = re.sub(r'\b' + abbr + r'\b', full, text.lower())
    
    return text, hashtags, mentions, urls

def preprocess_text(text):
    # First clean social media specific content
    text, hashtags, mentions, urls = clean_social_media_text(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Handle extra whitespace
    text = ' '.join(text.split())
    
    # Simple tokenization by splitting on whitespace
    tokens = text.split()
    
    # Basic stopwords list
    basic_stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it',
                       'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'}
    
    # Keep important negative words
    negative_words = {'no', 'not', 'nor', 'none', 'never', 'against'}
    stop_words = basic_stop_words - negative_words
    
    # Filter out stopwords
    tokens = [t for t in tokens if t not in stop_words]
    
    # Basic word normalization (just lowercase is enough for our purpose)
    tokens = [t.lower() for t in tokens]
    
    return ' '.join(tokens), hashtags, mentions, urls

import re

# Define word lists and phrases
HATE_PHRASES = {
    # Identity-based hate
    r'\b(i|we|they)\s+hate\s+\w+\b',
    r'\b\w+\s+should\s+(die|burn|suffer)\b',
    r'\bkill\w*\s+\w*\b',  # Matches kill, killer, killing, etc.
    
    # Threats and violent expressions
    r'\b(going|gonna)\s+to\s+(kill|hurt|beat)\b',
    r'\bwish\s+\w+\s+(would\s+)?(die|disappear)\b',
    r'\b(murder|assault|attack)\w*\b',  # Matches variations of violent words
    
    # Offensive identity-based phrases
    r'\b\w+\s+people\s+are\s+(stupid|dumb|bad|evil)\b',
    r'\ball\s+\w+\s+are\b',
    
    # Personal attacks and direct insults
    r'\byou\s+(are|r|re)\s+(stupid|dumb|idiot|ugly|bitch|asshole|bastard)\b',
    r'\bi\s+hope\s+you\s+(die|suffer|burn)\b',
    r'\byou\s+\w*\s*(bitch|asshole|bastard|idiot|stupid|dumb)\b',
    r'\b(bitch|asshole|bastard|idiot|stupid|dumb)\b',
    
    # Violent words and their variations
    r'\b(kill|murder|assault|attack|beat|hurt)\w*\b'
}

# Define negation words that can flip the meaning
NEGATION_WORDS = {'not', 'never', 'no', 'none', 'neither', 'nowhere', 'nothing'}

def check_hate_phrases(text):
    text = text.lower()
    
    # Check for negation context
    words = text.split()
    has_negation = any(neg in words for neg in NEGATION_WORDS)
    
    # Check for hate phrases
    for pattern in HATE_PHRASES:
        if re.search(pattern, text):
            # If there's negation, it might be defending against hate
            if has_negation and ('hate' in text or 'racist' in text or 'discrimination' in text):
                continue
            return True
    
    return False

def analyze_sentiment(text):
    # Positive words that should never trigger hate speech
    POSITIVE_WORDS = {
        'good', 'great', 'awesome', 'nice', 'happy', 'love', 'wonderful', 'beautiful', 'amazing',
        'excellent', 'fantastic', 'brilliant', 'perfect', 'thank', 'thanks', 'blessed', 'joy', 'peaceful'
    }
    
    # Get text words
    text_lower = text.lower()
    words = set(text_lower.split())
    
    # If the text contains mostly positive words, it's not hate speech
    positive_count = len(words.intersection(POSITIVE_WORDS))
    if positive_count > 0 and len(words) > 0:
        positive_ratio = positive_count / len(words)
        if positive_ratio > 0.3:  # If more than 30% of words are positive
            return False
    
    # Check for hate phrases only if no strong positive sentiment
    if check_hate_phrases(text):
        return True
    
    # Common offensive words that indicate hate speech
    offensive_words = {
        'fuck', 'shit', 'bitch', 'ass', 'dick', 'damn', 'cunt', 'bastard', 'whore',
        'kill', 'killer', 'killing', 'killed', 'die', 'died', 'dying', 'dead',
        'hate', 'hated', 'hating', 'hater',
        'stupid', 'idiot', 'dumb', 'ugly', 'asshole',
        'murder', 'murderer', 'murdering', 'murdered',
        'attack', 'attacker', 'attacking', 'attacked',
        'assault', 'assaulting', 'assaulted', 'assaulter'
    }
    
    # Check for offensive words with context
    offensive_count = len(words.intersection(offensive_words))
    
    # Only consider it hate speech if there are offensive words AND no positive words
    return offensive_count > 0 and positive_count == 0

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info('Received prediction request')
        data = request.json
        logger.debug(f'Request data: {data}')
        
        if not data:
            logger.error('No JSON data received')
            return jsonify({'error': 'No input data provided'}), 400
            
        text = data.get('text', '')
        if not text:
            logger.error('No text provided')
            return jsonify({'error': 'No text provided'}), 400
            
        timestamp = data.get('timestamp', datetime.now().isoformat())
        username = data.get('username', '')
        platform = data.get('platform', 'unknown')
        reply_to = data.get('reply_to', None)
        
        logger.info(f'Processing text: {text}')
        
        # Check if model is loaded
        if model is None:
            logger.error('Model not loaded')
            return jsonify({'error': 'Model not initialized'}), 500
        
        # Process the text and get social media components
        processed_text, hashtags, mentions, urls = preprocess_text(text)
        logger.debug(f'Processed text: {processed_text}')
        logger.debug(f'Found hashtags: {hashtags}')
        logger.debug(f'Found mentions: {mentions}')
        
        # Get sentiment scores using TextBlob
        sentiment = TextBlob(text)
        sentiment_score = sentiment.sentiment.polarity
        logger.debug(f'Sentiment score: {sentiment_score}')
        
        # Analyze the content
        if processed_text.strip():
            # Get ML model prediction
            try:
                model_prediction = model.predict([processed_text])[0] if processed_text else 0
                logger.debug(f'Model prediction: {model_prediction}')
            except Exception as e:
                logger.error(f'Error in model prediction: {str(e)}')
                return jsonify({'error': 'Error in prediction'}), 500
            
            # Get rule-based sentiment analysis
            sentiment_prediction = analyze_sentiment(text)
            logger.debug(f'Sentiment prediction: {sentiment_prediction}')
            
            # Additional social media specific checks
            social_media_signals = {
                'hashtag_signal': any(analyze_sentiment(tag) for tag in hashtags),
                'mention_abuse': len(mentions) > 0 and sentiment_score < -0.5,
                'url_abuse': len(urls) > 0 and sentiment_score < -0.5,
                'excessive_mentions': len(mentions) > 5,
                'is_reply': bool(reply_to) and sentiment_score < -0.3
            }
            logger.debug(f'Social media signals: {social_media_signals}')
            
            # Combine all signals with better logic
            sentiment_threshold = -0.3  # Only consider very negative sentiment
            
            is_hate = (
                (model_prediction == 1 and sentiment_score < 0) or  # Model predicts hate and negative sentiment
                (sentiment_prediction and sentiment_score < sentiment_threshold) or  # Rule-based detection with strong negative sentiment
                (any(social_media_signals.values()) and sentiment_score < sentiment_threshold)  # Social signals with strong negative sentiment
            )
            
            # Override for clearly positive content
            if sentiment_score > 0.5:  # If sentiment is very positive
                is_hate = False
            
            result = "Hate/Offensive" if is_hate else "Not Hate Speech"
        else:
            result = "Not Hate Speech"
        
        response_data = {
            'prediction': result,
            'metadata': {
                'timestamp': timestamp,
                'platform': platform,
                'username': username,
                'reply_to': reply_to,
                'hashtags': hashtags,
                'mentions': mentions,
                'urls': urls,
                'sentiment_score': sentiment_score
            }
        }
        logger.info(f'Returning prediction: {result}')
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f'Error processing request: {str(e)}')
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
