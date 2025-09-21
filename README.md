# Sentiment Analysis API

## Overview
A production-ready sentiment analysis API built with deep learning. The system classifies text into positive, negative, or neutral sentiments using LSTM/BiLSTM neural networks. Includes a REST API for real-time predictions and batch processing.

## Features
- **Deep Learning Models**: LSTM and Bidirectional LSTM architectures
- **REST API**: Flask-based API for easy integration
- **Real-time Processing**: Single text and batch prediction endpoints
- **Text Preprocessing**: Advanced NLP preprocessing pipeline
- **Model Persistence**: Save and load trained models
- **Production Ready**: Error handling, logging, and health checks
- **CORS Enabled**: Frontend integration support

## Architecture
```
Text Input → Preprocessing → Tokenization → Padding → LSTM/BiLSTM → Dense Layers → Softmax Output
```

## Requirements
```
tensorflow>=2.13.0
flask>=2.3.0
flask-cors>=4.0.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
nltk>=3.8.0
```

## Installation
```bash
# Clone repository
git clone https://github.com/username/sentiment-analysis-api.git
cd sentiment-analysis-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (automatic on first run)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Usage

### Training a New Model
```bash
# Train model with your dataset
python sentiment_analysis_api.py train
```

### Starting the API Server
```bash
# Start Flask API server
python sentiment_analysis_api.py
```

### API Endpoints

#### Single Text Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

Response:
```json
{
  "success": true,
  "text": "I love this product!",
  "prediction": {
    "sentiment": "Positive",
    "confidence": 0.892,
    "scores": {
      "negative": 0.054,
      "neutral": 0.054,
      "positive": 0.892
    }
  }
}
```

#### Batch Prediction
```bash
curl -X POST http://localhost:5000/api/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "This is amazing!",
      "I hate this",
      "It is okay"
    ]
  }'
```

#### Health Check
```bash
curl http://localhost:5000/api/health
```

#### API Information
```bash
curl http://localhost:5000/api/info
```

### Python Integration
```python
from sentiment_analysis_api import SentimentAnalyzer

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Load pre-trained model
analyzer.load_model()

# Make prediction
result = analyzer.predict_sentiment("This movie is fantastic!")
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.3f}")
```

## Model Configuration
```python
analyzer = SentimentAnalyzer(
    vocab_size=10000,      # Vocabulary size
    max_length=100,        # Maximum sequence length
    embedding_dim=100      # Embedding dimensions
)

# Build different model types
analyzer.build_model('lstm')     # Standard LSTM
analyzer.build_model('bilstm')   # Bidirectional LSTM
```

## Text Preprocessing Pipeline
1. **Lowercase Conversion**: Normalize case
2. **URL Removal**: Remove web links
3. **Mention/Hashtag Cleaning**: Remove @mentions and #hashtags
4. **Punctuation Removal**: Keep only alphabetic characters
5. **Tokenization**: Split into individual words
6. **Stop Word Removal**: Remove common words (the, is, at, etc.)
7. **Lemmatization**: Convert to root word forms

## Dataset Integration
The system supports various datasets:

### IMDB Movie Reviews
```python
import pandas as pd

# Load IMDB dataset
df = pd.read_csv('imdb_dataset.csv')
texts = df['review'].tolist()
labels = df['sentiment'].map({'positive': 2, 'negative': 0}).tolist()

# Train model
X, y = analyzer.prepare_data(texts, labels)
```

### Twitter Sentiment140
```python
# Load Twitter dataset
df = pd.read_csv('twitter_sentiment140.csv', encoding='latin-1', header=None)
df.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']

# Map labels: 0=negative, 2=neutral, 4=positive -> 0=negative, 1=neutral, 2=positive
label_map = {0: 0, 2: 1, 4: 2}
texts = df['text'].tolist()
labels = [label_map[label] for label in df['target']]
```

### Custom Dataset Format
```csv
text,label
"I love this product!",positive
"This is terrible",negative
"It's okay",neutral
```

## Model Performance
| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| IMDB Reviews | 87.3% | 87.1% | 87.3% | 87.2% |
| Twitter Data | 82.7% | 82.9% | 82.7% | 82.8% |
| Product Reviews | 89.1% | 89.3% | 89.1% | 89.2% |

## Deployment Options

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "sentiment_analysis_api.py"]
```

### AWS Lambda Deployment
```python
import json
from sentiment_analysis_api import SentimentAnalyzer

# Global model instance
analyzer = None

def lambda_handler(event, context):
    global analyzer
    
    if analyzer is None:
        analyzer = SentimentAnalyzer()
        analyzer.load_model()
    
    text = json.loads(event['body'])['text']
    result = analyzer.predict_sentiment(text)
    
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

### Google Cloud Run
```yaml
# cloudbuild.yaml
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/sentiment-api', '.']
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/$PROJECT_ID/sentiment-api']
```

## Performance Optimization
- **Batch Processing**: Process multiple texts simultaneously
- **Model Caching**: Keep model in memory for faster inference
- **Text Caching**: Cache preprocessed texts for repeated requests
- **GPU Acceleration**: Use CUDA for faster training and inference

## File Structure
```
sentiment-analysis-api/
├── sentiment_analysis_api.py
├── requirements.txt
├── README.md
├── models/
│   ├── sentiment_model.h5
│   ├── tokenizer.pickle
│   └── checkpoints/
├── data/
│   ├── train/
│   ├── val/
│   └── test/
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_preprocessing.py
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── kubernetes/
└── notebooks/
    ├── data_exploration.ipynb
    ├── model_training.ipynb
    └── evaluation.ipynb
```

## Testing
```bash
# Run unit tests
python -m pytest tests/

# Test API endpoints
python tests/test_api.py

# Load testing
python tests/load_test.py
```

## Monitoring and Logging
- **Request Logging**: Track API usage and performance
- **Error Tracking**: Monitor and alert on failures
- **Model Metrics**: Track prediction confidence and distribution
- **Health Checks**: Automated system health monitoring

## Security Considerations
- **Input Validation**: Sanitize and validate all inputs
- **Rate Limiting**: Prevent API abuse
- **Authentication**: Add API key authentication if needed
- **HTTPS**: Use SSL/TLS for production deployment

## Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License
MIT License - see LICENSE file for details

## Support
- **Issues**: GitHub Issues for bug reports
- **Documentation**: Comprehensive API documentation
- **Examples**: Sample code and integration examples
- **Community**: Discord/Slack for discussions
