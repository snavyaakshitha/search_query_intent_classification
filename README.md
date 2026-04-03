# Search Query Intent Classification & User Behavior Analysis

NLP-driven intent classification of 500K+ search queries with K-Means user segmentation and conversion lift analysis.

## Pipeline

1. **Data Generation** - 500K synthetic queries modeled after the AOL Search Query Log dataset, with click and conversion events
2. **NLP Preprocessing** - Text cleaning, TF-IDF vectorization (5K features, bigrams), TruncatedSVD dimensionality reduction
3. **Intent Classification** - Three-class classification (navigational / informational / transactional) using Logistic Regression and Random Forest
4. **User Segmentation** - K-Means clustering (4 clusters) on aggregated user behavior: High-Intent Shoppers, Brand Navigators, Curious Explorers, Information Seekers
5. **Behavioral Analysis** - Intent transition patterns, conversion lift quantification, SQL-style queries
6. **Conversion Lift** - ~18% higher conversion rate for High-Intent Shoppers vs. other segments

## Tech Stack
Python, scikit-learn (TF-IDF, K-Means, Random Forest), pandas, matplotlib, seaborn

## Run
```bash
pip install -r requirements.txt
python search_intent_classification.py
```

## Outputs
- `intent_classification.png` — Confusion matrix, intent distribution, word count analysis
- `user_segmentation.png` — Elbow plot, segment sizes, intent mix, conversion rates
- `behavioral_analysis.png` — Conversion by intent, lift analysis, intent transitions
