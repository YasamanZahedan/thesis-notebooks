import os
import json
import time
import pandas as pd
from collections import Counter
from tqdm import tqdm
from groq import Groq

# 1. SETUP CLOUD API & PATHS

GROQ_API_KEY = "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" 

client = Groq(api_key=GROQ_API_KEY)

COMMENTS_DIR = "Thesis_Data/comments" 
CSV_PATH = "videos_for_thesis.csv"
OUTPUT_PATH = "videos_with_sentiment_labels.csv"

# 2. LOAD DATA
try:
    df_final = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"Error: Could not find {CSV_PATH}.")
    exit()

# 3. HELPER FUNCTION (With Rate Limit Protection)
def analyze_comment(comment):
    prompt = f"""
You are an expert sentiment analysis model for YouTube comments on Food & Beverage advertisements.

Task:
Classify the sentiment of the comment toward the product, brand, or advertisement.

Labels:
- Positive: praise, craving, desire to buy, enjoyment
- Negative: dislike, criticism, disgust, complaints (e.g., price, health, ad quality)
- Neutral: questions, factual statements, spam, or unclear sentiment

Rules:
- Handle slang, emojis, and sarcasm.
- If mixed sentiment, choose the dominant tone.
- If irrelevant or spam, output Neutral.
- Do NOT explain.

Comment: "{comment}"

Output:
Return EXACTLY ONE WORD: Positive, Negative, or Neutral
"""
    
    while True:
        try:
            # Send the comment to Groq's Llama 3.1 8B
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a strict data categorizer."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.001,  # Low temperature = strict, predictable answers
                max_tokens=4
            )
            
            # Clean and return the output
            answer = chat_completion.choices[0].message.content.lower().strip()
            if "positive" in answer: return "Positive"
            elif "negative" in answer: return "Negative"
            else: return "Neutral"

        except Exception as e:
            # If we hit Groq's 30 Requests-Per-Minute limit, pause for 5 seconds and try again
            if "429" in str(e) or "rate limit" in str(e).lower():
                time.sleep(5)
            else:
                print(f"\nUnexpected Error: {e}")
                return "Neutral"

# 4. MAIN EXECUTION LOOP
video_final_labels = []

print(f"Processing {df_final['video_id'].nunique()} videos via Groq API...")
for video_id in tqdm(df_final['video_id'].unique()):
    file_path = os.path.join(COMMENTS_DIR, f"{video_id}.json")
    
    if not os.path.exists(file_path):
        continue
        
    with open(file_path, "r", encoding="utf-8") as f:
        comments = json.load(f)
        
    if not comments:
        continue
        
    # Process comments
    sentiment_labels = []
    for comment in comments:
        sentiment = analyze_comment(comment)
        sentiment_labels.append(sentiment)
        
        # Artificial delay to perfectly respect Groq's 30 Requests/Minute limit
        time.sleep(2.1) 
    
    # Signal-Weighted Majority Voting Logic (Handling Ties)
    label_counts = Counter(sentiment_labels)
    top_two = label_counts.most_common(2)
    
    if len(top_two) == 0:
        majority_label = "Neutral" 
    elif len(top_two) == 1:
        majority_label = top_two[0][0] 
    elif top_two[0][1] == top_two[1][1]:
        # TIE-BREAKER LOGIC:
        tied_labels = {top_two[0][0], top_two[1][0]}
        
        if tied_labels == {"Positive", "Neutral"}:
            majority_label = "Positive" # Positive beats Neutral noise
        elif tied_labels == {"Negative", "Neutral"}:
            majority_label = "Negative" # Negative beats Neutral noise
        else:
            majority_label = "Neutral"  # A tie between Positive and Negative is polarizing (Neutral)
    else:
        majority_label = top_two[0][0]
    
    video_final_labels.append({
        "video_id": video_id,
        "majority_sentiment": majority_label,
        "positive_count": label_counts.get("Positive", 0),
        "negative_count": label_counts.get("Negative", 0),
        "neutral_count": label_counts.get("Neutral", 0),
        "total_analyzed": len(sentiment_labels)
    })
    
    # Save a backup after EVERY video. 
    pd.DataFrame(video_final_labels).to_csv("backup_labels.csv", index=False)

# 5. SAVE FINAL COMBINED RESULTS
df_labels = pd.DataFrame(video_final_labels)

if df_labels.empty:
    print("\nERROR: No comments were processed! Check your COMMENTS_DIR path.")
else:
    df_merged = pd.merge(df_final, df_labels, on="video_id", how="inner")
    df_merged.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFinished! Results saved to {OUTPUT_PATH}.")