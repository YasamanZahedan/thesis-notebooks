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

# 2. LOAD YOUR DATA
try:
    df_final = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"Error: Could not find {CSV_PATH}.")
    exit()

# 3. HELPER FUNCTION (With Rate Limit Protection)
def (comment):
    prompt = f"You are a sentiment analyzer. Classify this comment: '{comment}'. Respond with EXACTLY ONE WORD: Positive, Negative, or Neutral. Output nothing else."
    
    while True:
        try:
            # Send the comment to Groq's Llama 3.1 8B
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a strict data categorizer."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.0,  # Low temperature = strict, predictable answers
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

print(f"Processing {len(df_final)} videos via Groq API...")
for video_id in tqdm(df_final['video_id']):
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
    
    # Majority voting logic
    label_counts = Counter(sentiment_labels)
    majority_label = label_counts.most_common(1)[0][0] 
    
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
df_merged = pd.merge(df_final, df_labels, on="video_id", how="inner")
df_merged.to_csv(OUTPUT_PATH, index=False)

print(f"\nFinished! Results saved to {OUTPUT_PATH}.")