from feedback import process_all
import json


import pandas as pd


file_path = 'reviews_0-250_masked.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display the column names to help you identify the correct one
print("Available columns:", df.columns.tolist())

# Replace 'review' with the actual column name that contains the text reviews
text_reviews = df['review_text']  # or df['feedback'] or the actual column name

# Print the extracted text reviews
print(text_reviews)

# Optionally, you can convert them to a list
review_list = text_reviews.tolist()



# Process the feedbacks
results = process_all(review_list)

# Print results

for feedback in results:
    with open('result.json', 'w',encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
print("✓ Results saved to result.json")

