import json
import csv
import os

# --- Update these paths to match your local setup ---
BUSINESS_JSON = './data/business.json'
REVIEW_JSON = './data/review.json'
OUTPUT_CSV = './data/user_business.csv'

# Step 1: Filter businesses in Arizona (state == "AZ")
az_business_ids = set()
with open(BUSINESS_JSON, 'r', encoding='utf-8') as f:
    for line in f:
        biz = json.loads(line)
        if biz.get('state') == 'AZ':
            az_business_ids.add(biz['business_id'])

print(f"Found {len(az_business_ids)} businesses in Arizona.")

# Step 2: Select (user_id, business_id) from reviews where business is in AZ
# Step 3: Write to CSV with unique pairs
seen_pairs = set()
rows = []

with open(REVIEW_JSON, 'r', encoding='utf-8') as f:
    for line in f:
        review = json.loads(line)
        bid = review['business_id']
        uid = review['user_id']
        if bid in az_business_ids:
            pair = (uid, bid)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                rows.append(pair)

print(f"Found {len(rows)} unique (user, business) pairs from AZ reviews.")

# Write CSV
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['user_id', 'business_id'])
    for uid, bid in rows:
        writer.writerow([uid, bid])

print(f"Wrote {OUTPUT_CSV}")