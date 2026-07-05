"""
Grabs Amazon 2023 reviews for some categories and saves an equal number
of each star rating (1-5) to data/reviews.jsonl.
"""

from pathlib import Path
import argparse
from datasets import load_dataset, Dataset, concatenate_datasets, Value

# paths are based off the repo folder so it runs from anywhere
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / 'data' / 'raw'
FILE = DATA_ROOT / 'reviews.jsonl'  # the full dataset before splitting
# the three split files we save for training/dev/testing
TRAIN_FILE = DATA_ROOT / 'train.jsonl'
DEV_FILE = DATA_ROOT / 'dev.jsonl'
TEST_FILE = DATA_ROOT / 'test.jsonl'

# all the category names in the dataset
CATEGORIES = [
    "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing",
    "Automotive", "Baby_Products", "Beauty_and_Personal_Care", "Books",
    "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry",
    "Digital_Music", "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food",
    "Handmade_Products", "Health_and_Household", "Health_and_Personal_Care",
    "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store",
    "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments",
    "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies", "Software",
    "Sports_and_Outdoors", "Subscription_Boxes", "Tools_and_Home_Improvement",
    "Toys_and_Games", "Unknown", "Video_Games",
]

SEED = 42  # same seed every time so we get the same data
BASE = "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories"
RATINGS = [1.0, 2.0, 3.0, 4.0, 5.0]  # the 5 possible star ratings


def sentiment(rating):
    # turn the star rating into a sentiment label
    # 1-2 = negative, 3 = neutral, 4-5 = positive
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"


def parse_args():
    # read the two settings from the command line
    args = argparse.ArgumentParser()
    # how many categories to use (leave empty to use all of them)
    args.add_argument("--review_category_count", type=int, default=len(CATEGORIES))
    # how many reviews per category (gets split evenly over the 5 ratings)
    args.add_argument("--number_of_reviews_each_category", type=int, default=5000)

    return args.parse_args()

def main():
    args = parse_args()
    N_CATEGORIES = int(args.review_category_count)
    N_PER_CATEGORY = int(args.number_of_reviews_each_category)

    parts = []  # holds one dataset per category, joined together later


    # go through each category one at a time
    for cat_i in range(N_CATEGORIES):

        # stream the file instead of downloading the whole thing, the
        # category files are too big to fit in memory
        stream = load_dataset(
            "json",
            data_files=f"{BASE}/{CATEGORIES[cat_i]}.jsonl",
            split="train",
            streaming=True
        )

        # mix up the order so we don't just grab the first rows, and
        # only keep reviews that are verified purchases
        stream = stream.shuffle(seed=SEED, buffer_size=20_000)
        stream = stream.filter(lambda r: r["verified_purchase"])

        # make a bucket for each rating and fill them up equally. we loop
        # through the stream because there are way more 5 star reviews, so
        # we have to keep going until the rare ratings are also full
        per_rating = N_PER_CATEGORY // len(RATINGS)
        buckets = {r: [] for r in RATINGS}
        for r in stream:
            rating = float(r["rating"])
            # only add it if that rating's bucket still has room
            if rating in buckets and len(buckets[rating]) < per_rating:
                # add the sentiment label as an extra column
                buckets[rating].append({
                    "rating": rating,
                    "title": r["title"],
                    "sentiment": sentiment(rating),
                })
                # stop once every bucket is full
                if all(len(b) == per_rating for b in buckets.values()):
                    break

        # flatten all the buckets into one list of rows
        rows = [row for b in buckets.values() for row in b]
        parts.append(Dataset.from_list(rows))
        counts = {r: len(b) for r, b in buckets.items()}
        print(f"loaded {CATEGORIES[cat_i]} data  {counts}")

    # combine every category and shuffle it all together
    dataset = concatenate_datasets(parts).shuffle(seed=SEED)

    # some categories might run short, so cut every rating down to the
    # smallest one's count to make it perfectly balanced
    by_rating = {r: dataset.filter(lambda x, r=r: float(x["rating"]) == r) for r in RATINGS}
    min_count = min(len(d) for d in by_rating.values())
    dataset = concatenate_datasets(
        [d.select(range(min_count)) for d in by_rating.values()]
    ).shuffle(seed=SEED)

    print(f"\nFinal dataset: {len(dataset)} reviews from {len(CATEGORIES)} categories")
    print(f"Rating distribution: {min_count} reviews each for {RATINGS}")
    print(dataset[0])

    # make the data folder if it isn't there, then save one review per line
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    dataset.to_json(FILE, force_ascii=False)
    print(f"Saved dataset to {FILE}")

    # split into train / dev / test = 75% / 10% / 15%
    # stratify needs the column to be a ClassLabel, so encode sentiment into
    # numbers first (we turn it back into text before saving)
    dataset = dataset.class_encode_column("sentiment")

    # first pull out 15% for test, then split the rest into dev and train.
    # stratify_by_column keeps the sentiment balance the same in each split
    split1 = dataset.train_test_split(
        test_size=0.15, seed=SEED, stratify_by_column="sentiment"
    )
    test = split1["test"]
    # of the remaining 85%, dev should be 10% of the whole -> 10/85
    split2 = split1["train"].train_test_split(
        test_size=0.10 / 0.85, seed=SEED, stratify_by_column="sentiment"
    )
    train = split2["train"]
    dev = split2["test"]

    # put the sentiment label back to text, then save each split to its own file.
    # cast_column turns the ClassLabel back into a plain string, otherwise
    # to_json would write the encoded number (0/1/2) instead of the word.
    for split, path in [(train, TRAIN_FILE), (dev, DEV_FILE), (test, TEST_FILE)]:
        split = split.cast_column("sentiment", Value("string"))
        split.to_json(path, force_ascii=False)
    print(f"train: {len(train)}  dev: {len(dev)}  test: {len(test)}")



if __name__ =="__main__":
    main()
