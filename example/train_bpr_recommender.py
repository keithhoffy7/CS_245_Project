"""
Train a Bayesian Personalized Ranking (BPR) model for recommendation.
BPR is optimized for ranking tasks with implicit feedback.
"""

import json
import pickle
import numpy as np
from scipy.sparse import csr_matrix
import implicit
from collections import defaultdict
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

# Configuration
REVIEW_FILE = "/srv/output/data1/output/review.json"
OUTPUT_FILE = "/srv/CS_245_Project/example/bpr_model.pkl"

# BPR hyperparameters
FACTORS = 128  # Embedding dimension
REGULARIZATION = 0.01  # L2 regularization
ITERATIONS = 50  # Number of training iterations
LEARNING_RATE = 0.05  # Learning rate

# Filter for positive interactions (ratings >= 4.0)
MIN_RATING = 4.0


def load_reviews():
    """Load reviews and filter for positive interactions."""
    logging.info(f"Loading reviews from {REVIEW_FILE}...")
    
    user_item_ratings = defaultdict(list)  # user_id -> [(item_id, rating), ...]
    user_set = set()
    item_set = set()
    
    with open(REVIEW_FILE, 'r') as f:
        for line in tqdm(f, desc="Reading reviews"):
            try:
                review = json.loads(line)
                user_id = review.get('user_id')
                item_id = review.get('item_id')
                rating = review.get('stars') or review.get('rating')
                
                if not user_id or not item_id or not rating:
                    continue
                
                try:
                    rating = float(rating)
                except (ValueError, TypeError):
                    continue
                
                # Filter for positive interactions
                if rating >= MIN_RATING:
                    user_item_ratings[user_id].append((item_id, rating))
                    user_set.add(user_id)
                    item_set.add(item_id)
            except json.JSONDecodeError:
                continue
    
    logging.info(f"Loaded {len(user_item_ratings)} users with positive interactions")
    logging.info(f"Total unique users: {len(user_set)}, items: {len(item_set)}")
    
    return user_item_ratings, sorted(user_set), sorted(item_set)


def build_interaction_matrix(user_item_ratings, user_list, item_list):
    """Build sparse user-item interaction matrix."""
    logging.info("Building interaction matrix...")
    
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_list)}
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_list)}
    
    rows = []
    cols = []
    data = []
    
    for user_id, interactions in tqdm(user_item_ratings.items(), desc="Building matrix"):
        if user_id not in user_id_to_idx:
            continue
        
        user_idx = user_id_to_idx[user_id]
        for item_id, rating in interactions:
            if item_id not in item_id_to_idx:
                continue
            
            item_idx = item_id_to_idx[item_id]
            rows.append(user_idx)
            cols.append(item_idx)
            # Use rating as weight (or 1.0 for binary)
            data.append(float(rating))
    
    # Create sparse matrix (user-item)
    matrix = csr_matrix((data, (rows, cols)), shape=(len(user_list), len(item_list)))
    
    logging.info(f"Matrix shape: {matrix.shape}, non-zero entries: {matrix.nnz}")
    
    return matrix, user_id_to_idx, item_id_to_idx


def train_bpr_model(matrix):
    """Train BPR model using implicit library."""
    logging.info("Training BPR model...")
    logging.info(f"Hyperparameters: factors={FACTORS}, regularization={REGULARIZATION}, "
                 f"iterations={ITERATIONS}, learning_rate={LEARNING_RATE}")
    
    # Initialize BPR model
    model = implicit.bpr.BayesianPersonalizedRanking(
        factors=FACTORS,
        regularization=REGULARIZATION,
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        num_threads=4
    )
    
    # Train the model
    # Note: implicit expects item-user matrix (transposed)
    # But BPR internally handles this correctly
    model.fit(matrix.T)  # Transpose to item-user format (as implicit expects)
    
    logging.info("Training completed!")
    
    return model


def extract_factors(model, user_list, item_list):
    """
    Extract user and item factors from the trained model.
    
    IMPORTANT: implicit.bpr uses confusing naming:
    - When you pass item-user matrix (transposed), model.user_factors are actually ITEM factors
    - And model.item_factors are actually USER factors
    So we need to swap them!
    """
    logging.info("Extracting factors...")
    
    # IMPORTANT: Due to implicit's naming convention when using transposed matrix:
    # model.user_factors are actually item factors (because we passed item-user matrix)
    # model.item_factors are actually user factors
    # So we swap them to match our user-item convention
    
    item_factors = model.user_factors  # Actually item factors (confusing naming!)
    user_factors = model.item_factors  # Actually user factors (confusing naming!)
    
    logging.info(f"User factors shape: {user_factors.shape}")
    logging.info(f"Item factors shape: {item_factors.shape}")
    
    return user_factors, item_factors


def main():
    """Main training pipeline."""
    logging.info("Starting BPR model training...")
    
    # Step 1: Load reviews
    user_item_ratings, user_list, item_list = load_reviews()
    
    # Step 2: Build interaction matrix
    matrix, user_id_to_idx, item_id_to_idx = build_interaction_matrix(
        user_item_ratings, user_list, item_list
    )
    
    # Step 3: Train BPR model
    model = train_bpr_model(matrix)
    
    # Step 4: Extract factors (with correct swapping)
    user_factors, item_factors = extract_factors(model, user_list, item_list)
    
    # Step 5: Save model
    model_dict = {
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_id_to_idx": user_id_to_idx,
        "item_id_to_idx": item_id_to_idx,
        "user_list": user_list,
        "item_list": item_list,
        "model_type": "bpr",
        "hyperparameters": {
            "factors": FACTORS,
            "regularization": REGULARIZATION,
            "iterations": ITERATIONS,
            "learning_rate": LEARNING_RATE,
            "min_rating": MIN_RATING
        }
    }
    
    logging.info(f"Saving model to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(model_dict, f)
    
    logging.info("Model saved successfully!")
    logging.info(f"Model contains {len(user_list)} users and {len(item_list)} items")
    logging.info(f"User factors shape: {user_factors.shape}, Item factors shape: {item_factors.shape}")


if __name__ == "__main__":
    main()

