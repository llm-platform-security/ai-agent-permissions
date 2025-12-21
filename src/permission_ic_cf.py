"""
LLM Inference with Collaborative Filtering (IC+CF Hybrid)

This script implements the hybrid approach combining in-context learning (IC)
with collaborative filtering (CF) scores for automated permission decisions.

Inputs:
    - ../data/processed_dataset.json - Processed dataset (181 filtered participants)
    - ../queries.json - Study scenarios
    - cf_scores.csv - CF scores from permission_cf_only.py

Outputs:
    - Permission predictions with confidence scores using IC+CF hybrid approach
    - Saved to: ic_cf_predictions.json

Requirements:
    - OpenAI API key (set via environment variable)
    - CF scores from permission_cf_only.py must be generated first
"""

import os
import sys
import json
import random
from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

# Import unified evaluation utilities
from evaluation_utils import calculate_metrics, save_metrics, print_metrics

# Load environment variables from .env file
load_dotenv()

# Set random seed for reproducibility
random.seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Check for OpenAI API key
if 'OPENAI_API_KEY' not in os.environ:
    raise EnvironmentError(
        "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.\n"
        "You can either:\n"
        "  1. Create a .env file with: OPENAI_API_KEY=your-api-key-here\n"
        "  2. Export it: export OPENAI_API_KEY='your-api-key-here'"
    )

# Initialize OpenAI client
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Model configuration - can be overridden via environment variable
MODEL = os.getenv('OPENAI_MODEL', 'o4-mini')
logger.info(f"Using model: {MODEL}")

# Configuration
DATASET_PATH = "../data/processed_dataset.json"
QUERIES_PATH = "../queries.json"
CF_SCORES_PATH = "../results/cf_scores.csv"  # Output from permission_cf_only.py
OUTPUT_PATH = "../results/ic_cf_predictions.json"
MAX_WORKERS = 5  # Parallel processing workers
TOP_K = 5  # Top K CF recommendations to include

# Example output format for few-shot learning
EXAMPLE_OUTPUTS = '''
[
    {
        "query": "I want to lose weight. Can you create an at-home full-body workout plan for me?",
        "id": 0,
        "permission": {
            "Fitness Tracking,Fitness goal (i.e. weight loss)": {"label": "Yes, always share", "score": 0.92},
            "Fitness Tracking,Gender": {"label": "Yes, always share", "score": 0.88}
        }
    },
    {
        "query": "Can you check my latest blood test results and let me know if my blood sugar levels look stable?",
        "id": 1,
        "permission": {
            "LLM,Blood sugar data": {"label": "No, never share", "score": 0.85}
        }
    },
    {
        "query": "How has my sleep duration changed over the past week?",
        "id": 3,
        "permission": {
            "LLM,Sleep duration": {"label": "Yes, always share", "score": 0.90},
            "LLM,Sleep quality data (e.g. snoring)": {"label": "No, never share", "score": 0.80}
        }
    }
]
'''


def load_cf_scores(cf_path):
    """
    Load collaborative filtering scores.

    Args:
        cf_path: Path to CF scores CSV file

    Returns:
        DataFrame: CF scores with columns [user_id, query_id, score]
    """
    if not os.path.exists(cf_path):
        logger.warning(f"CF scores not found at {cf_path}")
        logger.warning("Run permission_cf_only.py first to generate CF scores.")
        return pd.DataFrame(columns=['user_id', 'query_id', 'score'])

    return pd.read_csv(cf_path)


def get_top_k_recommendations(participant_id, cf_scores, queries_data, k=TOP_K):
    """
    Get top-K CF recommendations for a participant in the original format.

    Args:
        participant_id: Participant ID
        cf_scores: CF scores DataFrame
        queries_data: Queries data for formatting recommendations
        k: Number of top recommendations

    Returns:
        str: Formatted CF recommendations as string
    """
    user_scores = cf_scores[cf_scores['user_id'] == participant_id]
    top_k = user_scores.nlargest(k, 'score')

    # Format as "query:::LLM/tool name:::data name"
    recommendations = []
    for _, row in top_k.iterrows():
        query_id = row['query_id']
        score = row['score']

        # Find query in queries_data
        query_info = next((q for q in queries_data if q['id'] == query_id), None)
        if query_info:
            # Format: query:::tool:::datatype
            query_text = query_info.get('query', '')
            tool = query_info.get('tool1', 'LLM')
            datatypes = query_info.get('datatype', [])

            # For each datatype, create a recommendation
            for datatype in datatypes[:2]:  # Limit to avoid overwhelming prompt
                rec = f"{query_text}:::{tool}:::{datatype}"
                recommendations.append(rec)

    return "\n".join(recommendations) if recommendations else "No recommendations available"


def create_ic_cf_prompt(bio_info, ai_experiences, permission_history, cf_recommendations, query_list):
    """
    Create IC+CF hybrid prompt with collaborative filtering recommendations.

    Uses the exact prompt structure from the original implementation.

    Args:
        bio_info: User demographic information
        ai_experiences: User's AI tool experiences
        permission_history: Past permission decisions
        cf_recommendations: Formatted CF recommendations string
        query_list: New queries to predict

    Returns:
        str: Formatted prompt for LLM
    """
    prompt = (
        "You are an intelligent, personalized assistant tasked with predicting a user's permission decisions within an LLM-based agentic system. "
        "Your task is to analyze the following data and output, for each permission request, a prediction label along with a confidence score between 0 and 1 for that decision. "
        "The factors to consider are:\n"
        "- User demographic and bio information\n"
        "- AI tool usage experiences\n"
        "- Permission decision history\n"
        "- The semantic meaning of the new permission request\n"
        "- Additionally, you have access to permission recommendations provided by a collaborative filtering model. "
        "These recommendations list the top five permission requests that the user may allow, and are formatted as: 'query:::LLM/tool name:::data name'.\n\n"
        "The output should include two keys for each permission request:\n"
        "1. 'label': with either 'Yes, always share' or 'No, never share'\n"
        "2. 'score': a numeric confidence score between 0 and 1 for the predicted label\n\n"
        "Please follow the JSON format exactly, without any extra text. Use the following format as an example:\n"
        f"{EXAMPLE_OUTPUTS}\n\n"
        "User's demographic and bio information:\n"
        f"{bio_info}\n\n"
        "User's AI tool usage experiences:\n"
        f"{ai_experiences}\n\n"
        "User's permission decision history:\n"
        f"{permission_history}\n\n"
        "New permission requests:\n"
        f"{query_list}\n\n"
        "Permission recommendations from collaborative filtering:\n"
        f"{cf_recommendations}\n\n"
        "For each permission request, output the decision (label) and the corresponding confidence score."
        "IMPORTANT1: Your output must be in valid JSON format exactly, without any extra text or commentary outside the JSON structure.\n"
        "IMPORTANT2: All numeric fields (such as the score) must be valid numbers (e.g. 0.90) and not words.\n"
        "IMPORTANT3: Neither the label nor the score fields should be None."
    )

    return prompt


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=lambda retry_state: logger.warning(f"API call failed, retrying... (attempt {retry_state.attempt_number})")
)
def llm_inference(prompt):
    """
    Call OpenAI API for inference with automatic retry logic.

    Args:
        prompt: Input prompt

    Returns:
        str: Model response

    Raises:
        Exception: If API call fails after all retries
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "assistant", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"API call failed: {e}")
        raise


def extract_predictions(response):
    """
    Extract JSON predictions from LLM response.

    Args:
        response: Raw LLM response text

    Returns:
        list: Parsed predictions
    """
    try:
        # Try to find JSON array in response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        else:
            # Try to parse entire response as JSON
            return json.loads(response)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing LLM response: {e}")
        logger.debug(f"Response: {response[:500]}...")
        return []


def process_participant(participant_id, participant_data, queries, cf_scores):
    """
    Process one participant: make predictions using IC+CF.

    Uses pre-split training and testing data from the dataset.

    Args:
        participant_id: Participant ID (e.g., P001)
        participant_data: Participant's profile and training data
        queries: All study queries
        cf_scores: Collaborative filtering scores

    Returns:
        dict: Predictions for this participant
    """
    try:
        # Prepare participant profile
        bio_info = participant_data['bio']
        ai_experiences = participant_data['ai_experience']

        # Use pre-split training and testing data from dataset
        training_examples = participant_data.get('training', [])
        test_queries_data = participant_data.get('testing', [])

        # If no test data, skip this participant
        if not test_queries_data:
            logger.info(f"Skipping {participant_id} - no test data")
            return {
                "participant_id": participant_id,
                "predictions": [],
                "skipped": True,
                "reason": "no_test_data"
            }

        # Get CF recommendations
        cf_recommendations = get_top_k_recommendations(participant_id, cf_scores, queries)

        # Format test queries for prediction (include permission structure with null values)
        test_queries = [
            {"query": ex['query'], "id": ex['id'], "permission": ex.get('permission', {})}
            for ex in test_queries_data
        ]

        # Create prompt
        prompt = create_ic_cf_prompt(
            bio_info, ai_experiences, training_examples,
            cf_recommendations, test_queries
        )

        # Get predictions
        response = llm_inference(prompt)
        predictions = extract_predictions(response)

        logger.info(f"✓ Processed {participant_id} ({len(training_examples)} train, {len(test_queries_data)} test)")

        return {
            "participant_id": participant_id,
            "cf_recommendations": cf_recommendations,
            "predictions": predictions,
            "num_train": len(training_examples),
            "num_test": len(test_queries_data),
            "ground_truth": test_queries_data
        }

    except Exception as e:
        logger.error(f"✗ Error processing {participant_id}: {e}")
        return {
            "participant_id": participant_id,
            "predictions": [],
            "error": str(e)
        }


def main():
    """Main execution function."""

    logger.info("="*60)
    logger.info("LLM Inference - IC+CF Hybrid")
    logger.info("="*60)
    logger.info(f"Using model: {MODEL}")

    # Load data
    logger.info("Loading data...")
    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)

    with open(QUERIES_PATH, 'r') as f:
        queries = json.load(f)

    cf_scores = load_cf_scores(CF_SCORES_PATH)

    logger.info(f"Loaded {len(dataset)} participants")
    logger.info(f"Loaded {len(queries)} queries")
    logger.info(f"Loaded {len(cf_scores)} CF scores")

    # Process participants in parallel with progress bar
    logger.info(f"Processing with {MAX_WORKERS} parallel workers...")
    results = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_participant, pid, pdata, queries, cf_scores): pid
            for pid, pdata in dataset.items()
        }

        # Add progress bar
        with tqdm(total=len(futures), desc="Processing participants", unit="participant") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results[result['participant_id']] = result
                pbar.update(1)

    # Calculate statistics
    successful = sum(1 for r in results.values() if r.get('predictions') and not r.get('skipped'))
    skipped = sum(1 for r in results.values() if r.get('skipped'))
    errors = sum(1 for r in results.values() if 'error' in r)

    # Save results
    logger.info(f"Saving results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ Complete!")
    logger.info(f"  - Total participants: {len(results)}")
    logger.info(f"  - Successfully processed: {successful}")
    logger.info(f"  - Skipped (insufficient data): {skipped}")
    logger.info(f"  - Errors: {errors}")

    # Calculate and print metrics
    if successful > 0:
        all_true = []
        all_pred = []

        for participant_id, data in results.items():
            if data.get('skipped') or 'error' in data:
                continue

            predictions = data.get('predictions', [])
            ground_truth = data.get('ground_truth', [])

            # Match predictions with ground truth by query ID
            for gt_item in ground_truth:
                query_id = gt_item['id']
                gt_permissions = gt_item.get('answer', {})

                # Find corresponding prediction
                pred_item = next((p for p in predictions if p.get('id') == query_id), None)
                if not pred_item:
                    continue  # Skip if no prediction for this query

                pred_permissions = pred_item.get('permission', {})

                # For each data type in PREDICTIONS (paper's approach)
                for data_type, pred_data in pred_permissions.items():
                    # Skip if this data type is not in ground truth
                    if data_type not in gt_permissions:
                        continue

                    # Get predicted label and convert to binary
                    pred_label = pred_data.get('label', '')
                    pred_binary = 1 if "Yes" in pred_label else 0

                    # Get ground truth label and convert to binary
                    gt_label = gt_permissions[data_type]
                    gt_binary = 1 if "Yes" in gt_label else 0

                    all_true.append(gt_binary)
                    all_pred.append(pred_binary)

        if len(all_true) > 0:
            # Calculate metrics using unified utilities
            metrics = calculate_metrics(
                y_true=all_true,
                y_pred=all_pred,
                method_name="IC+CF",
                threshold=0.5  # IC+CF uses simple majority voting
            )
            metrics['n_participants'] = successful

            # Print metrics using unified format
            print_metrics(metrics)

            # Save metrics to results/ folder
            save_metrics(metrics, "../results/ic_cf_metrics.json")
        else:
            logger.warning("No valid predictions found for metric calculation")

    logger.info("="*60)


if __name__ == "__main__":
    main()
