"""
LLM Inference without Collaborative Filtering (IC Only Baseline)

This script implements the baseline in-context learning (IC) approach
for automated permission decisions without collaborative filtering.

Inputs:
    - ../data/processed_dataset.json - Processed dataset (181 filtered participants)
    - ../queries.json - Study scenarios

Outputs:
    - Permission predictions with confidence scores using IC-only approach
    - Saved to: ic_only_predictions.json

Requirements:
    - OpenAI API key (set via environment variable)
"""

import os
import sys
import json
import random
from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError
import re
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
OUTPUT_PATH = "../results/ic_only_predictions.json"
MAX_WORKERS = 5  # Parallel processing workers

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


def create_ic_prompt(bio_info, ai_experiences, permission_history, query_list):
    """
    Create in-context learning prompt for permission prediction.

    Uses the exact prompt structure from the original implementation.

    Args:
        bio_info: User demographic information
        ai_experiences: User's AI tool experiences
        permission_history: Past permission decisions (training examples)
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
        "For each permission request, output the decision (label) and the corresponding confidence score.\n"
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


def process_participant(participant_id, participant_data):
    """
    Process one participant: make predictions for their test queries.

    Uses pre-split training and testing data from the dataset.

    Args:
        participant_id: Participant ID (e.g., P001)
        participant_data: Participant's profile and training data

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

        # Format test queries for prediction (include permission structure with null values)
        test_queries = [
            {"query": ex['query'], "id": ex['id'], "permission": ex.get('permission', {})}
            for ex in test_queries_data
        ]

        # Create prompt
        prompt = create_ic_prompt(bio_info, ai_experiences, training_examples, test_queries)

        # Get predictions
        response = llm_inference(prompt)
        predictions = extract_predictions(response)

        logger.info(f"✓ Processed {participant_id} ({len(training_examples)} train, {len(test_queries_data)} test)")

        return {
            "participant_id": participant_id,
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
    logger.info("LLM Inference - IC Only Baseline")
    logger.info("="*60)
    logger.info(f"Using model: {MODEL}")

    # Load data
    logger.info("Loading data...")
    with open(DATASET_PATH, 'r') as f:
        dataset = json.load(f)

    with open(QUERIES_PATH, 'r') as f:
        queries = json.load(f)

    logger.info(f"Loaded {len(dataset)} participants")
    logger.info(f"Loaded {len(queries)} queries")

    # Process participants in parallel with progress bar
    logger.info(f"Processing with {MAX_WORKERS} parallel workers...")
    results = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_participant, pid, pdata): pid
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
                method_name="IC Only",
                threshold=0.5  # IC uses simple majority voting
            )
            metrics['n_participants'] = successful

            # Print metrics using unified format
            print_metrics(metrics)

            # Save metrics to results/ folder
            save_metrics(metrics, "../results/ic_only_metrics.json")
        else:
            logger.warning("No valid predictions found for metric calculation")

    logger.info("="*60)


if __name__ == "__main__":
    main()
