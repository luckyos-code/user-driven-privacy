import os, argparse
from src.Main import run_evaluation
from src.DatasetCreation import create_dataset_versions
from src.PreparingMethod import PreparingMethod

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', required=True, help='Directory to save results')
parser.add_argument('--data_dir', required=True, help='Base directory containing datasets')
parser.add_argument('--dataset', required=True, help='Name of the dataset to load')
parser.add_argument('--train_method', required=True, help='Method to apply to training data')
parser.add_argument('--test_method', required=True, help='Method to apply to testing data')
parser.add_argument('--group_duplicates', action='store_true', help='Whether to deduplicate records')
parser.add_argument('--use_gpu', action='store_true', help='Whether to use GPU accelartion')
parser.add_argument('--n_workers', required=True, help='Size of dask cluster to use')
parser.add_argument('--filter_by_record_id', action='store_true', help='Whether to filter by record id')
parser.add_argument('--percentages', required=True, help='Percentages for the dataset split')
parser.add_argument('--cache_only', action='store_true', help='Whether to only prepare and cache datasets without running evaluations')
args = parser.parse_args()

print(f"Saving to: {args.save_dir}")
print(f"Dataset: {args.dataset}")
print(f"Train method: {args.train_method}")
print(f"Test method: {args.test_method}")
print(f"Group duplicates: {args.group_duplicates}")
print(f"Filter by record id: {args.filter_by_record_id}")
print(f"Using GPU: {args.use_gpu}")
print(f"Using n_workers: {args.n_workers}")
print(f"Percentages: {args.percentages}")
print(f"Cache only: {args.cache_only}")

# Convert string arguments to appropriate PreparingMethod enum values
train_method = getattr(PreparingMethod, args.train_method)
test_method = getattr(PreparingMethod, args.test_method)

# Parse percentages string into three floats
try:
    pct_values = [float(x) for x in args.percentages.replace(',', ' ').split()]
    if len(pct_values) != 3:
        raise ValueError("Percentages must have exactly three values.")
    original_pct, generalized_pct, missing_pct = pct_values
except Exception as e:
    raise ValueError(f"Error parsing percentages: {args.percentages} ({e})")

# Create dataset versions based on percentages
pct_str = f"{int(round(original_pct*100))}-{int(round(generalized_pct*100))}-{int(round(missing_pct*100))}"

print(f"Preparing data for {args.dataset} with {pct_str} split...")
seed=42
create_dataset_versions(args.dataset, original_pct, generalized_pct, missing_pct, seed, args.data_dir)

# Run the evaluation with the provided parameters
run_evaluation(
    n_workers=args.n_workers,
    use_gpu=args.use_gpu,
    save_dir=args.save_dir,
    data_dir=args.data_dir,
    dataset=args.dataset,
    train_method=train_method,
    test_method=test_method,
    group_duplicates=args.group_duplicates,
    filter_by_record_id=args.filter_by_record_id,
    percentages=pct_str,
    cache_only=args.cache_only,
)