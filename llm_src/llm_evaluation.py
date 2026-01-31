#!/usr/bin/env python3
"""
LLM Evaluation Script for Privacy Risk Assessment

Evaluates LLM privacy risks by testing:
1. Value Imputation - Predicting missing/generalized values
2. Target Prediction - Classifying outcomes from anonymized features

Usage:
    python llm_evaluation.py --percentage 33-33-34 --datasets Adult,Diabetes
    python llm_evaluation.py --percentage 33-33-34 --datasets Adult --n-samples 100
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
from asyncio import Semaphore
from dotenv import load_dotenv
import tempfile
from typing import Dict, List, Any, Optional

# Fix import path when script is in llm_src/ subfolder
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.DatasetManager import DatasetManager


# ============================================================================
# CONFIGURATION
# ============================================================================

base_path = Path(__file__).parent.parent

# Load environment variables
load_dotenv(base_path / '.env_llm')

LLM_API_BASE_URL = os.getenv('LLM_API_BASE_URL')
LLM_API_KEY = os.getenv('LLM_API_KEY')
LLM_MODEL = os.getenv('LLM_MODEL')
# Task configurations
IMPUTATION_CONFIG = {
    'system_prompt': "You are a data analyst filling missing or generalized values.",
    'temperature': 0.0,
    'max_tokens': 100,  # Increased tokens to allow for multiple column predictions per row
        'prompt_template': """Dataset: {dataset_name}
Task: Impute specific concrete values for the indicated columns in each record.

Context:
- "?" means value is completely missing
- Values like [30-39] are generalized ranges
- Semantic values like "young" are generalizations

Instructions:
- For each record, provide a specific concrete value for EVERY column listed in "Targets".
- Predict specific values (e.g., "35" instead of "[30-39]", "Private" instead of "private_sector").

Records to process:
{records_block}

Instructions for output format:
- Return ONE LINE PER RECORD in this exact format (no JSON, no markdown):
    REQ_ID<TAB>col1=value1|col2=value2|...
- Use the pipe character `|` to separate column predictions.
- Values may contain spaces but MUST NOT contain the `|` or tab characters.
- If a value is unknown, return the string "UNK" for that column.

Example lines:
REQ_0\tage=35|workclass=Private
REQ_1\toccupation=Sales

Return ONLY the lines, nothing else.
"""
}

PREDICTION_CONFIG = {
    'system_prompt': "You are predicting target variables.",
    'temperature': 0.0,
    'max_tokens': 20,
        'prompt_template': """Dataset: {dataset_name}
Task: Predict the target variable '{target_name}' (0 or 1) for the following records.

{target_info}

Instructions:
- Return ONE LINE PER RECORD in this exact format (no JSON, no markdown):
    REQ_ID<TAB>value
- value must be 0 or 1

Records to process:
{records_block}

Example:
REQ_1\t0
REQ_2\t1

Return ONLY the lines, nothing else.
"""
}


# ============================================================================
# LLM API FUNCTIONS
# ============================================================================

async def call_llm_async(
    session: aiohttp.ClientSession,
    semaphore: Semaphore,
    prompt: str,
    system_prompt: str = None,
    temperature: float = 0.0,
    max_tokens: int = 100,
    max_retries: int = 5
) -> str:
    """Call the hosted LLM API asynchronously with retry logic."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    
    for attempt in range(max_retries):
        async with semaphore:
            try:
                async with session.post(
                    f"{LLM_API_BASE_URL}/api/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    try:
                        response.raise_for_status()
                    except Exception as e:
                        # Surface status for debugging (SLURM logs will capture this)
                        text = await response.text()
                        print(f"LLM API returned status {response.status}: {text[:200]}")
                        raise
                    # Try to parse json, fallback to text if content-type is not json
                    try:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    except Exception:
                        text = await response.text()
                        return text
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"LLM call attempt {attempt+1} failed: {type(e).__name__}: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"LLM API call failed after {max_retries} attempts: {type(e).__name__}: {str(e)}")
                await asyncio.sleep(min(30, 2 ** attempt))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_record_for_prompt(record, exclude_columns=None, dataset_name=None):
    """Format a pandas Series (record) as a string for LLM prompts."""
    if exclude_columns is None:
        if dataset_name:
            record_id_col = DatasetManager.get_record_id_column(dataset_name)
            label_col = DatasetManager.get_label_column(dataset_name)
            exclude_columns = [record_id_col, label_col]
        else:
            exclude_columns = ['record_id']
    
    features = []
    for col, val in record.items():
        if col not in exclude_columns:
            features.append(f"- {col}: {val}")
    
    return "\n".join(features)


def _strip_markdown_fences(text: str) -> str:
    """Remove surrounding ``` fences (with optional json language tag)."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    return stripped


def _sanitize_json_like(text: str) -> str:
    """Quote bare tokens so that json.loads can consume almost-JSON strings."""
    sanitized = text

    # Remove dangling commas before closing braces/brackets
    sanitized = re.sub(r",(\s*[}\\]])", r"\1", sanitized)

    # Quote bracketed range tokens such as [30-39]
    range_pattern = re.compile(r"(:\s*)(\[\s*-?\d+\s*-\s*-?\d+\s*\])(\s*[,\}])")
    sanitized = range_pattern.sub(lambda m: f"{m.group(1)}\"{m.group(2)}\"{m.group(3)}", sanitized)

    def _quote_bare_value(match):
        prefix, value, suffix = match.groups()
        raw = value.strip()

        # Already quoted
        if raw.startswith('"') or raw.startswith("'"):
            return match.group(0)

        # Preserve numeric literals
        if re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
            return f"{prefix}{raw}{suffix}"

        lowered = raw.lower()
        if lowered in {"true", "false", "null"}:
            return f"{prefix}{lowered}{suffix}"

        # Leave actual arrays/objects untouched (start with { or [)
        if raw.startswith('{') or raw.startswith('['):
            return match.group(0)

        return f"{prefix}\"{raw}\"{suffix}"

    bare_value_pattern = re.compile(r"(:\s*)([^,\}\]\{\"\n]+)(\s*[,\}])")
    sanitized = bare_value_pattern.sub(_quote_bare_value, sanitized)

    return sanitized


def _parse_llm_json_response(raw_prediction: str, task_label: str = "") -> Optional[Dict[str, Any]]:
    """Parse LLM output, falling back to lenient sanitization if needed."""
    clean = _strip_markdown_fences(raw_prediction)

    start_idx = clean.find('{')
    end_idx = clean.rfind('}')
    if start_idx != -1 and end_idx != -1:
        clean = clean[start_idx:end_idx + 1]

    # Remove simple comment lines (// or #)
    clean_lines = []
    for line in clean.split('\n'):
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('#'):
            continue
        clean_lines.append(line)
    clean = '\n'.join(clean_lines)

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        sanitized = _sanitize_json_like(clean)
        try:
            return json.loads(sanitized)
        except json.JSONDecodeError:
            return None


def _parse_line_response_imputation(raw_prediction: str) -> Optional[Dict[str, Dict[str, str]]]:
    """Parse line-based imputation responses of the form:
    REQ_ID<TAB>col1=val1|col2=val2
    Returns dict: req_id -> {col: val}
    """
    result = {}
    for raw_line in raw_prediction.split('\n'):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith('```') or line.startswith('//') or line.startswith('#'):
            continue

        # Try to find request id at start
        m = re.match(r'^(REQ[^\s\t:]+)[\t\s:-]+(.*)$', line)
        if not m:
            # fallback: skip
            continue
        req_id = m.group(1).strip()
        rest = m.group(2).strip()

        # split key-value pairs by | first, then ;, then ,
        if '|' in rest:
            pairs = rest.split('|')
        elif ';' in rest:
            pairs = rest.split(';')
        else:
            pairs = [rest]

        row = {}
        for p in pairs:
            if not p:
                continue
            # split on first = or :
            if '=' in p:
                k, v = p.split('=', 1)
            elif ':' in p:
                k, v = p.split(':', 1)
            else:
                continue
            row[k.strip()] = v.strip().strip('"')
        if row:
            result[req_id] = row
    return result if result else None


def _parse_line_response_prediction(raw_prediction: str) -> Optional[Dict[str, str]]:
    """Parse line-based prediction responses of the form:
    REQ_ID<TAB>value
    Returns dict: req_id -> value
    """
    result = {}
    for raw_line in raw_prediction.split('\n'):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith('```') or line.startswith('//') or line.startswith('#'):
            continue
        m = re.match(r'^(REQ[^\s\t:]+)[\t\s:-]+(.+)$', line)
        if not m:
            continue
        req_id = m.group(1).strip()
        val = m.group(2).strip().strip('"')
        result[req_id] = val
    return result if result else None


def _parse_llm_response(raw_prediction: str, mode: str = 'imputation', task_label: str = '') -> Optional[Dict[str, Any]]:
    """Try JSON parsing first, then fall back to line-based parsing depending on mode."""
    j = _parse_llm_json_response(raw_prediction, task_label=task_label)
    if j:
        return j
    if mode == 'imputation':
        return _parse_line_response_imputation(raw_prediction)
    else:
        return _parse_line_response_prediction(raw_prediction)


def _is_binary_labels(series):
    """Return True if the series contains only binary labels (0/1)."""
    vals = set(series.dropna().astype(str).str.strip())
    if len(vals) == 0:
        return False
    normalized = set()
    for v in vals:
        try:
            nv = str(int(float(v)))
            normalized.add(nv)
        except Exception:
            normalized.add(v)
    return normalized <= {"0", "1"}


def compute_classification_metrics(results_df, positive_label='1'):
    """Compute binary classification metrics."""
    if results_df is None or len(results_df) == 0:
        return None

    y_true = results_df['true_value'].astype(str).str.strip().fillna("")
    y_pred = results_df['predicted_value'].astype(str).str.strip().fillna("")

    def _norm_label(x):
        try:
            return str(int(float(x)))
        except Exception:
            return x

    y_true = y_true.map(_norm_label)
    y_pred = y_pred.map(_norm_label)

    pos = str(positive_label)
    tp = int(((y_pred == pos) & (y_true == pos)).sum())
    fp = int(((y_pred == pos) & (y_true != pos)).sum())
    fn = int(((y_pred != pos) & (y_true == pos)).sum())
    tn = int(((y_pred != pos) & (y_true != pos)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    support = int((y_true == pos).sum())
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else None

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'support': support, 'accuracy': accuracy
    }


def apply_imputation_results(anon_df, imputation_results_df, id_col='record_id', log_file=None):
    """Create dataset with only imputed records, filling predicted values."""
    def _log(msg):
        print(msg)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
    
    if imputation_results_df is None or imputation_results_df.empty:
        _log("No imputation results to apply; returning empty dataframe.")
        return pd.DataFrame()

    imputed_record_ids = imputation_results_df['record_id'].unique()
    imputed_df = anon_df[anon_df[id_col].isin(imputed_record_ids)].copy()
    
    _log(f"Creating imputed dataset with {len(imputed_df)} records (from {len(anon_df)} total)")
    
    imputed_df_indexed = imputed_df.set_index(id_col)
    applied_count = 0
    missing_ids = set()
    
    valid_results = imputation_results_df.dropna(subset=['record_id', 'column', 'predicted_value'])
    
    for rid, group in valid_results.groupby('record_id'):
        if rid not in imputed_df_indexed.index:
            missing_ids.add(rid)
            continue
        
        for _, row in group.iterrows():
            col = row['column']
            pred = row['predicted_value']
            if col in imputed_df_indexed.columns:
                imputed_df_indexed.at[rid, col] = pred
                applied_count += 1
    
    imputed_df = imputed_df_indexed.reset_index()

    _log(f"✓ Applied {applied_count} imputation predictions")
    _log(f"✓ Dataset contains {len(imputed_df)} records with {applied_count/max(1, len(imputed_df)):.1f} imputed columns per record on average")
    
    if missing_ids:
        unique_missing = list(missing_ids)
        _log(f"⚠️  Warning: {len(unique_missing)} record_ids from imputation results not found in dataset")
        if len(unique_missing) <= 5:
            _log(f"   Missing ids: {unique_missing}")

    return imputed_df


def print_evaluation_summary(results_df, task_name, log_file=None):
    """Print summary statistics for evaluation results."""
    def _log(msg):
        print(msg)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
    
    _log(f"\n{'='*80}")
    _log(f"{task_name} EVALUATION RESULTS")
    _log(f"{'='*80}")

    total = len(results_df)
    _log(f"Total predictions: {total}")
    
    if total == 0:
        _log("⚠️  No predictions to evaluate (empty results)")
        return

    if 'correct' in results_df.columns:
        correct_count = int(results_df['correct'].sum())
        accuracy = (correct_count / total) if total > 0 else None
        _log(f"Correct predictions: {correct_count}")
        if accuracy is not None:
            _log(f"Accuracy: {accuracy:.2%}")

    if all(c in results_df.columns for c in ('predicted_value', 'true_value')):
        if _is_binary_labels(results_df['true_value']) and _is_binary_labels(results_df['predicted_value']):
            metrics = compute_classification_metrics(results_df, positive_label='1')
            if metrics:
                _log(f"\nBinary classification metrics (positive label=1):")
                _log(f"  Precision: {metrics['precision']:.2%}")
                _log(f"  Recall:    {metrics['recall']:.2%}")
                _log(f"  F1 score:  {metrics['f1']:.2%}")
                _log(f"  Support:   {metrics['support']}")
                _log(f"  Confusion: TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")

    if 'value_type' in results_df.columns:
        _log(f"\nBy value type:")
        grouped = results_df.groupby('value_type')['correct'].agg(['mean', 'sum', 'count'])
        for vtype, row in grouped.iterrows():
            _log(f"  {vtype}: {row['mean']:.2%} ({int(row['sum'])}/{int(row['count'])})")

    if 'column' in results_df.columns:
        _log(f"\nBy column:")
        grouped = results_df.groupby('column')['correct'].agg(['mean', 'sum', 'count'])
        for col, row in grouped.iterrows():
            _log(f"  {col}: {row['mean']:.2%} ({int(row['sum'])}/{int(row['count'])})")


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

async def evaluate_imputation(
    anon_data, 
    original_data, 
    dataset_name, 
    n_samples=100, 
    columns_to_test=None, 
    concurrency=1,
    partition=None,
    batch_size=1,
    specific_record_ids: Optional[List[Any]] = None
):
    """Evaluate LLM's ability to predict missing/generalized values."""
    # Get column classes from DatasetManager
    try:
        spalten_dict, spalten_list = DatasetManager.get_spalten_classes(dataset_name)
        testable_columns = spalten_dict
    except ValueError as e:
        raise ValueError(f"Cannot evaluate imputation for dataset '{dataset_name}': {e}")
    
    if columns_to_test:
        testable_columns = {k: v for k, v in testable_columns.items() if k in columns_to_test}
    
    # Get dataset-specific column names
    record_id_col = DatasetManager.get_record_id_column(dataset_name)
    label_col = DatasetManager.get_label_column(dataset_name)
    
    all_items = []
    
    # If a specific list of record ids was provided, evaluate only those rows
    if specific_record_ids is not None:
        record_id_col = DatasetManager.get_record_id_column(dataset_name)
        sample_indices = anon_data[anon_data[record_id_col].isin(specific_record_ids)].index
        print(f"Processing specific record ids: found {len(sample_indices)} matching rows for provided ids")
    else:
        # Handle partitioning if specified
        if partition:
            part_num, total_parts = map(int, partition.split('/'))
            total_rows = len(anon_data)
            rows_per_part = total_rows // total_parts
            start_idx = (part_num - 1) * rows_per_part
            end_idx = start_idx + rows_per_part if part_num < total_parts else total_rows

            # Use partition subset instead of random sampling
            sample_indices = anon_data.iloc[start_idx:end_idx].index
            print(f"Processing partition {part_num}/{total_parts}: rows {start_idx} to {end_idx} ({len(sample_indices)} records)")
        else:
            sample_indices = anon_data.sample(n=min(n_samples, len(anon_data)), random_state=42).index
    
    # Group targets by record (Row-level imputation)
    row_tasks = []
    
    for idx in sample_indices:
        anon_record = anon_data.loc[idx]
        orig_record = original_data.loc[idx]
        
        targets_for_row = []
        
        for col_name, col_class in testable_columns.items():
            anon_value = str(anon_record[col_name])
            orig_value = str(orig_record[col_name])
            
            if anon_value == orig_value:
                continue
            
            is_missing = (anon_value == '?' or pd.isna(anon_value) or anon_value == 'nan')
            is_generalized = False
            
            if not is_missing and hasattr(col_class, 'is_generalized'):
                is_generalized = col_class.is_generalized(anon_value)
            
            if is_missing or is_generalized:
                targets_for_row.append({
                    'column': col_name,
                    'anon_value': anon_value,
                    'true_value': orig_value,
                    'is_missing': is_missing
                })
        
        if targets_for_row:
            # Include ALL features in the prompt (including '?' and ranges) so model sees the context
            # Only exclude ID and Label if necessary
            exclude_cols = [record_id_col, label_col]
            record_features = format_record_for_prompt(anon_record, exclude_columns=exclude_cols)
            
            row_tasks.append({
                'record_id': anon_record[record_id_col],
                'targets': targets_for_row,
                'record_features': record_features,
                'dataset_name': dataset_name
            })

    # Group rows into batches
    batches = []
    for i in range(0, len(row_tasks), batch_size):
        batches.append(row_tasks[i:i + batch_size])
    
    print(f"Created {len(batches)} batches from {len(row_tasks)} records (batch_size={batch_size})")

    tasks = []
    for batch_idx, batch in enumerate(batches):
        records_block = []
        batch_mapping = {} # req_id -> metadata
        
        for item_idx, item in enumerate(batch):
            req_id = f"REQ_{batch_idx}_{item_idx}"
            batch_mapping[req_id] = item
            
            target_list_str = ", ".join([f"{t['column']} (Current: {t['anon_value']})" for t in item['targets']])
            
            records_block.append(f"Request ID: {req_id}")
            records_block.append(item['record_features'])
            records_block.append(f"Targets: {target_list_str}")
            records_block.append("---")
        
        prompt = IMPUTATION_CONFIG['prompt_template'].format(
            dataset_name=dataset_name,
            records_block="\n".join(records_block)
        )
        
        # Scale max_tokens based on number of targets in the batch
        total_targets = sum(len(item['targets']) for item in batch)
        # Base tokens per target + overhead
        dynamic_max_tokens = max(100, total_targets * 15)
        
        tasks.append({
            'prompt': prompt,
            'system_prompt': IMPUTATION_CONFIG['system_prompt'],
            'temperature': IMPUTATION_CONFIG['temperature'],
            'max_tokens': dynamic_max_tokens,
            'is_batch': True,
            'batch_mapping': batch_mapping
        })
    
    semaphore = Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        coroutines = [_execute_imputation_task(session, semaphore, task) for task in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Flatten results
        flat_results = []
        for r in results:
            if isinstance(r, Exception) or r is None:
                continue
            if isinstance(r, list):
                flat_results.extend(r)
            else:
                flat_results.append(r)

    df = pd.DataFrame(flat_results)

    # Ensure no record was dropped: every row_task record_id must appear at least once
    expected_record_ids = {t['record_id'] for t in row_tasks}
    found_record_ids = set(df['record_id'].unique()) if not df.empty else set()
    missing = expected_record_ids - found_record_ids
    if missing:
        # Do not crash the whole run for a small number of missing predictions.
        # Attach missing ids to the returned DataFrame for downstream diagnostics
        print(f"⚠️  Imputation incomplete: missing predictions for {len(missing)} records. Missing sample ids (showing up to 10): {list(missing)[:10]}")
        try:
            df.attrs['missing_record_ids'] = list(missing)
        except Exception:
            # If df is not a DataFrame or attrs not supported, ignore
            pass

    return df


async def _execute_imputation_task(session, semaphore, task):
    """Execute a single imputation task (batch or single)."""
    try:
        prediction = await call_llm_async(
            session, semaphore, task['prompt'],
            system_prompt=task['system_prompt'],
            temperature=task['temperature'],
            max_tokens=task['max_tokens']
        )
        
        pred_dict = _parse_llm_response(prediction, mode='imputation', task_label="imputation batch")
        if pred_dict is None:
            # Fallback: retry each record in the batch individually to avoid losing entire batch
            batch_results = []
            for req_id, meta in task['batch_mapping'].items():
                # construct a single-record prompt
                target_list_str = ", ".join([f"{t['column']} (Current: {t['anon_value']})" for t in meta['targets']])
                records_block = []
                records_block.append(f"Request ID: {req_id}")
                records_block.append(meta['record_features'])
                records_block.append(f"Targets: {target_list_str}")
                prompt = IMPUTATION_CONFIG['prompt_template'].format(
                    dataset_name=meta.get('dataset_name', ''),
                    records_block="\n".join(records_block)
                )

                single_max_tokens = max(80, len(meta['targets']) * 30)
                try:
                    single_pred = await call_llm_async(
                        session, semaphore, prompt,
                        system_prompt=IMPUTATION_CONFIG['system_prompt'],
                        temperature=IMPUTATION_CONFIG['temperature'],
                        max_tokens=single_max_tokens
                    )
                except Exception as e:
                    print(f"Single-record fallback failed for {req_id}: {e}")
                    await asyncio.sleep(0.1)
                    continue

                single_dict = _parse_llm_response(single_pred, mode='imputation', task_label=f"imputation single {req_id}")
                if not single_dict:
                    await asyncio.sleep(0.05)
                    continue

                # Extract predictions for this req_id
                row_preds = single_dict.get(req_id) if isinstance(single_dict, dict) else None
                if not isinstance(row_preds, dict):
                    # maybe model returned direct mapping (col->val)
                    if isinstance(single_dict, dict):
                        row_preds = single_dict
                if not isinstance(row_preds, dict):
                    continue

                for target in meta['targets']:
                    col = target['column']
                    pred_value = None
                    if col in row_preds:
                        pred_value = row_preds[col]
                    else:
                        for k, v in row_preds.items():
                            if k.lower() == col.lower():
                                pred_value = v
                                break
                    if pred_value is not None:
                        batch_results.append({
                            'record_id': meta['record_id'],
                            'column': col,
                            'anon_value': target['anon_value'],
                            'predicted_value': str(pred_value),
                            'true_value': target['true_value'],
                            'correct': str(pred_value) == target['true_value'],
                            'value_type': 'missing' if target['is_missing'] else 'generalized'
                        })
                # small pause to avoid bursting the server
                await asyncio.sleep(0.05)
            return batch_results

        batch_results = []
        for req_id, row_preds in pred_dict.items():
            if req_id in task['batch_mapping']:
                meta = task['batch_mapping'][req_id]

                # row_preds should be a dict of col -> value
                if not isinstance(row_preds, dict):
                    continue

                for target in meta['targets']:
                    col = target['column']
                    # Check if this column was predicted
                    # Case-insensitive match attempt if direct match fails
                    pred_value = None
                    if col in row_preds:
                        pred_value = row_preds[col]
                    else:
                        for k, v in row_preds.items():
                            if k.lower() == col.lower():
                                pred_value = v
                                break

                    if pred_value is not None:
                        batch_results.append({
                            'record_id': meta['record_id'],
                            'column': col,
                            'anon_value': target['anon_value'],
                            'predicted_value': str(pred_value),
                            'true_value': target['true_value'],
                            'correct': str(pred_value) == target['true_value'],
                            'value_type': 'missing' if target['is_missing'] else 'generalized'
                        })
        return batch_results
    except Exception as e:
        print(f"Error: {e}")
        return None


async def evaluate_prediction(
    anon_data, 
    original_data, 
    dataset_name, 
    n_samples=100, 
    concurrency=1,
    partition=None,
    batch_size=1,
    specific_record_ids: Optional[List[Any]] = None
):
    """Evaluate LLM's ability to predict target variable from anonymized features."""
    # Get target column from DatasetManager
    target_col = DatasetManager.get_label_column(dataset_name)
    record_id_col = DatasetManager.get_record_id_column(dataset_name)
    
    # Get target description (with fallback for datasets without descriptions)
    config = DatasetManager.get_config(dataset_name)
    target_info = config.get('target_description', f"Target column: {target_col} (0 or 1)")
    
    all_items = []
    
    # If a specific list of record ids was provided, evaluate only those rows
    if specific_record_ids is not None:
        sample_indices = anon_data[anon_data[record_id_col].isin(specific_record_ids)].index
        print(f"Processing specific record ids: found {len(sample_indices)} matching rows for provided ids")
    else:
        # Handle partitioning if specified
        if partition:
            part_num, total_parts = map(int, partition.split('/'))
            total_rows = len(anon_data)
            rows_per_part = total_rows // total_parts
            start_idx = (part_num - 1) * rows_per_part
            end_idx = start_idx + rows_per_part if part_num < total_parts else total_rows
            
            # Use partition subset instead of random sampling
            sample_indices = anon_data.iloc[start_idx:end_idx].index
            print(f"Processing partition {part_num}/{total_parts}: rows {start_idx} to {end_idx} ({len(sample_indices)} records)")
        else:
            sample_indices = anon_data.sample(n=min(n_samples, len(anon_data)), random_state=42).index
    
    for idx in sample_indices:
        anon_record = anon_data.loc[idx]
        orig_record = original_data.loc[idx]
        
        record_features = format_record_for_prompt(
            anon_record, 
            exclude_columns=[record_id_col, target_col]
        )
        
        all_items.append({
            'record_id': anon_record[record_id_col],
            'true_value': str(orig_record[target_col]),
            'record_features': record_features
        })

    # Group into batches
    batches = []
    for i in range(0, len(all_items), batch_size):
        batches.append(all_items[i:i + batch_size])
        
    print(f"Created {len(batches)} batches from {len(all_items)} items (batch_size={batch_size})")
    
    tasks = []
    for batch_idx, batch in enumerate(batches):
        # Always use batch mode logic
        records_block = []
        batch_mapping = {}
        
        for item_idx, item in enumerate(batch):
            req_id = f"REQ_{batch_idx}_{item_idx}"
            batch_mapping[req_id] = item
            
            records_block.append(f"Request ID: {req_id}")
            records_block.append(item['record_features'])
            records_block.append("---")
        
        prompt = PREDICTION_CONFIG['prompt_template'].format(
            dataset_name=dataset_name,
            target_name=target_col,
            target_info=target_info,
            records_block="\n".join(records_block)
        )
        
        tasks.append({
            'prompt': prompt,
            'system_prompt': PREDICTION_CONFIG['system_prompt'],
            'temperature': PREDICTION_CONFIG['temperature'],
            'max_tokens': PREDICTION_CONFIG['max_tokens'] * len(batch),
            'is_batch': True,
            'batch_mapping': batch_mapping,
            'dataset_name': dataset_name,
            'target_name': target_col,
            'target_info': target_info
        })
    
    semaphore = Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        coroutines = [_execute_prediction_task(session, semaphore, task) for task in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Flatten results
        flat_results = []
        for r in results:
            if isinstance(r, Exception) or r is None:
                continue
            if isinstance(r, list):
                flat_results.extend(r)
            else:
                flat_results.append(r)

    df = pd.DataFrame(flat_results)

    # Ensure we produced a prediction for every requested item
    expected_ids = {item['record_id'] for batch in batches for item in batch}
    found_ids = set(df['record_id'].unique()) if not df.empty else set()
    missing = expected_ids - found_ids
    if missing:
        # Avoid raising to allow the evaluation run to finish; record missing ids for diagnostics
        print(f"⚠️  Prediction incomplete: missing predictions for {len(missing)} items. Missing ids (up to 10): {list(missing)[:10]}")
        try:
            df.attrs['missing_record_ids'] = list(missing)
        except Exception:
            pass

    return df


async def _execute_prediction_task(session, semaphore, task):
    """Execute a single prediction task (batch or single)."""
    try:
        prediction = await call_llm_async(
            session, semaphore, task['prompt'],
            system_prompt=task['system_prompt'],
            temperature=task['temperature'],
            max_tokens=task['max_tokens']
        )
        
        pred_dict = _parse_llm_response(prediction, mode='prediction', task_label="prediction batch")
        if pred_dict is None:
            # Fallback: evaluate each record in the batch individually
            batch_results = []
            for req_id, meta in task['batch_mapping'].items():
                records_block = []
                records_block.append(f"Request ID: {req_id}")
                records_block.append(meta['record_features'])
                records_block.append("---")
                # Use task-level metadata (set when tasks were created)
                ds_name = task.get('dataset_name', '')
                tgt_name = task.get('target_name', '')
                tgt_info = task.get('target_info', '')
                prompt = PREDICTION_CONFIG['prompt_template'].format(
                    dataset_name=ds_name,
                    target_name=tgt_name,
                    target_info=tgt_info,
                    records_block="\n".join(records_block)
                )
                try:
                    single_pred = await call_llm_async(
                        session, semaphore, prompt,
                        system_prompt=PREDICTION_CONFIG['system_prompt'],
                        temperature=PREDICTION_CONFIG['temperature'],
                        max_tokens=PREDICTION_CONFIG['max_tokens']
                    )
                except Exception as e:
                    print(f"Single-record prediction fallback failed for {req_id}: {e}")
                    await asyncio.sleep(0.1)
                    continue

                single_dict = _parse_llm_response(single_pred, mode='prediction', task_label=f"prediction single {req_id}")
                if not single_dict:
                    await asyncio.sleep(0.05)
                    continue

                val = None
                if isinstance(single_dict, dict):
                    if req_id in single_dict:
                        val = single_dict[req_id]
                    else:
                        # maybe returned direct scalar
                        if len(single_dict) == 1:
                            val = list(single_dict.values())[0]
                if val is None:
                    continue

                clean_val = str(val).strip()
                batch_results.append({
                    'record_id': meta['record_id'],
                    'predicted_value': clean_val,
                    'true_value': meta['true_value'],
                    'correct': clean_val == meta['true_value']
                })
                await asyncio.sleep(0.05)
            return batch_results

        batch_results = []
        for req_id, pred_value in pred_dict.items():
            if req_id in task['batch_mapping']:
                meta = task['batch_mapping'][req_id]
                clean_val = str(pred_value).strip()
                batch_results.append({
                    'record_id': meta['record_id'],
                    'predicted_value': clean_val,
                    'true_value': meta['true_value'],
                    'correct': clean_val == meta['true_value']
                })
        return batch_results
    except Exception as e:
        print(f"Error: {e}")
        return None


# ============================================================================
# MAIN EVALUATION RUNNER
# ============================================================================

async def run_evaluation(percentage, datasets, n_samples, concurrency, results_dir, input_dir='data', partition=None, batch_size=1, specific_record_ids: Optional[List[Any]] = None):
    """Run the complete evaluation pipeline."""
    print(f"{'='*80}")
    print(f"LLM PRIVACY RISK EVALUATION")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Percentage: {percentage}")
    print(f"  Datasets: {', '.join(datasets)}")
    print(f"  Sample size: {n_samples}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Partition: {partition if partition else 'None (full dataset)'}")
    print(f"  Results directory: {results_dir}")
    print(f"  API: {LLM_API_BASE_URL}")
    print(f"  Model: {LLM_MODEL}")
    print(f"{'='*80}\n")
    
    # Load data from configurable input directory
    data = {}
    for dataset in datasets:
        ds_lower = dataset.lower()
        # For percentage "1-0-0", use original data directly (no generalization folder)
        # Otherwise, anonymized (anon) files are expected under: {input_dir}/{dataset}/generalization/{percentage}/{dataset}_train.csv
        # original files are expected under: {input_dir}/{dataset}/{dataset}_train.csv
        if percentage == '1-0-0':
            # Original data evaluation: use top-level files for both anon and orig
            anon_train_path = os.path.join(input_dir, ds_lower, f'{ds_lower}_train.csv')
            anon_test_path = os.path.join(input_dir, ds_lower, f'{ds_lower}_test.csv')
        else:
            anon_train_path = os.path.join(input_dir, ds_lower, 'generalization', percentage, f'{ds_lower}_train.csv')
            anon_test_path = os.path.join(input_dir, ds_lower, 'generalization', percentage, f'{ds_lower}_test.csv')
        orig_train_path = os.path.join(input_dir, ds_lower, f'{ds_lower}_train.csv')
        orig_test_path = os.path.join(input_dir, ds_lower, f'{ds_lower}_test.csv')

        data[f'{ds_lower}_train'] = pd.read_csv(anon_train_path)
        data[f'{ds_lower}_test'] = pd.read_csv(anon_test_path)
        data[f'{ds_lower}_train_original'] = pd.read_csv(orig_train_path)
        data[f'{ds_lower}_test_original'] = pd.read_csv(orig_test_path)
    
    # Define evaluation tasks
    evaluation_tasks = []
    for dataset in datasets:
        ds_lower = dataset.lower()
        task = {
            'name': dataset,
            'imputations': [
                {'part': 'train', 'anon_data': data[f'{ds_lower}_train'], 'original_data': data[f'{ds_lower}_train_original'], 'columns': None},
                {'part': 'test', 'anon_data': data[f'{ds_lower}_test'], 'original_data': data[f'{ds_lower}_test_original'], 'columns': None}
            ],
            'predictions': [
                {'part': 'train', 'anon_data': data[f'{ds_lower}_train'], 'original_data': data[f'{ds_lower}_train_original']},
                {'part': 'test', 'anon_data': data[f'{ds_lower}_test'], 'original_data': data[f'{ds_lower}_test_original']}
            ]
        }
        evaluation_tasks.append(task)
    
    # Run evaluations
    results = {}
    total_start_time = time.time()
    files_created = 0
    
    os.makedirs(results_dir, exist_ok=True)
    
    for task in evaluation_tasks:
        dataset_name = task['name'].lower()
        
        # Determine partition suffix (only when total partitions > 1)
        if partition:
            part_num, total_parts = map(int, partition.split('/'))
            partition_suffix = f"_part{part_num}of{total_parts}" if total_parts > 1 else ""
        else:
            partition_suffix = ""
        ids_suffix = "_ids_replay" if specific_record_ids else ""
        suffix = partition_suffix + ids_suffix
        log_file = f"{results_dir}/{dataset_name}_evaluation{suffix}.log"
        
        with open(log_file, 'w') as f:
            f.write(f"{'='*80}\n")
            # Validate input files and print the exact paths that will be used
            def _collect_expected_paths(input_dir, dataset, percentage):
                ds_lower = dataset.lower()
                # For percentage "1-0-0" (original data), use top-level files directly
                if percentage == '1-0-0':
                    paths = {
                        'anon_train': os.path.join(input_dir, ds_lower, f'{ds_lower}_train.csv'),
                        'anon_test': os.path.join(input_dir, ds_lower, f'{ds_lower}_test.csv'),
                        'orig_train': os.path.join(input_dir, ds_lower, f'{ds_lower}_train.csv'),
                        'orig_test': os.path.join(input_dir, ds_lower, f'{ds_lower}_test.csv')
                    }
                else:
                    paths = {
                        'anon_train': os.path.join(input_dir, ds_lower, 'generalization', percentage, f'{ds_lower}_train.csv'),
                        'anon_test': os.path.join(input_dir, ds_lower, 'generalization', percentage, f'{ds_lower}_test.csv'),
                        'orig_train': os.path.join(input_dir, ds_lower, f'{ds_lower}_train.csv'),
                        'orig_test': os.path.join(input_dir, ds_lower, f'{ds_lower}_test.csv')
                    }
                return paths

            expected_paths = {}
            missing = []
            for dataset in datasets:
                paths = _collect_expected_paths(input_dir, dataset, percentage)
                expected_paths[dataset] = paths
                for name, p in paths.items():
                    if not os.path.exists(p):
                        missing.append((dataset, name, p))

            # Print/log the paths we will attempt to read
            print("Input files to be used:")
            for dataset, paths in expected_paths.items():
                print(f"- Dataset: {dataset}")
                for name, p in paths.items():
                    print(f"    {name}: {p}")

            if missing:
                print("\nERROR: Some required input files are missing:")
                for dataset, name, p in missing:
                    print(f"  - {dataset}: missing {name} -> {p}")
                print("\nPlease ensure the input directory contains the expected files with the correct structure.")
                raise FileNotFoundError(f"Missing {len(missing)} input files. See output for details.")

            # Load data after validation
            data = {}
            for dataset in datasets:
                ds_lower = dataset.lower()
                paths = expected_paths[dataset]
                data[f'{ds_lower}_train'] = pd.read_csv(paths['anon_train'])
                data[f'{ds_lower}_test'] = pd.read_csv(paths['anon_test'])
                data[f'{ds_lower}_train_original'] = pd.read_csv(paths['orig_train'])
                data[f'{ds_lower}_test_original'] = pd.read_csv(paths['orig_test'])
        
        for imputation_task in task['imputations']:
            part_name = imputation_task['part']
            
            start_time = time.time()
            
            try:
                imputation_results = await evaluate_imputation(
                    imputation_task['anon_data'],
                    imputation_task['original_data'],
                    dataset_name=dataset_name,
                    n_samples=n_samples,
                    columns_to_test=imputation_task['columns'],
                    concurrency=concurrency,
                    partition=partition,
                    batch_size=batch_size,
                    specific_record_ids=specific_record_ids
                )
            except Exception as e:
                # Capture missing-id exceptions and continue the run.
                print(f"Caught exception during imputation: {e}")
                # Try to extract numeric ids from the exception message (supports formats like [np.int64(32347)])
                missing_ids = []
                try:
                    msg = str(e)
                    # find patterns like np.int64(12345) or plain integers
                    nums = re.findall(r"np\.int64\((\d+)\)", msg)
                    if not nums:
                        nums = re.findall(r"-?\d+", msg)
                    missing_ids = [int(x) for x in nums] if nums else []
                except Exception:
                    missing_ids = []

                # Build an empty DataFrame placeholder and attach missing ids for downstream logic
                imputation_results = pd.DataFrame()
                try:
                    imputation_results.attrs['missing_record_ids'] = missing_ids
                except Exception:
                    pass

                # Write diagnostics JSON atomically so downstream tooling can pick it up
                if missing_ids:
                    diag_path = f"{results_dir}/{dataset_name}_{part_name}_imputation_missing_ids{ids_suffix}.json"
                    try:
                        fd, tmp_path = tempfile.mkstemp(prefix=diag_path, suffix='.tmp')
                        with os.fdopen(fd, 'w') as dfh:
                            json.dump({'missing_count': len(missing_ids), 'missing_ids': missing_ids}, dfh)
                            dfh.flush()
                            os.fsync(dfh.fileno())
                        os.replace(tmp_path, diag_path)
                        msg = f"\n⚠️ Missing predictions (from exception) saved to {diag_path}"
                        print(msg)
                        with open(log_file, 'a') as f:
                            f.write(msg + '\n')
                    except Exception as e2:
                        print(f"Failed to write diagnostics JSON: {e2}")
            
            elapsed = time.time() - start_time
            msg = f"\n⏱️  Time elapsed: {elapsed/60:.2f} minutes"
            print(msg)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
            
            print_evaluation_summary(
                imputation_results, 
                f"{task['name'].upper()} IMPUTATION ({part_name})",
                log_file=log_file
            )
            
            # Save results (partition_suffix computed earlier; empty when not partitioned or single partition)
            results_filename = f"{results_dir}/{dataset_name}_{part_name}_imputation_results{suffix}.csv"
            imputation_results.to_csv(results_filename, index=False)
            msg = f"\n✓ Results saved to {results_filename}"
            print(msg)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
            # If some record ids were missing, write a diagnostics file for easy navigation
            try:
                missing_ids = imputation_results.attrs.get('missing_record_ids') if hasattr(imputation_results, 'attrs') else None
            except Exception:
                missing_ids = None
            if missing_ids:
                diag_path = f"{results_dir}/{dataset_name}_{part_name}_imputation_missing_ids{ids_suffix}.json"
                try:
                    fd, tmp_path = tempfile.mkstemp(prefix=diag_path, suffix='.tmp')
                    with os.fdopen(fd, 'w') as dfh:
                        json.dump({'missing_count': len(missing_ids), 'missing_ids': missing_ids}, dfh)
                        dfh.flush()
                        os.fsync(dfh.fileno())
                    os.replace(tmp_path, diag_path)
                    msg = f"\n⚠️ Missing predictions saved to {diag_path}"
                    print(msg)
                    with open(log_file, 'a') as f:
                        f.write(msg + '\n')
                except Exception as e:
                    print(f"Failed to write diagnostics JSON: {e}")
            files_created += 1
            
            # Save imputed dataset
            if imputation_results is not None and not imputation_results.empty:
                imputed_df = apply_imputation_results(
                    imputation_task['anon_data'], 
                    imputation_results,
                    log_file=log_file
                )
                # imputed filename uses the same partition_suffix
                imputed_filename = f"{results_dir}/{dataset_name}_{part_name}_imputed_dataset{suffix}.csv"
                imputed_df.to_csv(imputed_filename, index=False)
                msg = f"✓ Imputed dataset saved to {imputed_filename}"
                print(msg)
                with open(log_file, 'a') as f:
                    f.write(msg + '\n')
                files_created += 1
            
            results[f'{dataset_name}_{part_name}_imputation'] = imputation_results
        
        # PREDICTION EVALUATION
        msg = f"\n{'='*80}\n{task['name'].upper()} TARGET PREDICTION\n{'='*80}"
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
        
        for prediction_task in task['predictions']:
            part_name = prediction_task['part']
            
            start_time = time.time()
            
            prediction_results = await evaluate_prediction(
                prediction_task['anon_data'],
                prediction_task['original_data'],
                dataset_name=dataset_name,
                n_samples=n_samples,
                concurrency=concurrency,
                partition=partition,
                batch_size=batch_size
                ,specific_record_ids=specific_record_ids
            )
            
            elapsed = time.time() - start_time
            msg = f"\n⏱️  Time elapsed: {elapsed/60:.2f} minutes"
            print(msg)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
            
            print_evaluation_summary(
                prediction_results, 
                f"{task['name'].upper()} PREDICTION ({part_name})",
                log_file=log_file
            )
            
            # Save results (partition_suffix computed earlier)
            prediction_filename = f"{results_dir}/{dataset_name}_{part_name}_prediction_results{suffix}.csv"
            prediction_results.to_csv(prediction_filename, index=False)
            msg = f"\n✓ Results saved to {prediction_filename}"
            print(msg)
            with open(log_file, 'a') as f:
                f.write(msg + '\n')
            files_created += 1
            
            results[f'{dataset_name}_{part_name}_prediction'] = prediction_results
        
        msg = f"\n{'='*80}\n{task['name'].upper()} COMPLETE\n{'='*80}\n"
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg)
            f.write(f"\nLog file: {log_file}\n")
        # write a simple completion marker file so external monitors can detect finished datasets
        try:
            marker = f"{results_dir}/{dataset_name}_complete{suffix}.marker"
            with open(marker, 'w') as mf:
                mf.write(json.dumps({'dataset': dataset_name, 'completed_at': time.time()}))
                mf.flush()
        except Exception:
            pass
    
    # Final summary
    total_elapsed = time.time() - total_start_time
    
    print(f"\n{'#'*80}")
    print(f"ALL EVALUATIONS COMPLETE")
    print(f"{'#'*80}")
    print(f"\n✓ Processed {len(evaluation_tasks)} datasets ({', '.join([t['name'] for t in evaluation_tasks])})")
    print(f"✓ Generated {files_created} files")
    print(f"✓ Results directory: {results_dir}")
    print(f"\n⏱️  Total runtime: {total_elapsed/60:.2f} minutes ({total_elapsed:.1f} seconds)")
    
    return results


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='LLM Privacy Risk Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--percentage',
        type=str,
        default='33-33-34',
        help='Dataset split percentage (e.g., 33-33-34)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        default='Adult-Diabetes',
        help='Hyphen-separated list of datasets to evaluate (e.g., Adult-Diabetes)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=None,
        help='Number of samples per dataset (default: 100 for local, 1M for SLURM)'
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        default=20,
        help='Number of parallel API calls (default: 20)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory to save results (default: llm_evaluation/{percentage}_results)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='llm_evaluation/llm_test_data',
        help='Base input directory containing dataset folders (default: llm_evaluation/llm_test_data)'
    )
    parser.add_argument(
        '--results-base',
        type=str,
        default='../llm_evaluation',
        help='Base folder for results (default: llm_evaluation). Final results dir will be {results_base}/{percentage}_results'
    )
    parser.add_argument(
        '--partition',
        type=str,
        default=None,
        help='Process only a partition of the dataset (format: N/TOTAL, e.g., 1/4 for first quarter)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Number of records to process in a single LLM request (default: 1)'
    )
    parser.add_argument(
        '--record-ids',
        type=str,
        default=None,
        help='Comma-separated list of record ids to evaluate (e.g., 123,456,789)'
    )
    parser.add_argument(
        '--ids-file',
        type=str,
        default=None,
        help='Path to a file containing one record id per line to evaluate'
    )
    
    args = parser.parse_args()
    
    # Parse datasets
    datasets = [d.strip() for d in args.datasets.split('-')]
    
    # Determine n_samples
    if args.n_samples is None:
        n_samples = 1_000_000 if 'SLURM_JOB_ID' in os.environ else 100
    else:
        n_samples = args.n_samples
    
    # Determine results directory (results base is configurable; final dir is {results_base}/{percentage}_results)
    if args.results_dir is None:
        results_dir = os.path.join(args.results_base, f'{args.percentage}_results')
    else:
        results_dir = args.results_dir

    input_dir = args.input_dir

    # Parse optional specific record ids
    specific_record_ids = None
    if args.record_ids:
        try:
            specific_record_ids = [int(x) if x.isdigit() else x for x in args.record_ids.split(',') if x.strip()]
        except Exception:
            specific_record_ids = [x.strip() for x in args.record_ids.split(',') if x.strip()]
    elif args.ids_file:
        if os.path.exists(args.ids_file):
            if args.ids_file.endswith('.json'):
                try:
                    with open(args.ids_file) as fh:
                        data = json.load(fh)
                        # Support both the generated format {'missing_ids': [...]} and simple list [1, 2, 3]
                        if isinstance(data, dict) and 'missing_ids' in data:
                            specific_record_ids = data['missing_ids']
                        elif isinstance(data, list):
                            specific_record_ids = data
                        else:
                            print(f"Warning: JSON file {args.ids_file} does not contain a list or 'missing_ids' key")
                except Exception as e:
                    print(f"Error reading JSON ids file: {e}")
            else:
                with open(args.ids_file) as fh:
                    lines = [l.strip() for l in fh if l.strip()]
                    try:
                        specific_record_ids = [int(x) if x.isdigit() else x for x in lines]
                    except Exception:
                        specific_record_ids = lines
        else:
            print(f"Warning: ids file {args.ids_file} not found; ignoring")

    # Run evaluation
    asyncio.run(run_evaluation(
        percentage=args.percentage,
        datasets=datasets,
        n_samples=n_samples,
        concurrency=args.concurrency,
        results_dir=results_dir,
        input_dir=input_dir,
        partition=args.partition,
        batch_size=args.batch_size,
        specific_record_ids=specific_record_ids
    ))


if __name__ == '__main__':
    main()
