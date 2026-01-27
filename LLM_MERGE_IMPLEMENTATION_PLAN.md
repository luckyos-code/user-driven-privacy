# Implementation Plan: LLM-Enhanced Observed Values for Specialization

## Context

**Current Implementation:**
- Realistic specialization mode extracts observed values from original (non-generalized) data
- Only applies to **categorical columns** (numerical columns use mean imputation)
- Default: `limit_to_observed_values=True` in `Vorverarbeitung.py`

## Key Insight

✅ **Numerical columns don't need observed values** - they use `ersetze_durch_mittelwert()` (mean imputation)  
✅ **Categorical columns need observed values** - they use `erstelle_neue_zeilen()` (specialization)

## Analysis Results

**Categorical columns in Adult dataset:**
- `education`, `marital-status`, `native-country`, `occupation`, `race`, `relationship`, `sex`, `workclass`

**LLM data shows:**
- ✅ 213 "valid" new values across all 8 categorical columns
- ❌ BUT: Many are format variations or questionable (e.g., "11th-grade" in occupation, "husband" in sex)
- ⚠️ Validation heuristics caught obvious cross-column contamination but not all issues

## Quality Concerns

**Problems with LLM values:**
1. **Format variations**: `"Married-civ-spouse"` vs `"married"` vs `"married_civ_spouse"`
2. **Semantic issues**: `"11th-grade"` appearing in `occupation` (should be in `education`)
3. **Gender confusion**: `"husband"` appearing in `sex` column (should be in `relationship`)
4. **Massive inflation**: `occupation` 14→121 values (764% increase!)

**Examples of questionable "valid" values:**
- `education`: Format variants like `"10th grade"`, `"10th-grade"`, `"10th_grade"` (same concept, 3 variants)
- `occupation`: Contains `"11th-grade"`, `"25-34"`, `"High-School"` (clearly wrong column)
- `marital-status`: Contains `"husband"`, `"wife"` (these are relationships, not marital status)

## Implementation Plan

### Phase 1: Add LLM Merge Parameter (Conservative Approach)

**Goal:** Allow optional merging of LLM values with validation

**Changes needed:**

1. **Update `specialize_data_and_save_to_csv()` signature:**
   ```python
   def specialize_data_and_save_to_csv(
       df: pd.DataFrame, 
       folder_name: str, 
       dataset_name, 
       data_dir='data', 
       percentages: str = None, 
       extended: bool = False, 
       limit_to_observed_values: bool = True,
       merge_llm_values: bool = False  # NEW parameter
   ):
   ```

2. **Extract observed values (existing logic):**
   ```python
   observed_values_dict = {}
   for column in spalten_list:
       if column.name in df.columns and column.name not in numerical_columns:
           # Get original observed values
           original_values = df[column.name][...].unique()
           observed_values_dict[column.name] = set(original_values)
   ```

3. **Merge with LLM values (NEW logic):**
   ```python
   if merge_llm_values and limit_to_observed_values:
       llm_path = os.path.join(data_dir, '..', 'llm_evaluation', percentages, 
                               f'{dataset_name}_train_imputed_dataset.csv')
       if os.path.exists(llm_path):
           llm_df = pd.read_csv(llm_path)
           
           for column in spalten_list:
               if column.name in observed_values_dict and column.name in llm_df.columns:
                   # Extract LLM values for this categorical column
                   llm_values = set(llm_df[column.name][
                       llm_df[column.name].notna() &
                       (llm_df[column.name] != '?') &
                       (llm_df[column.name] != '')
                   ].astype(str).unique())
                   
                   # Merge with set union (automatic deduplication)
                   original_count = len(observed_values_dict[column.name])
                   observed_values_dict[column.name] = observed_values_dict[column.name] | llm_values
                   new_count = len(observed_values_dict[column.name])
                   
                   if new_count > original_count:
                       print(f"  {column.name}: {original_count} → {new_count} values (+{new_count - original_count} from LLM)")
   ```

4. **Update function signatures up the chain:**
   - `prepare_specialization()`: Add `merge_llm_values=False` parameter
   - `prepare_extended_specialization()`: Add `merge_llm_values=False` parameter
   - `create_dataset_versions()`: Add `merge_llm_values=False` parameter
   - `run.py`: Add `--merge-llm-values` argparse flag

### Phase 2: Validation Layer (Recommended)

**Goal:** Filter out invalid/contaminated LLM values

**Validation strategies:**

1. **Schema validation:**
   ```python
   # Define expected value patterns per column
   COLUMN_VALIDATORS = {
       'education': lambda v: 'grade' in v.lower() or 'school' in v.lower() or ...,
       'marital-status': lambda v: v not in ['husband', 'wife', 'Male', 'Female'],
       'occupation': lambda v: not v.isdigit() and not any(x in v for x in ['grade', '25-34']),
       # etc.
   }
   ```

2. **Cross-column contamination check:**
   ```python
   # Check if value appears in OTHER columns' original values
   all_other_column_values = set()
   for other_col in spalten_list:
       if other_col.name != column.name:
           all_other_column_values.update(original_df[other_col.name].unique())
   
   # Filter out contaminated values
   llm_values = {v for v in llm_values if v not in all_other_column_values}
   ```

3. **Format normalization:**
   ```python
   # Normalize format before merging
   def normalize_value(v):
       return v.lower().replace('_', '-').strip()
   
   observed_values_dict[column.name] = {normalize_value(v) for v in observed_values_dict[column.name]}
   llm_values = {normalize_value(v) for v in llm_values}
   ```

### Phase 3: Testing & Evaluation

**Test scenarios:**

1. **Without LLM merge (baseline):**
   ```bash
   python run.py --dataset adult --percentages "33 33 34" ... 
   # Uses only observed values from original data
   ```

2. **With LLM merge (experimental):**
   ```bash
   python run.py --dataset adult --percentages "33 33 34" --merge-llm-values
   # Merges LLM values with observed values
   ```

3. **Compare results:**
   - Row counts after specialization
   - Runtime for filtering
   - Model accuracy
   - Privacy metrics (k-anonymity, l-diversity)

**Expected outcomes:**

- ✅ **If validation works well:** Moderate increase in observed values (10-30%), better coverage
- ❌ **If validation fails:** Massive explosion (200-700%), contaminated data, worse results
- ⚠️ **Most likely:** Mixed results, some columns benefit, others get polluted

## Recommendation

### Option A: Conservative (Recommended for now)

**DO NOT implement LLM merge** until:
1. LLM evaluation pipeline is fixed to prevent cross-column contamination
2. Format normalization is implemented
3. Schema validation is in place

**Rationale:**
- Current LLM data has serious quality issues
- Risk of polluting specialization with garbage values
- Better to use fewer high-quality values than many contaminated values

### Option B: Experimental (If you want to explore)

**Implement with strong validation:**
1. Add `--merge-llm-values` flag (default: False)
2. Implement cross-column contamination check
3. Add format normalization
4. Log all merged values for manual inspection
5. Compare results with/without merge

**Test on low-stakes scenario:**
- Small dataset (adult, not diabetes)
- High original_pct (66-17-17, not 0-66-34)
- Manual inspection of merged values before running full pipeline

## Decision Tree

```
Do you need more observed values?
├─ No → Use current implementation (observed values only)
│
└─ Yes → Is LLM data quality good?
    ├─ No → Fix LLM evaluation pipeline first
    │
    └─ Yes/Maybe → Implement merge with validation
        ├─ Start with conservative validation
        ├─ Test on small scale
        ├─ Inspect results manually
        └─ Expand if results are good
```

## Files to Modify

1. **`src/Vorverarbeitung.py`:**
   - `specialize_data_and_save_to_csv()`: Add LLM merge logic
   - `prepare_specialization()`: Add parameter
   - `prepare_extended_specialization()`: Add parameter

2. **`src/DatasetCreation.py`:**
   - `create_dataset_versions()`: Add parameter (optional, can skip)

3. **`run.py`:**
   - Add `--merge-llm-values` argparse flag (optional, can skip)

4. **NEW: `src/LLMValueValidator.py`:**
   - Validation logic
   - Cross-column contamination detection
   - Format normalization

## Next Steps

**Question for you:** How do you want to proceed?

A. **Skip LLM merge entirely** - Current implementation is good enough
B. **Implement basic merge** - Simple set union, no validation (risky!)
C. **Implement with validation** - Safe approach, more work
D. **Fix LLM pipeline first** - Address root cause, then revisit

My recommendation: **Option A or D** - Either skip it or fix the LLM pipeline first.
