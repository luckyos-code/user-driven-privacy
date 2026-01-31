"""
Utility functions for Dask operations.
"""
import pandas as pd


def count_partition(df):
    """
    Count rows in a partition, returning a Series for Dask compatibility.
    
    Returns:
        pd.Series: Series containing the count of rows in the partition
    """
    return pd.Series([len(df)])


def count_dask_rows(dask_df):
    """
    Efficiently count total rows in a Dask DataFrame using partition length metadata.
    This approach computes only small integers (row counts per partition), not the actual data.
    
    Args:
        dask_df: Dask DataFrame to count
        
    Returns:
        int: Total number of rows
    """
    # Map each partition to its length, returning a pandas Series
    # Then reduce by summing - much faster than computing actual data
    partition_lengths = dask_df.map_partitions(count_partition, meta=pd.Series(dtype='int64'))
    total = partition_lengths.sum().compute()
    return int(total)
