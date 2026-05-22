"""
Problem:
Prefix Cache Hit Rate Calculator.

Topic:
Machine learning implementation.

Idea:
Implement the requested computation directly while preserving the submitted Deep-ML solution logic.

Key Steps:
1. Prepare the input values in the expected shape or type.
2. Apply the core formula or update rule from the solution.
3. Return the computed result.

Complexity:
Time: O(n), where n is the number of processed values.
Space: O(n) for returned values or intermediate arrays.
"""

def prefix_cache_hit_rate(prompts: list, cache: list) -> dict:
    """
    Calculate the prefix cache hit rate for a batch of tokenized prompts.

    Args:
        prompts: List of tokenized prompts (list of list of ints).
        cache: List of cached prefixes (list of list of ints).

    Returns:
        Dictionary with 'hit_rate', 'cached_tokens', and 'total_tokens'.
    """
    total_tokens = sum(len(p) for p in prompts)
    cached_tokens = 0

    for prompt in prompts:
        best = 0
        for prefix in cache:
            n = len(prefix)
            if n > best and prompt[:n] == prefix:
                best = n
        cached_tokens += best
    hit_rate = round(cached_tokens / total_tokens, 4) if total_tokens > 0 else 0.0

    return {"hit_rate": hit_rate, "cached_tokens": cached_tokens, "total_tokens": total_tokens}
