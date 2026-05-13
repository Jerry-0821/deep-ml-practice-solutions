---
name: ml-practice-solution-refactor
description: Use this skill when cleaning, organizing, documenting, or polishing personal machine learning, deep learning, statistics, probability, or programming practice solution files for a GitHub portfolio.
---

# ML Practice Solution Refactor Skill

## Goal

Refactor personal practice solution files into clean GitHub-ready code while preserving the user's original thinking and algorithmic approach.

The final result should look like a serious learning portfolio, not copied official solutions.

## Core Principles

1. Preserve the original solution logic.
2. Improve readability without over-engineering.
3. Convert all Chinese comments into natural English.
4. Add useful reasoning comments only where the logic is difficult.
5. Keep explanations concise.
6. Do not copy full problem statements.
7. Do not add unnecessary advanced abstractions.
8. Keep the code beginner-friendly but professional.
9. Keep function signatures unchanged unless explicitly asked.
10. Do not rewrite a working solution into a completely different implementation.

## Workflow

### Step 1: Understand the File

Read the file and identify:

- Problem topic
- Main algorithm
- Input format
- Output format
- Important formula
- Important edge cases
- Whether the current code already works

Do not edit blindly. First infer the intended algorithm from the code.

### Step 2: Add or Improve Header Docstring

Each solution should begin with a short docstring:

    """
    Problem:
    Short title or summary.

    Topic:
    Main ML / math / programming concept.

    Idea:
    Brief explanation of the algorithm.

    Key Steps:
    1. ...
    2. ...
    3. ...

    Complexity:
    Time: ...
    Space: ...
    """

Rules:

- Do not paste the full original problem statement.
- Keep it short.
- Focus on the algorithm and learning value.
- Mention formulas only when useful.

### Step 3: Clean Imports

Remove unused imports.

Do not introduce new libraries unless necessary.

### Step 4: Polish Comments

Convert Chinese comments into English.

Use comments for:

- Shape reasoning
- Vectorized operations
- Axis meaning
- Algorithm steps
- Edge cases
- Numerical stability
- Non-obvious formulas

Avoid comments for:

- Basic Python syntax
- Obvious assignments
- Every single line

Good comment examples:

    # Compute the distance from each point to each centroid.

    # Keep the previous centroid if no points are assigned to this cluster.

    # axis=1 selects the nearest centroid for each point.

Bad comment examples:

    # This is a loop.

    # Add item to list.

### Step 5: Preserve Function Signature

Do not change:

- Function name
- Parameter names
- Return type
- Required output format

unless the user explicitly asks.

### Step 6: Improve Variable Names Carefully

Improve unclear names only when it does not change the logic.

Examples:

- arr -> points_array
- dists -> distances
- res -> result
- idx -> cluster_index

Do not rename aggressively if it makes the diff too large.

### Step 7: Preserve the Algorithm

Do not replace the user's solution with a totally different one.

For example:

- If the user uses NumPy vectorization, keep the NumPy approach.
- If the user uses loops from scratch, keep the loop-based approach.
- If the user uses a simple beginner-friendly method, do not replace it with a complex advanced method.

Only change the algorithm if:

1. There is a clear bug, or
2. The user explicitly asks for optimization or a different approach.

### Step 8: Handle Edge Cases Carefully

If the original code already handles edge cases, preserve the behavior.

If an obvious edge case is missing, add it only if it is safe and consistent with the problem.

Common edge cases:

- Empty cluster
- Division by zero
- Numerical stability
- Invalid dimensions
- Empty input
- Ties in nearest-neighbor assignment
- Rounding format

Do not add complicated validation unless asked.

### Step 9: Keep Explanations Short

The code file should not become a textbook.

Prefer:

- Short docstring
- Short comments
- Clear variable names

Avoid:

- Long theory sections
- Full lecture notes
- Full problem statement
- Overly detailed mathematical derivations

### Step 10: Final Review

Before finishing, check:

- Code still solves the same problem
- Original logic is preserved
- Comments are in English
- No full copyrighted prompt is copied
- Header docstring exists
- Difficult logic is explained
- No unnecessary rewrite was done
- Syntax is valid

## Folder Organization Guidelines

If asked to organize files, prefer this structure:

    practice-solutions/
    ├── README.md
    ├── machine-learning/
    │   ├── easy/
    │   ├── medium/
    │   └── hard/
    ├── deep-learning/
    │   ├── easy/
    │   ├── medium/
    │   └── hard/
    ├── statistics-probability/
    │   ├── easy/
    │   ├── medium/
    │   └── hard/
    └── notes/

Do not move files unless explicitly asked.

## Output Summary

When done, summarize briefly:

- Files changed
- Main improvements
- Whether logic was preserved
- Whether tests or syntax checks were run

Example:

Done.

Changed:
- machine-learning/medium/example.py

Improvements:
- Converted comments to English
- Added a short docstring
- Explained the non-obvious NumPy axis logic
- Preserved the original algorithm

Checks:
- Syntax check passed
