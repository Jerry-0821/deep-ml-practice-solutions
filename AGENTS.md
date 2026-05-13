# AGENTS.md

## Project Purpose

This repository contains my personal solutions to machine learning, deep learning, statistics, probability, and programming practice problems.

The goal is to build a clean GitHub portfolio that shows:

- My ability to implement algorithms from scratch
- My understanding of the mathematical ideas behind the code
- My progress in machine learning and data science practice
- Clean, readable, and well-documented Python code

This repository should look like a serious personal learning portfolio, not a copied answer bank.

## General Rules

1. Preserve my original solution logic unless there is a clear bug.
2. Do not rewrite code into a completely different style unless explicitly asked.
3. Do not copy official solutions or external answers.
4. Do not include full copyrighted problem descriptions.
5. Use short problem summaries instead of copying full prompts.
6. All code comments must be written in English.
7. Convert any Chinese comments into clear, natural English.
8. Comments should explain non-obvious logic, not every simple line.
9. Prefer beginner-friendly clarity over over-engineered code.
10. Keep function signatures unchanged unless explicitly asked.
11. Keep mathematical formulas accurate and readable.
12. Do not make the code look artificially advanced.
13. Do not change working code only for style.
14. If a solution already works, polish it carefully instead of replacing it.

## Standard Solution File Format

Each solution file should follow this structure when possible:

    """
    Problem:
    Short problem title or summary.

    Topic:
    Main topic, such as linear regression, clustering, probability, neural networks, etc.

    Idea:
    Briefly explain the core algorithm or mathematical idea.

    Key Steps:
    1. ...
    2. ...
    3. ...

    Complexity:
    Time: ...
    Space: ...
    """

    # imports

    # helper functions if needed

    # main function

## Commenting Rules

Use comments only when they help explain reasoning.

Good comments explain:

- Vectorized NumPy shapes
- Why a specific axis is used
- Algorithm steps
- Edge cases
- Numerical stability
- Non-obvious formulas
- Important implementation decisions

Bad comments:

- Repeat what the code obviously says
- Explain basic syntax
- Comment every line
- Add long paragraphs inside the code

Good example:

    # Compute the distance from each point to each centroid.
    distances = np.linalg.norm(points_array[:, None] - centroids[None, :], axis=2)

Bad example:

    # This is a for loop.
    for i in range(k):
        ...

## Copyright and Attribution Rules

Do not include:

- Full problem statements from practice websites
- Paid or premium content
- Official solutions
- Other people's code without attribution

It is acceptable to include:

- My own code
- My own explanation
- A short problem summary
- A link to the original problem page, if available
- My own notes about formulas, algorithms, and edge cases

## When Editing Existing Solution Files

Before editing:

1. Read the current file.
2. Identify the problem topic.
3. Identify whether the code already works.
4. Preserve the original algorithm unless there is a clear bug.
5. Check whether any comments are in Chinese.
6. Check whether the file has a useful docstring.
7. Check whether difficult logic needs short explanation.

When editing:

1. Convert Chinese comments into clear English.
2. Add or improve a short header docstring.
3. Keep the original function signature.
4. Keep the original algorithmic idea.
5. Rename variables only when it clearly improves readability.
6. Add comments only for difficult or important parts.
7. Remove unused imports.
8. Do not copy the full problem prompt.
9. Do not over-refactor.

After editing:

1. Check syntax.
2. Run available tests if they exist.
3. If there is no test system, do not create a large testing framework unless asked.
4. Summarize what changed briefly.

## Batch Refactor Rules

When asked to clean a folder of solution files:

1. Work file by file.
2. Preserve each solution's logic.
3. Convert comments into English.
4. Add short docstrings.
5. Do not add long explanations.
6. Do not change working solutions unnecessarily.
7. Do not ask follow-up questions unless blocked.
8. Summarize changed files at the end.

## Preferred Final Summary Format

After completing a task, respond briefly with:

- Files changed
- Main improvements
- Whether original logic was preserved
- Whether syntax checks or tests were run
