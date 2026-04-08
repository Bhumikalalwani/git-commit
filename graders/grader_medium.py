from __future__ import annotations

from typing import List


def lcs_similarity(seq1: List[str], seq2: List[str]) -> float:
    """Longest common subsequence normalized by the length of the reference."""
    m, n = len(seq1), len(seq2)
    if n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n] / n


def grade_structured(predicted_timeline: List[str], ideal_order: List[str]) -> float:
    """Grade the structured editing task (medium).

    Uses LCS similarity to measure how close the predicted sequence
    matches the ideal ordering.  Returns a score in [0.0, 1.0].
    """
    return round(lcs_similarity(predicted_timeline, ideal_order), 4)
