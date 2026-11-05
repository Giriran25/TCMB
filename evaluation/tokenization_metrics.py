"""
TCMB - Tokenization Evaluation Metrics

Measures:
1. Raw XLM-R sequence length.
2. Standalone SentencePiece marker frequency.
3. Content subword fragmentation.
4. Unicode grapheme-normalized tokenization statistics.
5. Whitespace-based word fertility where applicable.

Important:
- Raw token counts include every token processed by XLM-R.
- Content token counts exclude standalone SentencePiece marker tokens.
- Chinese and Japanese are excluded from whitespace-based word fertility.
"""

from typing import Any, Mapping, Sequence

import numpy as np
import regex


# Languages that do not generally use whitespace
# to separate ordinary words.
NON_WHITESPACE_SEGMENTED_LANGUAGES = {
    "cmn_Hans",  # Simplified Chinese
    "jpn_Jpan",  # Japanese
}


def count_graphemes(text: str) -> int:
    """
    Count Unicode grapheme clusters, excluding whitespace.

    A grapheme cluster approximates a user-perceived character.
    This is more appropriate than counting raw Unicode code points
    for scripts that use combining marks.

    Parameters
    ----------
    text : str
        Input sentence.

    Returns
    -------
    int
        Number of non-whitespace grapheme clusters.
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    clusters = regex.findall(r"\X", text)

    return sum(
        1
        for cluster in clusters
        if not cluster.isspace()
    )


def calculate_sentence_metrics(
    text: str,
    token_count: int,
    marker_count: int,
    language_code: str
) -> dict[str, Any]:
    """
    Calculate tokenization metrics for an individual sentence.

    Parameters
    ----------
    text : str
        Original input sentence.

    token_count : int
        Total number of XLM-R subword tokens.
        Includes standalone SentencePiece markers.

    marker_count : int
        Number of standalone SentencePiece marker tokens
        (token ID corresponding to '▁').

    language_code : str
        FLORES+ language identifier.

    Returns
    -------
    dict
        Sentence-level tokenization metrics.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if token_count < 0:
        raise ValueError("token_count cannot be negative.")

    if marker_count < 0 or marker_count > token_count:
        raise ValueError(
            "marker_count must be between 0 and token_count."
        )

    # Count non-whitespace Unicode grapheme clusters.
    grapheme_count = count_graphemes(text)

    # Keep actual model sequence length unchanged.
    raw_token_count = token_count

    # Exclude standalone SentencePiece markers only
    # for the lexical fragmentation analysis.
    content_token_count = token_count - marker_count

    # -------------------------------------------------
    # 1. Raw token-to-grapheme ratio
    # -------------------------------------------------

    raw_tokens_per_grapheme = (
        raw_token_count / grapheme_count
        if grapheme_count > 0
        else np.nan
    )

    # -------------------------------------------------
    # 2. Content token-to-grapheme ratio
    # -------------------------------------------------

    content_tokens_per_grapheme = (
        content_token_count / grapheme_count
        if grapheme_count > 0
        else np.nan
    )

    # -------------------------------------------------
    # 3. Graphemes per content subword
    # -------------------------------------------------

    graphemes_per_content_subword = (
        grapheme_count / content_token_count
        if content_token_count > 0
        else np.nan
    )

    # -------------------------------------------------
    # 4. Whitespace-based word fertility
    # -------------------------------------------------

    if language_code in NON_WHITESPACE_SEGMENTED_LANGUAGES:

        whitespace_word_count = np.nan

        content_subwords_per_whitespace_word = np.nan

    else:

        words = text.split()

        whitespace_word_count = len(words)

        content_subwords_per_whitespace_word = (
            content_token_count / whitespace_word_count
            if whitespace_word_count > 0
            else np.nan
        )

    return {
        "subword_tokens": raw_token_count,

        "standalone_marker_tokens": marker_count,

        "content_subword_tokens": content_token_count,

        "graphemes": grapheme_count,

        "raw_tokens_per_grapheme": raw_tokens_per_grapheme,

        "content_tokens_per_grapheme": content_tokens_per_grapheme,

        "graphemes_per_content_subword": graphemes_per_content_subword,

        "whitespace_word_count": whitespace_word_count,

        "content_subwords_per_whitespace_word": (
            content_subwords_per_whitespace_word
        ),
    }


def _safe_mean(values: np.ndarray) -> float:
    """Calculate mean safely for an array."""

    if len(values) == 0:
        return np.nan

    return float(np.mean(values))


def _safe_median(values: np.ndarray) -> float:
    """Calculate median safely for an array."""

    if len(values) == 0:
        return np.nan

    return float(np.median(values))


def _safe_percentile(
    values: np.ndarray,
    percentile: float
) -> float:
    """Calculate percentile safely for an array."""

    if len(values) == 0:
        return np.nan

    return float(np.percentile(values, percentile))


def summarize_language_metrics(
    sentence_records: Sequence[Mapping[str, Any]],
    language_code: str,
    language_name: str,
    split_name: str
) -> dict[str, Any]:
    """
    Aggregate sentence-level measurements into language-level
    summary statistics.

    Parameters
    ----------
    sentence_records : Sequence[Mapping]
        Sentence-level metric records generated by
        calculate_sentence_metrics().

    language_code : str
        FLORES+ language identifier.

    language_name : str
        Human-readable language name.

    split_name : str
        Dataset split, e.g. dev or devtest.

    Returns
    -------
    dict
        Aggregated language-level metrics.
    """

    if not sentence_records:
        raise ValueError(
            f"No sentence records provided for {language_name}."
        )

    # -------------------------------------------------
    # Extract sentence-level measurements
    # -------------------------------------------------

    tokens = np.array(
        [record["subword_tokens"] for record in sentence_records],
        dtype=float
    )

    content_tokens = np.array(
        [
            record["content_subword_tokens"]
            for record in sentence_records
        ],
        dtype=float
    )

    markers = np.array(
        [
            record["standalone_marker_tokens"]
            for record in sentence_records
        ],
        dtype=float
    )

    graphemes = np.array(
        [record["graphemes"] for record in sentence_records],
        dtype=float
    )

    # -------------------------------------------------
    # Calculate totals
    # -------------------------------------------------

    sentence_count = len(sentence_records)

    total_tokens = int(np.sum(tokens))

    total_content_tokens = int(np.sum(content_tokens))

    total_markers = int(np.sum(markers))

    total_graphemes = int(np.sum(graphemes))

    # -------------------------------------------------
    # Calculate marker statistics
    # -------------------------------------------------

    marker_share = (
        total_markers / total_tokens
        if total_tokens > 0
        else np.nan
    )

    markers_per_sentence = (
        total_markers / sentence_count
        if sentence_count > 0
        else np.nan
    )

    # -------------------------------------------------
    # Calculate word-level statistics
    # Only applicable to whitespace-segmented languages.
    # -------------------------------------------------

    valid_word_records = [
        record
        for record in sentence_records
        if not np.isnan(record["whitespace_word_count"])
    ]

    total_whitespace_words = sum(
        record["whitespace_word_count"]
        for record in valid_word_records
    )

    total_word_content_tokens = sum(
        record["content_subword_tokens"]
        for record in valid_word_records
    )

    if total_whitespace_words > 0:

        content_subwords_per_whitespace_word = (
            total_word_content_tokens / total_whitespace_words
        )

    else:

        content_subwords_per_whitespace_word = np.nan

    # -------------------------------------------------
    # Calculate language-level grapheme ratios
    # -------------------------------------------------

    raw_tokens_per_grapheme = (
        total_tokens / total_graphemes
        if total_graphemes > 0
        else np.nan
    )

    content_tokens_per_grapheme = (
        total_content_tokens / total_graphemes
        if total_graphemes > 0
        else np.nan
    )

    graphemes_per_content_subword = (
        total_graphemes / total_content_tokens
        if total_content_tokens > 0
        else np.nan
    )

    # -------------------------------------------------
    # Sequence length statistics
    # -------------------------------------------------

    mean_subwords_per_sentence = _safe_mean(tokens)

    median_subwords_per_sentence = _safe_median(tokens)

    p95_subwords_per_sentence = _safe_percentile(
        tokens,
        95
    )

    mean_content_subwords_per_sentence = _safe_mean(
        content_tokens
    )

    mean_graphemes_per_sentence = _safe_mean(
        graphemes
    )

    # -------------------------------------------------
    # Final summary
    # -------------------------------------------------

    summary = {

        "language_code": language_code,

        "language": language_name,

        "split": split_name,

        "sentence_count": sentence_count,

        # ---------------------------------------------
        # Actual XLM-R sequence statistics
        # ---------------------------------------------

        "total_subword_tokens": total_tokens,

        "mean_subwords_per_sentence": (
            mean_subwords_per_sentence
        ),

        "median_subwords_per_sentence": (
            median_subwords_per_sentence
        ),

        "p95_subwords_per_sentence": (
            p95_subwords_per_sentence
        ),

        # ---------------------------------------------
        # Standalone SentencePiece marker statistics
        # ---------------------------------------------

        "total_standalone_markers": total_markers,

        "markers_per_sentence": markers_per_sentence,

        "marker_token_share": marker_share,

        # ---------------------------------------------
        # Content token statistics
        # ---------------------------------------------

        "total_content_subword_tokens": (
            total_content_tokens
        ),

        "mean_content_subwords_per_sentence": (
            mean_content_subwords_per_sentence
        ),

        # ---------------------------------------------
        # Grapheme statistics
        # ---------------------------------------------

        "total_graphemes": total_graphemes,

        "mean_graphemes_per_sentence": (
            mean_graphemes_per_sentence
        ),

        # ---------------------------------------------
        # Fragmentation metrics
        # ---------------------------------------------

        "raw_tokens_per_grapheme": (
            raw_tokens_per_grapheme
        ),

        "content_tokens_per_grapheme": (
            content_tokens_per_grapheme
        ),

        "graphemes_per_content_subword": (
            graphemes_per_content_subword
        ),

        # ---------------------------------------------
        # Whitespace-based word fertility
        # ---------------------------------------------

        "content_subwords_per_whitespace_word": (
            content_subwords_per_whitespace_word
        ),
    }

    return summary