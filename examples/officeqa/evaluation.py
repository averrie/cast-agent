"""
OfficeQA reward function - fuzzy numerical matching.

Usage:
    from examples.officeqa.evaluation import score_answer
    score = score_answer(ground_truth, predicted, tolerance)
"""

import re


def normalize_text(text: str) -> str:
    if not text:
        raise ValueError("Cannot normalize empty or None text")
    normalized = text.replace("\u2212", "-")
    normalized = normalized.replace("\u2212", "-")
    return normalized


def extract_numbers_with_context(text: str) -> list[tuple[float, str, bool, bool]]:
    if not text:
        raise ValueError("Cannot extract numbers from empty text")
    text = normalize_text(text)
    text_no_commas = re.sub(
        r"\d{1,3}(?:,\d{3})+(?:\.\d+)?",
        lambda m: m.group().replace(",", ""),
        text,
    )
    numbers_with_context = []
    pattern = r"-?\d+\.?\d*%?"
    for match in re.finditer(pattern, text_no_commas):
        matched_text = match.group()
        if not matched_text or matched_text == "-":
            continue
        has_percent = matched_text.endswith("%")
        num_text = matched_text.rstrip("%")
        is_negative = num_text.startswith("-")
        try:
            num = float(num_text)
        except ValueError as e:
            raise ValueError(
                f"Failed to parse number from '{matched_text}': {e}"
            ) from e
        start = max(0, match.start() - 20)
        end = min(len(text_no_commas), match.end() + 20)
        context = text_no_commas[start:end].lower()
        numbers_with_context.append((num, context, has_percent, is_negative))
    return numbers_with_context


def detect_unit_in_context(context: str) -> tuple[str | None, float]:
    context_lower = context.lower()
    if re.search(r"\btrillions?\b", context_lower):
        return ("trillion", 1e12)
    if re.search(r"\bbillions?\b", context_lower) or re.search(r"\bb\b", context_lower):
        return ("billion", 1e9)
    if re.search(r"\bmillions?\b", context_lower) or re.search(r"\bm\b", context_lower):
        return ("million", 1e6)
    if re.search(r"\bthousands?\b", context_lower) or re.search(
        r"\bk\b", context_lower
    ):
        return ("thousand", 1e3)
    return (None, 1.0)


def normalize_number_with_units(
    number: float, context: str
) -> tuple[float, str | None]:
    try:
        unit_name, multiplier = detect_unit_in_context(context)
        return (number * multiplier, unit_name)
    except Exception as e:
        raise ValueError(
            f"Failed to normalize number {number} with context '{context}': {e}"
        ) from e


def _within_tolerance(gt_base: float, pred_base: float, tolerance: float) -> bool:
    if gt_base == 0:
        return pred_base == 0
    diff_pct = abs(gt_base - pred_base) / abs(gt_base)
    return diff_pct <= tolerance


def _one_to_one_match(
    gt_values: list[float],
    pred_values: list[float],
    tolerance: float,
) -> tuple[int, list[int]]:
    graph: list[list[int]] = []
    for gt_base in gt_values:
        matches = []
        for j, pred_base in enumerate(pred_values):
            if _within_tolerance(gt_base, pred_base, tolerance):
                matches.append(j)
        graph.append(matches)

    match_to_gt = [-1] * len(pred_values)

    def _dfs(gt_idx: int, seen: list[bool]) -> bool:
        for pred_idx in graph[gt_idx]:
            if seen[pred_idx]:
                continue
            seen[pred_idx] = True
            prev_gt = match_to_gt[pred_idx]
            if prev_gt == -1 or _dfs(prev_gt, seen):
                match_to_gt[pred_idx] = gt_idx
                return True
        return False

    matched = 0
    for gt_idx in range(len(gt_values)):
        seen = [False] * len(pred_values)
        if _dfs(gt_idx, seen):
            matched += 1

    matched_gt_indices = sorted({idx for idx in match_to_gt if idx != -1})
    return matched, matched_gt_indices


def is_likely_year(num: float) -> bool:
    return 1900 <= num <= 2100 and num == int(num)


def has_significant_text(text: str) -> tuple[bool, str]:
    if not text:
        return False, ""
    cleaned = normalize_text(text).lower()
    cleaned = re.sub(r"-?\d+\.?\d*%?", "", cleaned)
    cleaned = re.sub(r"[,]", "", cleaned)
    unit_words = [
        "trillion",
        "trillions",
        "billion",
        "billions",
        "million",
        "millions",
        "thousand",
        "thousands",
        "hundred",
        "hundreds",
        "percent",
        "percentage",
        "%",
    ]
    for unit in unit_words:
        cleaned = re.sub(r"\b" + unit + r"\b", "", cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    has_text = len(cleaned) >= 2
    return has_text, cleaned


def check_text_overlap(gt_text: str, pred_text: str) -> tuple[bool, str]:
    if not gt_text or not pred_text:
        return False, "Empty text in comparison"
    gt_has_text, gt_cleaned = has_significant_text(gt_text)
    pred_has_text, pred_cleaned = has_significant_text(pred_text)
    if not gt_has_text:
        return True, "GT is purely numeric, text check not required"
    if not pred_has_text:
        return False, f"GT has text '{gt_cleaned}' but prediction is purely numeric"
    if gt_cleaned in pred_cleaned:
        return True, f"Text overlap: '{gt_cleaned}' found in prediction"
    if pred_cleaned in gt_cleaned:
        return True, f"Text overlap: prediction text '{pred_cleaned}' matches GT"
    return False, f"Text mismatch: GT='{gt_cleaned}', Pred='{pred_cleaned}'"


def extract_final_answer(text: str) -> str:
    if not text:
        raise ValueError("Cannot extract from empty text")
    match = re.search(
        r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>", text, re.DOTALL | re.IGNORECASE
    )
    if match:
        content = match.group(1).strip()
        if not content:
            raise ValueError("FINAL_ANSWER tags are empty")
        return content
    return text


def fuzzy_match_answer(
    ground_truth: str, predicted: str, tolerance: float = 0.05
) -> tuple[bool, str]:
    if not ground_truth:
        raise ValueError("Ground truth cannot be empty")
    if not predicted:
        raise ValueError("Predicted answer cannot be empty")
    if not 0 <= tolerance <= 1:
        raise ValueError(f"Tolerance must be between 0 and 1, got {tolerance}")

    try:
        gt_numbers_with_context = extract_numbers_with_context(ground_truth)
        pred_numbers_with_context = extract_numbers_with_context(predicted)
    except Exception as e:
        raise ValueError(f"Failed to extract numbers: {e}") from e

    gt_numbers = [(num, ctx) for num, ctx, _, _ in gt_numbers_with_context]
    pred_numbers = [(num, ctx) for num, ctx, _, _ in pred_numbers_with_context]

    if gt_numbers and pred_numbers:
        if len(gt_numbers) > 1:
            pred_non_years = [
                (n, c)
                for n, c in pred_numbers
                if not is_likely_year(n)
                or any(is_likely_year(g) for g, _ in gt_numbers)
            ]
            text_matches, text_rationale = check_text_overlap(ground_truth, predicted)
            if not text_matches:
                return False, f"List mismatch: {text_rationale}"

            gt_values = [
                normalize_number_with_units(gt_val, gt_context)[0]
                for gt_val, gt_context in gt_numbers
            ]
            pred_values = [
                normalize_number_with_units(pred_val, pred_context)[0]
                for pred_val, pred_context in pred_non_years
            ]

            if not pred_values:
                return False, "List mismatch: No valid numbers found in prediction"

            matched_count, matched_gt_indices = _one_to_one_match(
                gt_values,
                pred_values,
                tolerance,
            )
            if matched_count == len(gt_numbers):
                return (
                    True,
                    f"List match: All {len(gt_numbers)} numbers found in prediction",
                )
            else:
                unmatched_gt = [
                    gt_numbers[i][0]
                    for i in range(len(gt_numbers))
                    if i not in matched_gt_indices
                ]
                return False, (
                    f"List mismatch: Found {matched_count}/{len(gt_numbers)} numbers. "
                    f"Missing: {unmatched_gt}"
                )
        else:
            gt_val, gt_context = gt_numbers[0]
            gt_base, gt_unit = normalize_number_with_units(gt_val, gt_context)
            gt_has_text, _ = has_significant_text(ground_truth)
            should_filter_years = not (is_likely_year(gt_val) or gt_has_text)
            best_match = None
            best_diff = float("inf")
            best_pred_info = None
            for pred_val, pred_context in pred_numbers:
                if should_filter_years and is_likely_year(pred_val):
                    continue
                pred_base, pred_unit = normalize_number_with_units(
                    pred_val, pred_context
                )
                if gt_base == 0:
                    if pred_base == 0:
                        text_matches, text_rationale = check_text_overlap(
                            ground_truth, predicted
                        )
                        if text_matches:
                            return (
                                True,
                                f"Exact match: Found 0 in response. {text_rationale}",
                            )
                    continue
                diff_pct = abs(gt_base - pred_base) / abs(gt_base)
                if diff_pct < best_diff:
                    best_diff = diff_pct
                    best_match = pred_base
                    best_pred_info = (pred_base, pred_unit)
                if diff_pct <= tolerance:
                    text_matches, text_rationale = check_text_overlap(
                        ground_truth, predicted
                    )
                    if not text_matches:
                        continue
                    return (
                        True,
                        f"Numerical match: GT={gt_base} ({gt_unit or 'no unit'}), Pred={pred_base} ({pred_unit or 'no unit'}), Diff={diff_pct*100:.2f}%. {text_rationale}",
                    )
            if best_match is not None:
                return (
                    False,
                    f"No match: GT={gt_base} ({gt_unit or 'no unit'}), Closest={best_pred_info[0]} ({best_pred_info[1] or 'no unit'}), Diff={best_diff*100:.2f}%",
                )
            else:
                return (
                    False,
                    f"No valid numbers found in prediction (filtered out years: {[n for n, _ in pred_numbers[:5]]})",
                )

    gt_clean = ground_truth.strip().lower().strip('"').strip("'")
    pred_clean = predicted.strip().lower().strip('"').strip("'")
    gt_clean = re.sub(r"\([^)]*\)", "", gt_clean).strip()
    pred_clean = re.sub(r"\([^)]*\)", "", pred_clean).strip()
    if gt_clean in pred_clean:
        return True, f"Text match: '{ground_truth}' found in prediction"
    if gt_clean == pred_clean:
        return True, "Exact text match"
    return (
        False,
        f"No match found. GT: '{ground_truth[:100]}', Pred: '{predicted[:100]}'",
    )


def score_answer(ground_truth: str, predicted: str, tolerance: float = 0.00) -> float:
    is_correct, rationale = fuzzy_match_answer(ground_truth, predicted, tolerance)
    return 1.0 if is_correct else 0.0
