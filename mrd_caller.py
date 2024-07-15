import argparse
import logging
import math
import numpy as np
import pandas as pd
from datetime import date
from scipy.stats import chi2
from typing import Optional, Tuple

MRD_CALLER_VERSION = "1.2"
ERROR_RATE_THRESHOLD_GENERAL = 100000
ERROR_RATE_THRESHOLD_C_T = 100000

TODAY = date.today()


def calculate_ts(total_depth: np.array, k: np.array) -> np.array:
    def st_approx(n):
        # approximate formula retrieved from
        # https://en.wikipedia.org/wiki/Stirling%27s_approximation#Speed_of_convergence_and_error_estimates
        return n * np.log(n) - n + 0.5 * np.log(2 * np.pi * n) + 1 / (12 * n)

    def get_log_factorial(n):
        rv = 0
        try:
            rv = np.log(math.factorial(n)) if n < 5 else st_approx(n)
        except ValueError:
            logging.warning(f"ValueError: math domain error for {n}. using 0 instead")
            logging.warning(f"\ttotal_depth  at index: {total_depth[idx]} ")
            logging.warning(f"\tk at index: {k[idx]}")
        return rv

    ancillary_term = np.zeros(len(total_depth))
    for idx in range(len(total_depth)):
        a1 = get_log_factorial(total_depth[idx])
        a2 = get_log_factorial(total_depth[idx] - k[idx])
        a3 = get_log_factorial(k[idx])
        ancillary_term[idx] = a1 - a2 - a3

    return ancillary_term


def get_bare_log_like(
    sum_dp: np.array, k: np.array, e: np.array, af: float = 0.0, fp_err: float = 1e-60
) -> float:

    cummulatives = []
    for idx in range(len(sum_dp)):
        p = af + e[idx] + fp_err
        cummulatives.append(k[idx] * np.log(p) + (sum_dp[idx] - k[idx]) * np.log(1 - p))
    return sum(cummulatives)


def calculate_log_like(
    sum_dp: np.array,
    k: np.array,
    e: np.array,
    ancillary_term: np.array,
    af: float = 0.5,
    fp_err: float = 1e-60,
) -> float:
    return get_bare_log_like(sum_dp, k, e, af=af, fp_err=fp_err) + sum(ancillary_term)


def apply_bisection(a: float, b: float, val: float) -> Tuple[float, float]:
    if val == 0:
        return (a + b) / 2, (a + b) / 2
    if val > 0:
        return (a + b) / 2, b
    return a, (a + b) / 2


def find_log_like_min(
    sum_dp: np.array,
    k: np.array,
    e: np.array,
    steps: int = 40,
    min_af: int = 0,
    fp_err: float = 1e-60,
) -> float:
    max_min_af = (min_af, 1 - min(e) - fp_err)
    cand_af = list(max_min_af)
    for _ in range(steps):
        af_test = sum(cand_af) / 2
        deriv_log_like = 0
        for idx in range(len(sum_dp)):
            p = af_test + e[idx] + fp_err
            numer = k[idx] * (1 - p) - (sum_dp[idx] - k[idx]) * p
            denom = p * (1 - p)
            term = numer / denom
            deriv_log_like += term
        cand_af = apply_bisection(cand_af[0], cand_af[1], deriv_log_like)
    return sum(cand_af) / 2


def find_the_cis(
    sum_dp: np.array,
    k: np.array,
    e: np.array,
    af0: float,
    specificity: float = 0.99,
    confidence: float = 0.95,
    af_ci: Tuple[int, float] = (0, 1 - 1e-10),
    steps: int = 60,
    fp_err: float = 1e-60,
) -> Tuple[float, float]:
    growth_factor = 2
    log_like_gap = chi2.ppf(confidence, 1) / 2
    log_like_chernoff_gap = chi2.ppf(2 * specificity - 1, 1) / 2

    def find_af_0_upper_bound(
            depth: np.array,
            events: np.array,
            error: np.array,
            like_gap: float) -> float:
        l0 = get_bare_log_like(depth, events, error, af=0, fp_err=fp_err)
        af1 = 1e-10
        while (
            get_bare_log_like(depth, events, error, af=af1, fp_err=fp_err) + like_gap
            > l0
        ):
            af1 = min(af1 * growth_factor, af1 + (1 - af1) / 2)
        a = [af1 / growth_factor, af1]
        for _ in range(steps):
            test_val = sum(a) / 2
            ltemp = get_bare_log_like(depth, events, error, af=test_val, fp_err=fp_err)
            if ltemp + like_gap >= l0:
                a[0] = test_val
            if ltemp + like_gap <= l0:
                a[1] = test_val
        return sum(a) / 2

    best_log_like = get_bare_log_like(sum_dp, k, e, af=af0)
    zero_log_like = get_bare_log_like(sum_dp, k, e, af=0)

    reject_zero = (best_log_like - zero_log_like > log_like_chernoff_gap) & (sum(k) > 1)
    # imposes more than one observation to reject the null hypothesis

    if not reject_zero:
        lower_bound = 0
        upper_bound = find_af_0_upper_bound(sum_dp, k, e, log_like_gap)

    else:
        temp_afs = (af_ci[0], af0)
        for _ in range(steps):
            test_af = sum(temp_afs) / 2
            test_like = get_bare_log_like(sum_dp, k, e, af=test_af)
            temp_afs = apply_bisection(
                temp_afs[0], temp_afs[1], -(test_like - best_log_like + log_like_gap)
            )

        lower_bound = sum(temp_afs) / 2
        temp_afs = (af0, af_ci[1])
        for _ in range(steps):
            test_af = sum(temp_afs) / 2
            test_like = get_bare_log_like(sum_dp, k, e, af=test_af)
            temp_afs = apply_bisection(
                temp_afs[0], temp_afs[1], (test_like - best_log_like + log_like_gap)
            )
        upper_bound = sum(temp_afs) / 2

    return lower_bound, upper_bound


def generate_gof(
    sum_dp: np.array,
    e: np.array,
    af: float = 0.1,
    actual_k: Optional[np.array] = None,
    steps: int = 25,
    repeats: int = 100,
    seed: Optional[int] = None,
) -> Tuple[float, float]:

    if seed:
        # added for unit testing.
        np.random.seed(seed)
    k_vals_v = []
    for idx in range(len(sum_dp)):
        k_vals_v.append(np.random.binomial(sum_dp[idx], (e[idx] + af), repeats))
    k_vals = np.transpose(k_vals_v)

    log_like_vals = []
    for k in k_vals:
        ancillary_term = calculate_ts(sum_dp, k)
        af_best = find_log_like_min(sum_dp, k, e, steps=steps, min_af=0)
        like_alpha = calculate_log_like(sum_dp, k, e, ancillary_term, af=af_best)
        log_like_vals.append(like_alpha)
    sorted_likelihood_values = np.array(sorted(log_like_vals))
    ancillary_term = calculate_ts(sum_dp, actual_k)
    actual_k_likelihood = calculate_log_like(sum_dp, actual_k, e, ancillary_term, af=af)
    minim_like = np.quantile(sorted_likelihood_values, [0.01])

    return actual_k_likelihood, minim_like[0]


def wrap_analysis(
    sum_dp: np.array,
    k: np.array,
    e: np.array,
    specificity: float = 0.99,
    confidence: float = 0.95,
    min_af_grade: float = 1e-8,
) -> Tuple[float, float, float, bool]:
    af0 = find_log_like_min(sum_dp, k, e, steps=35, min_af=0)
    if af0 < min_af_grade:
        af0 = 0
    lower_bound, upper_bound = find_the_cis(
        sum_dp, k, e, af0, specificity=specificity, confidence=confidence, steps=60
    )
    true_like, critical_like = generate_gof(sum_dp, e, af=af0, actual_k=k, repeats=400)
    return af0, lower_bound, upper_bound, true_like > critical_like


def produce_estimation(
    df: pd.DataFrame, specificity: float = 0.99, confidence: float = 0.95
) -> Tuple[float, float, float, bool]:
    data_df = df.loc[df["var_type"] != "nonC-T"]
    total_depth = np.array(data_df["N"].tolist())
    k = np.array(data_df["k"].tolist())
    e = np.array(data_df["e"].tolist())
    result_edited = list()
    point_estimate, lower_bound, upper_bound, gof = wrap_analysis(
        total_depth, k, e, specificity=specificity, confidence=confidence
    )

    if point_estimate <= 0:
        result_edited.append(0)
    else:
        digits_pe = -int(np.log10(point_estimate))
        result_edited.append(round(point_estimate, digits_pe + 3))
    if lower_bound > 0:
        digits_lb = -int(np.log10(lower_bound))
        result_edited.append(round(lower_bound, digits_lb + 3))
    else:
        result_edited.append(0)
    digits_ub = -int(np.log10(upper_bound))
    result_edited.append(round(upper_bound, digits_ub + 3))
    result_edited.append(gof)

    return (
        result_edited[0],  # point estimate
        result_edited[1],  # lower bound
        result_edited[2],  # upper bound
        result_edited[3],  # gof
    )


def get_mrd_call(
    sample_id: str,
    data_df: pd.DataFrame,
    error_rate_threshold_general: int,
    error_rate_threshold_c_t: int,
    error_format: str = "per_base",
    output_file: Optional[str] = None,
    specificity: float = 0.99,
    confidence: float = 0.95,
    return_result: bool = False,
) -> Tuple[pd.DataFrame, date, str]:
    def threshold(row):
        error_threshold = (
            error_rate_threshold_c_t
            if row["var_type"] == "C-T"
            else error_rate_threshold_general
        )
        if row["error_rate"] < error_threshold:
            logging.warning(
                f"variant, {row['var_type']}, with error, {row['error_rate']}, was filtered out because error_rate below threshold, {error_threshold}"
            )
        return row["error_rate"] > error_threshold

    # filter out variants with error rate below threshold

    confidence = min(confidence, 1 - (1 - specificity) * 2)

    data_df = data_df[data_df.apply(threshold, axis=1)]
    data_df["N"] = data_df["DP_sum"]
    data_df["k"] = data_df["UC_sum"]
    if error_format == "per_base":
        data_df["e"] = 1 / data_df["error_rate"]
    else:
        data_df["e"] = data_df["error_rate"]

    pest, lbound, ubound, gof = produce_estimation(
        data_df, specificity=specificity, confidence=confidence
    )
    if lbound <= 0:
        result, test = "negative", "one-sided"
    else:
        result, test = "positive", "two-sided"

    data_dict = {
        "result": [result],
        "test": [test],
        "point_estimate": [pest],
        "lower_bound": [lbound],
        "gof_pass": [gof],
        "upper_bound": [ubound],
        "sample": [sample_id],
    }
    df = pd.DataFrame(data=data_dict)

    if return_result:
        return df, TODAY, MRD_CALLER_VERSION

    if not output_file:
        output_file = f"{sample_id}.MRD_processed.tsv"
    df.to_csv(output_file, sep="\t", index=False)
    logging.info(f"Sample, {sample_id}, processed successfully")
    return df, TODAY, MRD_CALLER_VERSION


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate MRD call based on model of independence (Poisson distribution of events)"
    )
    parser.add_argument("sample_id", help="Sample ID")
    parser.add_argument(
        "infile", help="Filename with counts for depth, error rate and observations"
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Output filename, if different from a transformation of "
        "the input filename",
    )
    parser.add_argument(
        "--error_format",
        default="per_base",
        help="Format for expressing error rate, per_base is the average number of instances per error",
    )
    parser.add_argument(
        "--specificity", default=0.99, help="Specificity (1-type1 error rate)"
    )
    parser.add_argument(
        "--confidence", default=0.95, help="Confidence value for CI and upper bound"
    )
    parser.add_argument(
        "--error_rate_threshold_general",
        type=int,
        default=ERROR_RATE_THRESHOLD_GENERAL,
        help="General threshold for error rate applied to all except C-T",
    )
    parser.add_argument(
        "--error_rate_threshold_c_t",
        type=int,
        default=ERROR_RATE_THRESHOLD_C_T,
        help="Specific threshold for error rate for C-T",
    )
    parser.add_argument(
        "--return_result", action="store_true", help="Return result without saving it"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.infile, sep="\t")

    get_mrd_call(
        args.sample_id,
        df,
        args.error_rate_threshold_general,
        args.error_rate_threshold_c_t,
        error_format=args.error_format,
        output_file=args.output_file,
        specificity=args.specificity,
        confidence=args.confidence,
        return_result=args.return_result,
    )
