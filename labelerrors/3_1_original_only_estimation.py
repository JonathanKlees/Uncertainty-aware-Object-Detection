import math

def compute_stats(k, n, confidence=0.95):
    """
    k: Number of spurious annotations
    n: total number of original only (e.g., 200)
    """

    p_hat = k / n

    z = 1.96 

    se = math.sqrt(p_hat * (1 - p_hat) / n)

    ci_low = p_hat - z * se
    ci_high = p_hat + z * se

    p_pct = p_hat * 100
    ci_low_pct = ci_low * 100
    ci_high_pct = ci_high * 100

    return {
        "p": p_hat,
        "p_pct": p_pct,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_low_pct": ci_low_pct,
        "ci_high_pct": ci_high_pct
    }


def latex_string(category_name, k, n):
    stats = compute_stats(k, n)

    return (
        f"{category_name}: "
        f"{stats['p_pct']:.1f}\\% "
        f"(95\\% CI: {stats['ci_low_pct']:.1f}--{stats['ci_high_pct']:.1f}\\%)"
    )

def full_report(spurious_k, missing_k, n=200):
    spurious = compute_stats(spurious_k, n)
    missing = compute_stats(missing_k, n)

    print("Spurious:")
    print(latex_string("Spurious annotations", spurious_k, n))

    print("\nMissing in validated GT:")
    print(latex_string("Missing annotations", missing_k, n))

if __name__ == '__main__':
    k = 10   # spurious
    n = 200

    print(latex_string("Spurious annotations", k, n))