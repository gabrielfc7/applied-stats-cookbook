"""Command-line interface for applied-stats-cookbook."""

from __future__ import annotations

import argparse
import sys
import numpy as np

from applied_stats.parametric import one_sample_t_test, two_sample_t_test
from applied_stats.nonparametric import mann_whitney_u_test, kruskal_wallis_test
from applied_stats.anova import one_way_anova
from applied_stats.reporting.reporter import PublicationReporter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="applied-stats-cookbook: Automated hypothesis testing and publication reporting CLI."
    )
    subparsers = parser.add_subparsers(dest="command", help="Statistical test command")

    # t-test
    ttest_parser = subparsers.add_parser("ttest", help="Run independent 2-sample t-test")
    ttest_parser.add_argument("--g1", type=float, nargs="+", required=True, help="Samples for group 1")
    ttest_parser.add_argument("--g2", type=float, nargs="+", required=True, help="Samples for group 2")
    ttest_parser.add_argument("--equal-var", action="store_true", help="Assume equal variance (Student's t-test)")

    # mann-whitney
    mwu_parser = subparsers.add_parser("mannwhitney", help="Run Mann-Whitney U test")
    mwu_parser.add_argument("--g1", type=float, nargs="+", required=True, help="Samples for group 1")
    mwu_parser.add_argument("--g2", type=float, nargs="+", required=True, help="Samples for group 2")

    # anova
    anova_parser = subparsers.add_parser("anova", help="Run one-way ANOVA across groups")
    anova_parser.add_argument("--groups", type=str, nargs="+", required=True, help="Comma-separated values per group (e.g. 1,2,3 4,5,6)")

    args = parser.parse_args()

    if args.command == "ttest":
        res = two_sample_t_test(args.g1, args.g2, equal_var=args.equal_var)
        reporter = PublicationReporter(res)
        print("\n" + reporter.to_markdown() + "\n")
        print("APA 7th format:", res.apa_format())

    elif args.command == "mannwhitney":
        res = mann_whitney_u_test(args.g1, args.g2)
        reporter = PublicationReporter(res)
        print("\n" + reporter.to_markdown() + "\n")
        print("APA 7th format:", res.apa_format())

    elif args.command == "anova":
        parsed_groups = [[float(v) for v in g.split(",")] for g in args.groups]
        table = one_way_anova(*parsed_groups)
        print("\nOne-Way ANOVA Table:")
        print(table.to_dataframe().to_string())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
