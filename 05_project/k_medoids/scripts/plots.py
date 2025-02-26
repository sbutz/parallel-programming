#!/usr/bin/env python3

import pandas as pd
import sys

def main(report_path):
    df = pd.read_csv(report_path)

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.display.float_format = '{:.2f}'.format
    print(df)

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python3 plots.py <report_path>"
    main(sys.argv[1]);