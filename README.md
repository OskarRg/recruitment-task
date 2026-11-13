# recruitment-task

## Installation & Setup

Running this project is possible using either `uv` or `pip`.
Built with Python 3.14.

### Using `uv`
```bash
# Install dependencies
uv sync
# Run tests
uv run pytest
# Run pipeline
uv run python pipeline.py
```

### Using `pip` & `venv`
```bash
# Create virtual environment
python -m venv .venv
# Activate virtual environment:
source .venv/bin/activate # Linux
.venv\Scripts\activate # Windows
# Install dependencies and run
pip install -r requirements.txt
pytest
python pipeline.py
```

## Assumptions & Design Decisions

Throughout completing the tasks there were a few architectural decisions that had to be made. These decisions are listed below.

### Setup
- Kept a single `requirements.txt` without a separate dev file for simplicity, while keeping dev dependencies separated in the `uv` setup.

### Part 1

- The task specified addition, subtraction, and multiplication. Since division (`/`) was not mentioned, it was treated as unsupported and rejected during validation. This could easily be changed with a small modification to `_OPERATOR_SPLIT_PATTERN` and `_VALID_ROLE_CHARS_PATTERN`.

### Part 2 

#### EDA

EDA of `sales.csv` identified the following issues and decisions that had to be taken:

- Headers contained spaces and trailing whitespace (e.g., `'Quantity '`, `'Unit Price'`). They were normalized to `snake_case` to meet validation rules.
- One repeated header row and one summary `TOTAL` footer row were removed.
- Five duplicated rows were dropped, because duplicates had no clear use in the next steps.
- Unnecessary spaces and inconsistent casing in 15 rows (e.g., `' South'`, `'EAST'`) had to be normalized in `Region`.
- Four values used spaces as thousands separators in `Quantity`. The remaining rows had mostly lower values, but as spaces were stripped, higher quantity values were also kept.
- Six rows in `Quantity` contained negative values. These were treated as human-made typos and converted to positive values rather than dropped or zeroed. This decision had possible consequences, but human error seemed likely given the data structure.
- Decimal commas in seven rows (e.g. `'39,90'`) had to be replaced with dots in `Unit Price` to be treated as floats.
- Missing values in `Unit Price` were filled from other rows having the same product instead of dropping the four rows.
- Empty cells in the `Discount` column indicated no discount and were filled with `0.0`. This affected 963 rows before deduplication, or 959 deduplicated rows.

#### Data Pipeline

- Given the scope of this task, the implementation was kept in a single `pipeline.py` module and a single `test_pipeline.py` file. For a larger project, these files would preferably be split into separate modules with corresponding test suites.
- Following the pattern established in Part 1 and shown in the instruction, transformation helpers were written to operate on copies (`df.copy()`) and not modify input DataFrames in place.
- `order_id` values were validated against a general format (`^[A-Za-z]{2}-\d{1,5}$`), assuming a two-letter prefix to support larger ID capacity than only validating against `'SO'` and up to 5 digits.
- Missing or placeholder `region` values were replaced with `"Unknown"` instead of dropping rows. These operations were done after normalization of the `region` column.
- Negative values encountered across numeric columns (`quantity`, `unit_price`, `discount`) were treated as sign-entry errors and converted to positive values.
- Imputing missing `unit_price` values assumed each product had a single constant price across the dataset. If price changes over time or promotions were expected, the lookup strategy would need to be re-evaluated.
- The pipeline was implemented to raise `ValueError` on empty `product` values, unimputable `unit_price` values for unknown products, or unexpected unparseable values in `quantity`, `discount`, and `order_date` instead of silently dropping rows, as those could impact the analysis.
- Exact duplicate removal was configured to evaluate all columns rather than only `order_id`, allowing distinct order lines to share an `order_id` while removing identical export duplicates (useful if an orderer included multiple products). It was placed after normalization so rows differing only in formatting were deduplicated as well. At this dataset size, processing duplicates before dropping them was worth the trade-off compared to early raw-string deduplication, though it could become more time-consuming on a much larger dataset.
- Aggregated `revenue` values were rounded to 2 decimal places, matching the currency precision observed in `Unit Price` and `Discount` in order to avoid floating-point representation artifacts.
- Output in `revenue_by_region.csv` was sorted by `revenue` in descending order so the impact was easier to see.