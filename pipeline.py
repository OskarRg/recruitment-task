import logging
import re
from pathlib import Path

import pandas as pd

from solution import add_virtual_column

logger = logging.getLogger(__name__)

_BASE_DIR: Path = Path(__file__).resolve().parent
_DEFAULT_INPUT_PATH: Path = _BASE_DIR / "sales.csv"
_DEFAULT_OUTPUT_PATH: Path = _BASE_DIR / "revenue_by_region.csv"
_REVENUE_EXPRESSION: str = "quantity * unit_price - discount"
_ORDER_ID_PATTERN: re.Pattern[str] = re.compile(r"^[A-Za-z]{2}-\d{1,5}$")
_CURRENCY_PRECISION: int = 2
_PLACEHOLDER_VALUES: frozenset[str] = frozenset(
    {"", "na", "nan", "none", "n/a", "null", "n.a.", "-", "?"}
)
_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "order_id",
        "region",
        "product",
        "quantity",
        "unit_price",
        "discount",
        "order_date",
    }
)


def load_sales_data(input_path: Path) -> pd.DataFrame:
    """
    Load raw sales data from a CSV file as strings.

    :param input_path: Path to the input CSV file.
    :return: Raw DataFrame with all columns read as strings.
    :raises FileNotFoundError: If the input CSV file does not exist.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input sales file not found: {input_path}")

    df: pd.DataFrame = pd.read_csv(input_path, dtype=str, keep_default_na=False)
    logger.info("Loaded %d rows from %s.", len(df), input_path.name)
    return df


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame column labels to snake_case and verify required schema.

    :param df: Input DataFrame with raw column names.
    :return: DataFrame with normalized column names.
    :raises ValueError: If duplicate column names occur or any required column is missing.
    """
    normalized_df = df.copy()
    normalized_df.columns = [
        "_".join(str(col).lower().split()) for col in normalized_df.columns
    ]

    if normalized_df.columns.duplicated().any():
        duplicates = (
            normalized_df.columns[normalized_df.columns.duplicated()].unique().tolist()
        )
        raise ValueError(f"Duplicate columns after normalization: {duplicates}")

    missing_columns = _REQUIRED_COLUMNS - set(normalized_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    return normalized_df


def _filter_valid_order_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows whose `order_id` matches the expected format.

    :param df: Input DataFrame with normalized column names.
    :return: DataFrame containing only rows with valid `order_id` values and a reset index.
    """
    cleaned_order_ids: pd.Series = df["order_id"].str.strip()
    is_valid_order: pd.Series = cleaned_order_ids.str.match(_ORDER_ID_PATTERN, na=False)
    removed_count: int = int((~is_valid_order).sum())

    if removed_count > 0:
        invalid_samples = cleaned_order_ids.loc[~is_valid_order].unique().tolist()[:5]
        logger.info(
            "Removed %d rows with invalid order_id format (samples: %s).",
            removed_count,
            invalid_samples,
        )

    return df[is_valid_order].reset_index(drop=True)


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows across all columns.

    :param df: Input DataFrame.
    :return: Deduplicated DataFrame with a reset index.
    """
    deduplicated_df: pd.DataFrame = df.drop_duplicates().reset_index(drop=True)
    removed_count: int = len(df) - len(deduplicated_df)

    if removed_count > 0:
        logger.info("Removed %d exact duplicate rows.", removed_count)

    return deduplicated_df


def _normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip surrounding whitespace, standardize casing, and fill missing regions with 'Unknown'.

    :param df: Input DataFrame.
    :return: DataFrame with normalized `order_id`, `region`, and `product` columns.
    :raises ValueError: If any `product` value is empty after stripping.
    """
    text_df = df.copy()
    text_df["order_id"] = text_df["order_id"].str.strip()
    text_df["region"] = text_df["region"].str.split().str.join(" ").str.title()
    text_df["product"] = text_df["product"].str.split().str.join(" ")

    invalid_region_mask = text_df["region"].str.lower().isin(_PLACEHOLDER_VALUES)
    unknown_count: int = int(invalid_region_mask.sum())
    if unknown_count > 0:
        text_df["region"] = text_df["region"].mask(invalid_region_mask, "Unknown")
        logger.warning(
            "Replaced %d missing/invalid 'region' values with 'Unknown'.",
            unknown_count,
        )

    invalid_product_mask = text_df["product"].str.lower().isin(_PLACEHOLDER_VALUES)
    if invalid_product_mask.any():
        raise ValueError("Empty 'product' values encountered.")

    return text_df


def _parse_numeric_series(
    series: pd.Series,
    column_name: str,
    *,
    empty_replacement: str | None = None,
    allow_unparseable: bool = False,
) -> pd.Series:
    """
    Clean formatting, parse numeric values, and convert negative values to positive.

    :param series: Raw string Series for a numeric column.
    :param column_name: Name of the column for logging and error messages.
    :param empty_replacement: Optional string value to substitute for empty cells before parsing.
    :param allow_unparseable: If True, keep unparseable values as NaN.
    :return: Parsed numeric Series with non-negative values.
    :raises ValueError: If unparseable values exist and `allow_unparseable` is False.
    """
    cleaned_str: pd.Series = (
        series.str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    if empty_replacement is not None:
        cleaned_str = cleaned_str.mask(cleaned_str == "", empty_replacement)

    parsed = pd.Series(
        pd.to_numeric(cleaned_str, errors="coerce"),
        index=cleaned_str.index,
    )
    if not allow_unparseable and parsed.isna().any():
        bad_values = series.loc[parsed.isna()].unique().tolist()[:5]
        raise ValueError(
            f"Unparseable values encountered in '{column_name}': {bad_values}"
        )

    negative_count: int = int((parsed < 0).sum())
    if negative_count > 0:
        logger.info(
            "Converted %d negative '%s' values to positive.",
            negative_count,
            column_name,
        )
        parsed = parsed.abs()

    return parsed


def _clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean, impute, and convert `quantity`, `unit_price`, and `discount` to numeric types.

    :param df: Input DataFrame with normalized text columns.
    :return: DataFrame with validated numeric columns.
    :raises ValueError: If numeric parsing fails or missing `unit_price` cannot be imputed.
    """
    numeric_df = df.copy()

    numeric_df["quantity"] = _parse_numeric_series(
        numeric_df["quantity"], "quantity"
    ).astype("int64")

    price_num = _parse_numeric_series(
        numeric_df["unit_price"], "unit_price", allow_unparseable=True
    )
    missing_price_count: int = int(price_num.isna().sum())
    if missing_price_count > 0:
        product_prices: pd.Series = price_num.groupby(numeric_df["product"]).transform(
            "first"
        )
        price_num = price_num.fillna(product_prices)
        logger.info(
            "Imputed %d missing 'unit_price' values using product mapping.",
            missing_price_count,
        )
    if price_num.isna().any():
        raise ValueError(
            "Missing 'unit_price' values could not be imputed for unknown products."
        )
    numeric_df["unit_price"] = price_num.astype("float64")

    numeric_df["discount"] = _parse_numeric_series(
        numeric_df["discount"], "discount", empty_replacement="0"
    ).astype("float64")

    return numeric_df


def _validate_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse and validate `order_date` values against YYYY-MM-DD format.

    :param df: Input DataFrame.
    :return: DataFrame with `order_date` converted to datetime.
    :raises ValueError: If any `order_date` cannot be parsed as YYYY-MM-DD.
    """
    date_df: pd.DataFrame = df.copy()
    parsed_dates = pd.Series(
        pd.to_datetime(
            date_df["order_date"].str.strip(),
            format="%Y-%m-%d",
            errors="coerce",
        ),
        index=date_df.index,
    )
    if parsed_dates.isna().any():
        bad_dates = date_df.loc[parsed_dates.isna(), "order_date"].unique().tolist()[:5]
        raise ValueError(
            f"Invalid date format encountered in 'order_date': {bad_dates}"
        )
    date_df["order_date"] = parsed_dates
    return date_df


def clean_sales_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute the full data cleaning pipeline on the raw sales DataFrame.

    :param raw_df: Raw DataFrame loaded from `sales.csv`.
    :return: Cleaned DataFrame ready for revenue calculations.
    """
    cleaned_df = (
        raw_df.pipe(_normalize_column_names)
        .pipe(_filter_valid_order_rows)
        .pipe(_normalize_text_columns)
        .pipe(_clean_numeric_columns)
        .pipe(_validate_dates)
        .pipe(_remove_duplicates)
    )
    logger.info("Cleaned dataset contains %d valid order rows.", len(cleaned_df))
    return cleaned_df


def add_revenue_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute `revenue` column with `add_virtual_column`.

    :param df: Cleaned sales DataFrame containing `quantity`, `unit_price`, and `discount`.
    :return: DataFrame with an added `revenue` column.
    :raises ValueError: If `add_virtual_column` rejects the DataFrame or expression.
    """
    revenue_df = add_virtual_column(df, _REVENUE_EXPRESSION, "revenue")
    if "revenue" not in revenue_df.columns:
        raise ValueError(
            f"Failed to compute 'revenue' column using expression '{_REVENUE_EXPRESSION}'."
        )
    return revenue_df


def aggregate_revenue_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate total revenue per region, rounded to currency precision and sorted descending.

    :param df: Sales DataFrame containing `region` and `revenue` columns.
    :return: DataFrame with columns `region` and `revenue`, sorted by `revenue`.
    """
    aggregated_df: pd.DataFrame = df.groupby("region", as_index=False).agg(
        revenue=("revenue", "sum")
    )
    aggregated_df["revenue"] = aggregated_df["revenue"].round(_CURRENCY_PRECISION)
    return aggregated_df.sort_values(
        by=["revenue", "region"],
        ascending=[False, True],
    ).reset_index(drop=True)


def save_revenue_by_region(df: pd.DataFrame, output_path: Path) -> None:
    """
    Write the aggregated revenue DataFrame to a CSV file without row indices.

    :param df: Aggregated DataFrame with `region` and `revenue` columns.
    :param output_path: Destination CSV file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d region rows to %s.", len(df), output_path)


def run_pipeline(
    input_path: Path = _DEFAULT_INPUT_PATH,
    output_path: Path = _DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Run the end-to-end sales ETL pipeline from CSV extraction to regional revenue output.

    :param input_path: Path to the raw input CSV file.
    :param output_path: Path where aggregated revenue CSV will be written.
    :return: Aggregated DataFrame containing total revenue by region.
    """
    raw_df: pd.DataFrame = load_sales_data(input_path)
    revenue_by_region_df: pd.DataFrame = (
        clean_sales_data(raw_df)
        .pipe(add_revenue_column)
        .pipe(aggregate_revenue_by_region)
    )
    save_revenue_by_region(revenue_by_region_df, output_path)
    return revenue_by_region_df


def main() -> None:
    """
    Configure logging and execute the default sales pipeline.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run_pipeline()


if __name__ == "__main__":
    main()
