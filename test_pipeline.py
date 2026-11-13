from pathlib import Path

import pandas as pd
import pytest

from pipeline import (
    _clean_numeric_columns,
    _normalize_column_names,
    _normalize_text_columns,
    _parse_numeric_series,
    _remove_duplicates,
    _validate_dates,
    add_revenue_column,
    aggregate_revenue_by_region,
    load_sales_data,
    run_pipeline,
)


@pytest.fixture
def dirty_sales_df() -> pd.DataFrame:
    """
    Provide sales DataFrame containing representative data quality issues.
    """
    return pd.DataFrame(
        {
            "Order ID": [
                "SO-1001",
                "Order ID",
                "AB-2",
                "AB-2",
                "SO-1003",
                "SO-1004",
                "TOTAL",
            ],
            "Region ": [
                " europe ",
                "Region",
                "NORTH",
                "north",
                "N/A",
                "South",
                "",
            ],
            "Product": [
                " Laptop ",
                "Product",
                "Mouse",
                "Mouse",
                "Laptop",
                "Keyboard",
                "",
            ],
            "Quantity": [
                "1 000",
                "Quantity",
                "2",
                "2",
                "3",
                "-1",
                "1004",
            ],
            "Unit Price": [
                "1200,50",
                "Unit Price",
                "25.00",
                "25,00",
                "",
                "49.99",
                "",
            ],
            "Discount": [
                "",
                "Discount",
                "0.10",
                "0.10",
                "0.05",
                "-0.02",
                "",
            ],
            "Order Date": [
                "2025-01-15",
                "Order Date",
                "2025-01-16",
                " 2025-01-16 ",
                "2025-01-17",
                " 2025-01-18 ",
                "",
            ],
        }
    )


def test_load_sales_data_valid_csv(tmp_path: Path) -> None:
    """
    Test loading a CSV file with all fields preserved as raw strings.
    Precondition:
        A temporary CSV file exists with formatted numeric and empty cells.
    Expected:
        Returns a DataFrame with string/object columns and empty cells preserved as ''.
    """
    csv_file = tmp_path / "sales_sample.csv"
    csv_file.write_text(
        "Order ID,Region,Product,Quantity,Unit Price,Discount,Order Date\n"
        'SO-001,Europe,Laptop,"1 000","12,50",,2025-01-01\n',
        encoding="utf-8",
    )

    df = load_sales_data(csv_file)

    assert len(df) == 1
    assert df.loc[0, "Quantity"] == "1 000"
    assert df.loc[0, "Unit Price"] == "12,50"
    assert df.loc[0, "Discount"] == ""


def test_load_sales_data_missing_file(tmp_path: Path) -> None:
    """
    Test loading from a nonexistent file path.
    Precondition:
        The provided file path does not exist on disk.
    Expected:
        Raises FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError, match="Input sales file not found"):
        load_sales_data(tmp_path / "missing.csv")


def test_normalize_column_names_preserves_extra_columns() -> None:
    """
    Test normalizing column headers while keeping additional columns intact.
    Precondition:
        A DataFrame has all required columns with extra spaces/uppercase plus an extra column.
    Expected:
        Column names are stripped, lowercased, and snake_cased without dropping extra columns.
    """
    df = pd.DataFrame(
        {
            " Order  ID ": ["SO-1"],
            "Region": ["Europe"],
            "Product": ["Laptop"],
            "Quantity ": ["1"],
            "Unit Price": ["10.0"],
            "Discount": ["0"],
            "ORDER DATE": ["2025-01-01"],
            "Extra Note": ["vip"],
        }
    )

    result = _normalize_column_names(df)

    assert list(result.columns) == [
        "order_id",
        "region",
        "product",
        "quantity",
        "unit_price",
        "discount",
        "order_date",
        "extra_note",
    ]


def test_normalize_column_names_missing_required() -> None:
    """
    Test validation when one or more required columns are missing.
    Precondition:
        A DataFrame lacks the 'order_date' column.
    Expected:
        Raises ValueError naming the missing required column.
    """
    df = pd.DataFrame(
        {
            "Order ID": ["SO-1"],
            "Region": ["Europe"],
            "Product": ["Laptop"],
            "Quantity": ["1"],
            "Unit Price": ["10.0"],
            "Discount": ["0"],
        }
    )

    with pytest.raises(
        ValueError, match="Missing required columns: \\['order_date'\\]"
    ):
        _normalize_column_names(df)


def test_normalize_text_columns_and_unknown_region() -> None:
    """
    Test whitespace/casing cleanup and mapping placeholder regions to 'Unknown'.
    Precondition:
        A DataFrame has untrimmed text and placeholder region values.
    Expected:
        Text columns are normalized and placeholder region values become 'Unknown'.
    """
    df = pd.DataFrame(
        {
            "order_id": [" SO-1 ", "AB-2", "CD-3", "EF-4"],
            "region": [" europe ", "N/A", "", "null"],
            "product": [" Laptop ", "Mouse", "Cable", " Printer   paper "],
        }
    )

    result = _normalize_text_columns(df)

    assert result["order_id"].tolist() == ["SO-1", "AB-2", "CD-3", "EF-4"]
    assert result["region"].tolist() == ["Europe", "Unknown", "Unknown", "Unknown"]
    assert result["product"].tolist() == ["Laptop", "Mouse", "Cable", "Printer paper"]


@pytest.mark.parametrize("bad_product", ["   ", "N/A", "null", "-"])
def test_normalize_text_columns_empty_product(bad_product: str) -> None:
    """
    Test validation when any product value is empty or a placeholder after stripping.
    Precondition:
        A DataFrame contains a whitespace-only or placeholder product entry.
    Expected:
        Raises ValueError indicating empty product values.
    """
    df = pd.DataFrame(
        {
            "order_id": ["SO-1"],
            "region": ["Europe"],
            "product": [bad_product],
        }
    )

    with pytest.raises(ValueError, match="Empty 'product' values encountered"):
        _normalize_text_columns(df)


def test_remove_duplicates_preserves_multi_item_orders() -> None:
    """
    Test full-row deduplication while keeping rows that share order_id for distinct items.
    Precondition:
        A DataFrame contains one exact duplicate row and two distinct items sharing order_id.
    Expected:
        Only the exact duplicate row is dropped and the index is reset.
    """
    df = pd.DataFrame(
        {
            "order_id": ["SO-1", "SO-1", "SO-1"],
            "product": ["Laptop", "Laptop", "Mouse"],
        }
    )

    result = _remove_duplicates(df)

    assert len(result) == 2
    assert result["product"].tolist() == ["Laptop", "Mouse"]
    assert list(result.index) == [0, 1]


def test_parse_numeric_series_formatting_and_negatives() -> None:
    """
    Test parsing spaces, commas, empty replacement, and negative absolute conversion.
    Precondition:
        A string Series has thousand spaces, decimal commas, an empty value, and a negative.
    Expected:
        Numeric Series is returned with cleaned positive floats.
    """
    series = pd.Series(["1 250,50", "", " -10.5 "])

    result = _parse_numeric_series(series, "discount", empty_replacement="0")

    assert result.tolist() == pytest.approx([1250.5, 0.0, 10.5])


@pytest.mark.parametrize(
    ("overrides", "match_pattern"),
    [
        ({"quantity": ["abc"]}, "Unparseable values encountered in 'quantity'"),
        ({"discount": ["bad"]}, "Unparseable values encountered in 'discount'"),
        (
            {"product": ["UnknownGadget"], "unit_price": [""]},
            "Missing 'unit_price' values could not be imputed",
        ),
    ],
)
def test_clean_numeric_columns_errors(
    overrides: dict[str, list[str]],
    match_pattern: str,
) -> None:
    """
    Test fail-fast errors when numeric values are unparseable or unimputable.
    Precondition:
        A DataFrame overrides a base valid row with invalid text or unimputable unit_price.
    Expected:
        Raises ValueError with diagnostic context for the offending column.
    """
    base = {
        "product": ["Laptop"],
        "quantity": ["1"],
        "unit_price": ["10.0"],
        "discount": ["0"],
    }
    df = pd.DataFrame(base | overrides)

    with pytest.raises(ValueError, match=match_pattern):
        _clean_numeric_columns(df)


def test_clean_numeric_columns_imputation_and_dtypes() -> None:
    """
    Test numeric dtype conversion and deterministic unit_price imputation by product.
    Precondition:
        A DataFrame has space/negative quantity, missing unit_price for an existing product, and empty discount.
    Expected:
        Quantities become int64, missing unit_price is imputed as float64, empty discount becomes 0.0.
    """
    df = pd.DataFrame(
        {
            "product": ["Laptop", "Laptop"],
            "quantity": ["1 000", "-2"],
            "unit_price": ["1200,50", ""],
            "discount": ["", "-0.1"],
        }
    )

    result = _clean_numeric_columns(df)

    assert str(result["quantity"].dtype) == "int64"
    assert str(result["unit_price"].dtype) == "float64"
    assert str(result["discount"].dtype) == "float64"
    assert result["quantity"].tolist() == [1000, 2]
    assert result["unit_price"].tolist() == pytest.approx([1200.5, 1200.5])
    assert result["discount"].tolist() == pytest.approx([0.0, 0.1])


def test_validate_invalid_dates() -> None:
    """
    Test rejecting malformed date strings.
    Precondition:
        A DataFrame contains an order_date value not matching YYYY-MM-DD format.
    Expected:
        Invalid dates raise ValueError.
    """
    invalid_df = pd.DataFrame({"order_date": ["15-01-2025"]})
    with pytest.raises(ValueError, match="Invalid date format encountered"):
        _validate_dates(invalid_df)


def test_add_revenue_column_valid_and_invalid() -> None:
    """
    Test computing the revenue column and failing on invalid schema.
    Precondition:
        One DataFrame has valid numeric columns and another lacks a required operand column.
    Expected:
        Computes `revenue` or raises ValueError on failure.
    """
    valid_df = pd.DataFrame({"quantity": [2], "unit_price": [10.5], "discount": [1.0]})
    result = add_revenue_column(valid_df)

    assert result["revenue"].tolist() == pytest.approx([20.0])

    invalid_df = pd.DataFrame({"quantity": [2], "unit_price": [10.5]})
    with pytest.raises(ValueError, match="Failed to compute 'revenue' column"):
        add_revenue_column(invalid_df)


def test_aggregate_revenue_by_region_sorting_and_rounding() -> None:
    """
    Test regional revenue summation, currency rounding, and deterministic descending sort.
    Precondition:
        A DataFrame contains multiple rows per region and equal regional totals.
    Expected:
        Sums are rounded to 2 decimal places, sorted by revenue descending then region ascending.
    """
    df = pd.DataFrame(
        {
            "region": ["North", "North", "West", "East"],
            "revenue": [10.004, 10.004, 50.123, 50.124],
        }
    )
    result = aggregate_revenue_by_region(df)

    assert result["region"].tolist() == ["East", "West", "North"]
    assert result["revenue"].tolist() == pytest.approx([50.12, 50.12, 20.01])


def test_run_pipeline_end_to_end(
    dirty_sales_df: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """
    Test full ETL execution from raw CSV input to sorted regional revenue CSV output.
    Precondition:
        A temporary CSV contains representative dirty rows covering all cleaning rules.
    Expected:
        Writes `revenue_by_region.csv` with rounded regional totals sorted descending.
    """
    input_csv = tmp_path / "sales.csv"
    output_csv = tmp_path / "nested" / "revenue_by_region.csv"
    dirty_sales_df.to_csv(input_csv, index=False)

    result = run_pipeline(input_path=input_csv, output_path=output_csv)

    assert output_csv.is_file()
    written_df = pd.read_csv(output_csv)
    pd.testing.assert_frame_equal(result, written_df)
    assert result["region"].tolist() == ["Europe", "Unknown", "South", "North"]
    assert result["revenue"].tolist() == pytest.approx(
        [1200500.0, 3601.45, 49.97, 49.9]
    )
