import logging

import pandas as pd
import pytest

from solution import add_virtual_column


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "quantity": [2, 5],
            "unit_price": [10.0, 20.0],
            "discount": [1.0, 5.0],
        }
    )


def test_multi_operator(sample_df: pd.DataFrame) -> None:
    """
    Test when a role that contains multiple operators.
    Precondition:
        A sample DataFrame is provided.
    Expected:
        The new column is added correctly based on the role expression.
        The original DataFrame remains unchanged.
    """
    original_copy = sample_df.copy()
    result = add_virtual_column(
        sample_df, "quantity * unit_price - discount", "revenue"
    )

    expected = sample_df.copy()
    expected["revenue"] = [19.0, 95.0]

    assert result.equals(expected)
    assert sample_df.equals(original_copy)


@pytest.mark.parametrize(
    ("bad_df", "role", "new_column"),
    [
        ({"quantity": [1, 2]}, "quantity + quantity", "total"),
        (None, "quantity", "total"),
        ("valid_df", 123, "total"),
        ("valid_df", "quantity", 456),
    ],
)
def test_invalid_input_types(sample_df: pd.DataFrame, bad_df, role, new_column) -> None:
    """
    Test when invalid input types are provided.
    Precondition:
        Various invalid input data examples are provided.
    Expected:
        The function should return an empty DataFrame for invalid inputs.
    """
    df = sample_df if bad_df == "valid_df" else bad_df
    result = add_virtual_column(df, role, new_column)
    assert result.empty


@pytest.mark.parametrize(
    "columns",
    [
        ["quantity", "bad_col_2"],
        ["quantity", "bad col"],
    ],
)
def test_invalid_existing_dataframe_columns(columns: list[str]) -> None:
    """
    Test when the existing DataFrame has invalid column names.
    Precondition:
        A DataFrame with invalid column names is created.
    Expected:
        The function should return an empty DataFrame when the existing DataFrame has invalid column names.
    """
    df = pd.DataFrame([[1, 2]], columns=columns)
    result = add_virtual_column(df, "quantity + quantity", "total")
    assert result.empty


@pytest.mark.parametrize(
    "invalid_role",
    [
        "",
        "   ",
        "+ quantity",
        "quantity -",
        "quantity + * unit_price",
        "quantity unit_price",
        "quantity / unit_price",
    ],
)
def test_invalid_role_syntax_and_operators(
    sample_df: pd.DataFrame, invalid_role: str
) -> None:
    """
    Test when the role expression has invalid syntax or unsupported operators.
    Precondition:
        Various invalid role expressions are provided.
    Expected:
        The function should return an empty DataFrame for invalid role expressions.
    """
    result = add_virtual_column(sample_df, invalid_role, "revenue")
    assert result.empty


def test_evaluation_error_incompatible_column_types(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Test when the role expression causes an evaluation error due to incompatible column types.
    Precondition:
        A DataFrame with incompatible column types is created.
    Expected:
        The function should return an empty DataFrame when the role expression causes an evaluation error.
        The logger should log an error message.
    """
    df = pd.DataFrame({"col_a": ["foo"], "col_b": ["bar"]})
    with caplog.at_level(logging.ERROR):
        result = add_virtual_column(df, "col_a - col_b", "col_c")

    assert result.empty
    assert "Error evaluating role expression" in caplog.text


def test_unexpected_evaluation_error(
    sample_df: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Test when an unexpected error occurs during the evaluation of the role expression.
    Precondition:
        A sample DataFrame is provided.
        The pandas eval method is monkeypatched to raise a RuntimeError.
    Expected:
        The function should return an empty DataFrame when an unexpected error occurs during evaluation.
        The logger should log an error message.
    """

    def _broken_eval(self: pd.DataFrame, expr: str) -> pd.Series:
        raise RuntimeError("Simulated engine failure")

    monkeypatch.setattr(pd.DataFrame, "eval", _broken_eval)

    with caplog.at_level(logging.ERROR):
        result = add_virtual_column(sample_df, "quantity * unit_price", "revenue")

    assert result.empty
    assert "Unexpected error evaluating role expression" in caplog.text
