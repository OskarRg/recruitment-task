import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

_VALID_COLUMN_NAME_PATTERN = re.compile(r"^[a-zA-Z_]+$")
_VALID_ROLE_CHARS_PATTERN = re.compile(r"^[a-zA-Z_\s+\-*]+$")
_OPERATOR_SPLIT_PATTERN = re.compile(r"[+\-*]")


def _is_valid_column_name(column_name: str) -> bool:
    """
    Check if column name consists only of letters and underscores.

    :param column_name: The column name to validate.
    :return: True if valid, else False.
    """
    return isinstance(column_name, str) and bool(
        _VALID_COLUMN_NAME_PATTERN.fullmatch(column_name)
    )


def _validate_inputs(df: pd.DataFrame, role: str, new_column: str) -> bool:
    """
    Validates input data.
    :param df: The input DataFrame that should contain the columns referenced in the `role` parameter.
    :param role: The equation that should only consist of letters, underscores, spaces, and operators +, -, *.
    :param new_column: The name of the new virtual column.
    :return: True if inputs are valid, else False.
    """
    if not (
        isinstance(df, pd.DataFrame)
        and isinstance(role, str)
        and isinstance(new_column, str)
    ):
        return False

    if not _is_valid_column_name(new_column):
        return False

    if not all(_is_valid_column_name(col) for col in df.columns):
        return False

    if not _VALID_ROLE_CHARS_PATTERN.fullmatch(role):
        return False

    operands = [token.strip() for token in _OPERATOR_SPLIT_PATTERN.split(role)]
    if not operands or any(not operand for operand in operands):
        return False

    return all(
        _is_valid_column_name(column_name) and column_name in df.columns
        for column_name in operands
    )


def add_virtual_column(df: pd.DataFrame, role: str, new_column: str) -> pd.DataFrame:
    """
    Adds a virtual column to the DataFrame based on the expression passed in `role`.

    :param df: The input DataFrame.
    :param role: The equation based on which the virtual column will be calculated.
    :param new_column: The name of the new virtual column.
    :return: A new DataFrame with the added virtual column, or an empty DataFrame if inputs are invalid.
    """
    if not _validate_inputs(df, role, new_column):
        logger.warning(
            "Validation failed for role=%r and new_column=%r. Returning empty DataFrame.",
            role,
            new_column,
        )
        return pd.DataFrame([])

    try:
        df_copy = df.copy()
        df_copy[new_column] = df_copy.eval(role)
        return df_copy
    except (
        ValueError,
        TypeError,
        SyntaxError,
        KeyError,
        pd.errors.UndefinedVariableError,
    ) as exception:
        logger.error("Error evaluating role expression %r: %s", role, exception)
        return pd.DataFrame([])
    except Exception:
        logger.exception("Unexpected error evaluating role expression %r", role)
        return pd.DataFrame([])
