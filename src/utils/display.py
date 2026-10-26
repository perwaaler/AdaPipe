import sys
import inspect
import numpy as np
from rich.pretty import pprint
from pprint import pprint, pformat
from collections import OrderedDict
from IPython.display import display
from rich import print as rich_print
from typing import Literal

RESET_COLOR = "\033[0m"

COLOR_TO_ANSI_MAP = {  # All are in bold font
    "grey": "\033[1;30m",
    "white": "\033[1;37m",
    "green": "\033[1;92m",
    "lime": "\033[1;32m",  # Added lime (green with bold)
    "cyan": "\033[1;36m",
    "teal": "\033[1;96m",  # Added teal (cyan)
    "blue": "\033[1;94m",
    "dark_blue": "\033[1;34m",
    "orange": "\033[1;33m",
    "brown": "\033[38;5;94m",
    "red": "\033[1;91m",
    "dark_red": "\033[1;31m",
    "pink": "\033[1;95m",  # Added magenta
    "black": "\033[0;30m",
    "magenta": "\033[1;35m",  # Added pink (magenta with bold)
    "reset": "\033[0m",
}


def convert_numpy_types(obj):
    """Converts numbers to"""
    convert = lambda x: x if isinstance(x, str) else float(x)
    if isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, (np.float64, np.float32, np.int64, np.int32)):
        return obj.item()  # Convert to native Python type
    elif isinstance(obj, list):
        return [convert(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert(v) for v in obj)
    else:
        return obj


# Override the default pprint behavior
def pprint_cust(obj, *args, **kwargs):
    convert = lambda x: x if isinstance(x, str) else float(x)
    if isinstance(obj, dict):
        obj = {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, (np.float64, np.float32, np.int64, np.int32)):
        obj = obj.item()  # Convert to native Python type
    elif isinstance(obj, list):
        obj = [convert(v) for v in obj]
    elif isinstance(obj, tuple):
        obj = tuple(convert(v) for v in obj)
    else:
        obj = obj

    pprint(obj, *args, **kwargs)


# %% PRINTING UTILITES
def color_rows_on_condition(df, row_condition_map, default_color="black"):
    """Applies conditional color formatting on the row of the df. row_condition_map
    maps conditions to colors - its keys are conditions that will be interpreted
    as literals. Ex:

        row_condition_map = {
            "red": lambda row: row["Age"]>60 and row["BMI"]>30,
            "green": lambda row: row["age"]< 35 and row["BMI"]<30
        }
    """

    def style_row(row):
        for color, condition in row_condition_map.items():
            if condition(row):  # Apply the condition function
                return [f"color: {color}"] * len(row)
        return [f"color: {default_color}"] * len(row)

    # Apply the function to the DataFrame
    return df.style.apply(style_row, axis=1)


def color_cells_on_column_condition(df, column_condition_map, default_color="black"):
    """Applies conditional color formatting on the cells of the df. column_condition_map
    maps conditions to colors - its keys are conditions that will be interpreted
    as literals. Ex:

    side_conditions = {"fat": lambda x: x["BMI"] > 30}
    column_condition_map = {
        "Age": {
            "colorcodes":
                {"green": lambda c: c < 20 and not fat,
                "orange": lambda c: 20 <= c < 60,
                "red": lambda c: c >= 60,
                }
        },
        "BMI": {
            "green": lambda x: x < 20,
            "orange": lambda x: 20 <= x < 30,
            "red": lambda x: x >= 30
        }
    }
    """

    def style_cell(row):
        styles = []

        for col in df.columns:
            if col in column_condition_map:
                style = None
                for color, condition in column_condition_map[col].items():

                    argname = get_argnames(condition)[0]
                    if argname.startswith("r"):
                        # Pass whole row as input (assumed reliance on other columns)
                        in_data = row
                    elif argname.startswith("c"):
                        # Pass only value of column to color-code (no reliance on other columns)
                        in_data = row[col]
                    else:
                        red_print(f"Argname {argname} invalid: must start with r (for row) or c (for column)")

                    if condition(in_data):  # Check if condition matches
                        style = f"color: {color}"

                if style is None:
                    # If no condition matches, apply default color
                    style = f"color: {default_color}"

                styles.append(style)

            else:
                # Default color for columns without conditions
                styles.append(f"color: {default_color}")
        return styles

    # Apply the function to the DataFrame
    return df.style.apply(style_cell, axis=1)


def get_argnames(f):
    """Get the names of the input arguments"""
    return list(inspect.signature(f).parameters.keys())


def color_text(*args, c="grey", use_pprint=False):
    """Returns text in the desired colour (grey, green, or blue)."""
    colored_strings = []
    for arg in args:
        if use_pprint:
            arg = pformat(arg, indent=2)
        colored_strings.append(f"{COLOR_TO_ANSI_MAP[c]}{arg}{RESET_COLOR}")
    return colored_strings


def colored_pprint(obj, color="blue"):
    formatted_output = color_text(obj, c=color)[0]
    print(formatted_output)


def print_header1(
    title: str = "Example Title",
    vspace1=1,
    vspace2=0,
    c: Literal[
        "green", "blue", "grey", "white", "orange", "brown", "blue", "teal", "magenta", "red", "pink", "lime", "black"
    ] = "white",
):
    color_print("\n" * vspace1 + "=" * 80 + f"\n{title}\n" + "=" * 80 + "\n" * vspace2, c=c)


def print_header2(title: str = "Example Title", vspace1=1, c="white"):
    color_print("\n" * vspace1 + "-----" + f" {title} " + "---" * 100, c=c)


def color_print(
    *args,
    c: Literal[
        "green", "blue", "grey", "white", "orange", "brown", "blue", "teal", "magenta", "red", "pink", "lime", "black"
    ] = "green",
    sep="",
):
    """Prints text in the specified color."""
    try:
        if c == "default":
            print(*args, sep=sep)
        elif c in [
            "green",
            "blue",
            "grey",
            "white",
            "orange",
            "brown",
            "blue",
            "teal",
            "magenta",
            "red",
            "pink",
            "lime",
        ]:
            print(*color_text(*args, c=c), sep=sep)
        else:
            hex_print(*args, c=c, sep=sep)
    except Exception as e:
        red_print(f"[color_print warning: {e}]")


def hex_print(*args, c="#c90606", sep=" "):
    txt = sep.join(args)
    rich_print(f"[{c}]{txt}[/{c}]")


def silent_print(*args, sep=" "):
    """Prints text in console in dark grey."""
    print(*color_text(*args, c="grey"), sep=sep)


def white_print(*args, sep=" "):
    """Prints text in console in dark grey."""
    print(*color_text(*args, c="white"), sep=sep)


def orange_print(*args, sep=" "):
    """Prints each argument on a new line in orange."""
    print(*color_text(*args, c="orange"), sep=sep)


def brown_print(*args, sep=" "):
    """Prints each argument on a new line in orange."""
    print(*color_text(*args, c="brown"), sep=sep)


def green_print(*args, sep=" "):
    """Prints each argument on a new line in green."""
    print(*color_text(*args, c="green"), sep=sep)


def blue_print(*args, sep=" "):
    """Prints each argument on a new line in blue."""
    print(*color_text(*args, c="blue"), sep=sep)


def teal_print(*args, sep=" "):
    """Prints each argument on a new line in blue."""
    print(*color_text(*args, c="teal"), sep=sep)


def magenta_print(*args, sep=" "):
    """Prints each argument on a new line in magenta color."""
    print(*color_text(*args, c="magenta"), sep=sep)


def red_print(*args, sep=" "):
    """Prints each argument on a new line in red."""
    print(*color_text(*args, c="red"), sep=sep)


def lime_print(*args, sep=" "):
    """Prints each argument on a new line in red."""
    print(*color_text(*args, c="lime"), sep=sep)


def pink_print(*args, sep=" "):
    """Prints each argument on a new line in red."""
    print(*color_text(*args, c="pink"), sep=sep)


def print_all_colors():
    for color in COLOR_TO_ANSI_MAP:
        color_print(color, c=color)
