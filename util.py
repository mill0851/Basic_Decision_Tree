import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from information_gain import information_gain

console = Console()


def import_data(data_path: str) -> pd.DataFrame:

    data = pd.read_csv(data_path)
    print(f"dataframe : \n{data}")

    return data

def count_repeating(data: pd.DataFrame) -> list[tuple[pd.DataFrame, str]]:
    result = []

    headers = data.columns

    for header in headers:
        column = data[header]
        frequency_map = {}

        for value in column:
            if value in frequency_map:
                frequency_map[value] += 1
            else:
                frequency_map[value] = 1

        df = pd.DataFrame.from_dict(frequency_map, orient='index', columns=["Occurences"])
        result.append((df, header))

    return result

def plot_table(data: pd.DataFrame, title: str = "") -> None:
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("", style="dim")
    for col in data.columns:
        table.add_column(str(col))
    for idx, row in data.iterrows():
        table.add_row(str(idx), *[str(v) for v in row])
    console.print(table)

# def calculate_information_entropy(data: pd.DataFrame) -> pd.DataFrame:

if __name__ == "__main__":

    data = import_data("song_data.csv")
    data["genre"] = data["genre"].replace({"K-pop": "K-Pop"})

    category_frequencies = count_repeating(data)

    for category in category_frequencies:
        print(f"\nCategory: {category[1]}")
        print(f"frequencies:\n{category[0]}")
        plot_table(category[0], title=category[1])

    fig, table = information_gain(data, "cover-yn")                               # split at the root (Q5)
    fig, table = information_gain(data, "genre", node={"cover-yn": "N"})  # split an internal node (Q7)


    

    


