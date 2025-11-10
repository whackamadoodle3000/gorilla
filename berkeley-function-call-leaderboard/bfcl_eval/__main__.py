import csv
from datetime import datetime
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import typer
from importlib.metadata import version as _version
from bfcl_eval._llm_response_generation import main as generation_main
from bfcl_eval.ace.constants import (
    DEFAULT_PLAYBOOK_PATH,
    DEFAULT_SPLIT_PATH,
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
)
from bfcl_eval.ace.pipeline import train_playbook
from bfcl_eval.ace.playbook import PlaybookManager
from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING
from bfcl_eval.constants.eval_config import (
    DOTENV_PATH,
    PROJECT_ROOT,
    RESULT_PATH,
    SCORE_PATH,
)
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.eval_runner import main as evaluation_main
from dotenv import load_dotenv
from tabulate import tabulate


class ExecutionOrderGroup(typer.core.TyperGroup):
    def list_commands(self, ctx):
        return [
            "models",
            "test-categories",
            "generate",
            "ace-train-playbook",
            "ace-reset-playbook",
            "results",
            "evaluate",
            "scores",
            "version",
        ]


cli = typer.Typer(
    context_settings=dict(help_option_names=["-h", "--help"]),
    no_args_is_help=True,
    cls=ExecutionOrderGroup,
)


def handle_multiple_input(input_str):
    """
    Input is like 'a,b,c,d', we need to transform it to ['a', 'b', 'c', 'd'] because that's the expected format in the actual main funciton
    """
    if input_str is None:
        """
        Cannot return None here, as typer will check the length of the return value and len(None) will raise an error
        But when default is None, an empty list will be internally converted to None, and so the pipeline still works as expected
        ```
        if default_value is None and len(value) == 0:
            return None
        ```
        """
        return []

    return [item.strip() for item in ",".join(input_str).split(",") if item.strip()]

@cli.command()
def version():
    """
    Show the bfcl version. PyPI versions are in development, please rely on the commit hash for reproducibility.
    """
    print(f"bfcl version: {_version('bfcl')} \nNote: pypi versions are in development, please rely on the commit hash for reproducibility.")

@cli.command()
def test_categories():
    """
    List available test categories.
    """
    table = tabulate(
        [
            (category, "\n".join(test for test in tests))
            for category, tests in TEST_COLLECTION_MAPPING.items()
        ],
        headers=["Test category", "Test names"],
        tablefmt="grid",
    )
    print(table)


@cli.command()
def models():
    """
    List available models.
    """
    table = tabulate(
        [[model] for model in MODEL_CONFIG_MAPPING.keys()],
        tablefmt="plain",
        colalign=("left",),
    )
    print(table)


@cli.command()
def generate(
    model: List[str] = typer.Option(
        ["gorilla-openfunctions-v2"], 
        help="A list of model names to generate the llm response. Use commas to separate multiple models.",
        callback=handle_multiple_input
    ),
    test_category: List[str] = typer.Option(
        ["all"], 
        help="A list of test categories to run the evaluation on. Use commas to separate multiple test categories.",
        callback=handle_multiple_input
    ),
    temperature: float = typer.Option(
        0.001, help="The temperature parameter for the model."
    ),
    include_input_log: bool = typer.Option(
        False,
        "--include-input-log",
        help="Include the fully-transformed input to the model inference endpoint in the inference log; only relevant for debugging input integrity and format.",
    ),
    exclude_state_log: bool = typer.Option(
        False,
        "--exclude-state-log",
        help="Exclude info about the state of each API system after each turn in the inference log; only relevant for multi-turn categories.",
    ),
    num_gpus: int = typer.Option(1, help="The number of GPUs to use."),
    num_threads: Optional[int] = typer.Option(None, help="The number of threads to use."),
    gpu_memory_utilization: float = typer.Option(0.9, help="The GPU memory utilization."),
    backend: str = typer.Option("sglang", help="The backend to use for the model."),
    skip_server_setup: bool = typer.Option(
        False,
        "--skip-server-setup",
        help="Skip vLLM/SGLang server setup and use existing endpoint specified by the LOCAL_SERVER_ENDPOINT and LOCAL_SERVER_PORT environment variables.",
    ),
    local_model_path: Optional[str] = typer.Option(
        None,
        "--local-model-path",
        help="Specify the path to a local directory containing the model's config/tokenizer/weights for fully offline inference. Use this only if the model weights are stored in a location other than the default HF_HOME directory.",
    ),
    result_dir: str = typer.Option(
        RESULT_PATH,
        "--result-dir",
        help="Path to the folder where output files will be stored; Path should be relative to the `berkeley-function-call-leaderboard` root folder",
    ),
    allow_overwrite: bool = typer.Option(
        False,
        "--allow-overwrite",
        "-o",
        help="Allow overwriting existing results for regeneration.",
    ),
    run_ids: bool = typer.Option(
        False,
        "--run-ids",
        help="If true, also run the test entry mentioned in the test_case_ids_to_generate.json file, in addition to the --test_category argument.",
    ),
    dataset_split: str = typer.Option(
        "ace_test",
        "--dataset-split",
        help="Dataset partition to evaluate on (ace_train, ace_test, or full).",
    ),
    split_path: Optional[str] = typer.Option(
        None,
        "--split-path",
        help="Path to the dataset split JSON relative to project root (defaults to bfcl_eval/data/ace/dataset_split.json).",
    ),
    ace: bool = typer.Option(
        False,
        "--ace",
        help="Enable ACE mode by prepending the playbook to each prompt.",
    ),
    ace_playbook_path: Optional[str] = typer.Option(
        None,
        "--ace-playbook-path",
        help="Path to the ACE playbook JSON (defaults to bfcl_eval/data/ace/playbook.json).",
    ),
):
    """
    Generate the LLM response for one or more models on a test-category (same as openfunctions_evaluation.py).
    """

    dataset_split = (dataset_split or "ace_test").lower()
    if dataset_split not in {"ace_train", "ace_test", "full"}:
        raise typer.BadParameter(
            "dataset-split must be one of: ace_train, ace_test, full."
        )

    args = SimpleNamespace(
        model=model,
        test_category=test_category,
        temperature=temperature,
        include_input_log=include_input_log,
        exclude_state_log=exclude_state_log,
        num_gpus=num_gpus,
        num_threads=num_threads,
        gpu_memory_utilization=gpu_memory_utilization,
        backend=backend,
        skip_server_setup=skip_server_setup,
        local_model_path=local_model_path,
        result_dir=result_dir,
        allow_overwrite=allow_overwrite,
        run_ids=run_ids,
        dataset_split=dataset_split,
        split_path=split_path,
        ace=ace,
        ace_playbook_path=ace_playbook_path,
    )
    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)  # Load the .env file
    generation_main(args)


@cli.command("ace-train-playbook")
def ace_train_playbook(
    generator_model: str = typer.Option(
        "DeepSeek-V3.2-Exp-FC",
        "--generator-model",
        help="Model used during the generator phase (should support tool calling).",
    ),
    reflector_model: str = typer.Option(
        "deepseek-chat",
        "--reflector-model",
        help="Model used to produce reflections.",
    ),
    curator_model: str = typer.Option(
        "deepseek-chat",
        "--curator-model",
        help="Model used to emit playbook operations.",
    ),
    generator_temperature: float = typer.Option(
        0.001,
        help="Temperature for the generator model.",
    ),
    completion_temperature: float = typer.Option(
        0.0,
        help="Temperature for the reflector and curator models.",
    ),
    playbook_path: Optional[str] = typer.Option(
        None,
        "--playbook-path",
        help="Where to store the ACE playbook JSON (defaults to bfcl_eval/data/ace/playbook.json).",
    ),
    split_path: Optional[str] = typer.Option(
        None,
        "--split-path",
        help="Path to the dataset split JSON (defaults to bfcl_eval/data/ace/dataset_split.json).",
    ),
    limit: Optional[int] = typer.Option(
        None,
        help="Optionally limit the number of training samples processed.",
    ),
    start_offset: int = typer.Option(
        0,
        "--start-offset",
        help="Skip this many shuffled training samples before beginning (resume support).",
    ),
    regenerate_split: bool = typer.Option(
        False,
        "--regenerate-split",
        help="If set, the dataset split will be remade before training.",
    ),
    split_seed: int = typer.Option(
        DEFAULT_SPLIT_SEED,
        "--split-seed",
        help="Random seed used when (re)generating the dataset split.",
    ),
    train_ratio: float = typer.Option(
        DEFAULT_TRAIN_RATIO,
        "--train-ratio",
        help="Portion of data assigned to the training partition when remaking the split.",
    ),
    num_gpus: int = typer.Option(
        1,
        help="Number of GPUs for OSS models (ignored for API models).",
    ),
    backend: str = typer.Option(
        "sglang",
        help="Backend to use when spinning up OSS models.",
    ),
    gpu_memory_utilization: float = typer.Option(
        0.9,
        help="GPU memory utilization for OSS models.",
    ),
    skip_server_setup: bool = typer.Option(
        False,
        "--skip-server-setup",
        help="Skip local server setup for already running OSS endpoints.",
    ),
    local_model_path: Optional[str] = typer.Option(
        None,
        "--local-model-path",
        help="Custom path to local model weights for OSS models.",
    ),
):
    """
    Build or update the ACE playbook by running the Generator → Reflector → Curator pipeline.
    """
    resolved_playbook_path = (
        Path(playbook_path) if playbook_path else DEFAULT_PLAYBOOK_PATH
    )
    if not resolved_playbook_path.is_absolute():
        resolved_playbook_path = PROJECT_ROOT / resolved_playbook_path

    resolved_split_path = Path(split_path) if split_path else DEFAULT_SPLIT_PATH
    if not resolved_split_path.is_absolute():
        resolved_split_path = PROJECT_ROOT / resolved_split_path

    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)
    train_playbook(
        playbook_path=resolved_playbook_path,
        split_path=resolved_split_path,
        split_seed=split_seed,
        train_ratio=train_ratio,
        generator_model=generator_model,
        reflector_model=reflector_model,
        curator_model=curator_model,
        generator_temperature=generator_temperature,
        completion_temperature=completion_temperature,
        limit=limit,
        start_offset=start_offset,
        regenerate_split=regenerate_split,
        num_gpus=num_gpus,
        backend=backend,
        gpu_memory_utilization=gpu_memory_utilization,
        skip_server_setup=skip_server_setup,
        local_model_path=local_model_path,
    )


@cli.command("ace-reset-playbook")
def ace_reset_playbook(
    playbook_path: Optional[str] = typer.Option(
        None,
        "--playbook-path",
        help="Target playbook file to reset (defaults to bfcl_eval/data/ace/playbook.json).",
    )
) -> None:
    """
    Reset the ACE playbook to an empty state.
    """
    resolved_playbook_path = (
        Path(playbook_path) if playbook_path else DEFAULT_PLAYBOOK_PATH
    )
    if not resolved_playbook_path.is_absolute():
        resolved_playbook_path = PROJECT_ROOT / resolved_playbook_path

    PlaybookManager(resolved_playbook_path, reset=True)
    typer.echo(f"Reset playbook at {resolved_playbook_path}")


@cli.command()
def results(
    result_dir: str = typer.Option(
        None,
        "--result-dir",
        help="Relative path to the model response folder, if different from the default; Path should be relative to the `berkeley-function-call-leaderboard` root folder",
    ),
):
    """
    List the results available for evaluation.
    """

    def display_name(name: str):
        """
        Undo the / -> _ transformation if it happened.

        Args:
            name (str): The name of the model in the result directory.

        Returns:
            str: The original name of the model.
        """
        if name not in MODEL_CONFIG_MAPPING:
            candidate = name.replace("_", "/")
            if candidate in MODEL_CONFIG_MAPPING:
                return candidate
            print(f"Unknown model name: {name}")
        return name

    if result_dir is None:
        result_dir = RESULT_PATH
    else:
        result_dir = (PROJECT_ROOT / result_dir).resolve()

    results_data = []
    for dir in result_dir.iterdir():
        # Check if it is a directory and not a file
        if not dir.is_dir():
            continue

        results_data.append(
            (
                display_name(dir.name),
                datetime.fromtimestamp(dir.stat().st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
            )
        )

    print(
        tabulate(
            results_data,
            headers=["Model name", "Creation time"],
            tablefmt="pretty",
        )
    )


@cli.command()
def evaluate(
    model: List[str] = typer.Option(
        None, 
        help="A list of model names to evaluate.",
        callback=handle_multiple_input
    ),
    test_category: List[str] = typer.Option(
        ["all"], 
        help="A list of test categories to run the evaluation on.",
        callback=handle_multiple_input
    ),
    result_dir: str = typer.Option(
        None,
        "--result-dir",
        help="Relative path to the model response folder, if different from the default; Path should be relative to the `berkeley-function-call-leaderboard` root folder",
    ),
    score_dir: str = typer.Option(
        None,
        "--score-dir",
        help="Relative path to the evaluation score folder, if different from the default; Path should be relative to the `berkeley-function-call-leaderboard` root folder",
    ),
    partial_eval: bool = typer.Option(
        False,
        "--partial-eval",
        help="Run evaluation on a partial set of benchmark entries (eg. entries present in the model result files) without raising for missing IDs.",
    ),
):
    """
    Evaluate results from run of one or more models on a test-category (same as eval_runner.py).
    """

    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)  # Load the .env file
    evaluation_main(model, test_category, result_dir, score_dir, partial_eval)


@cli.command()
def scores(
    score_dir: str = typer.Option(
        None,
        "--score-dir",
        help="Relative path to the evaluation score folder, if different from the default; Path should be relative to the `berkeley-function-call-leaderboard` root folder",
    ),
):
    """
    Display the leaderboard.
    """

    def truncate(text, length=22):
        return (text[:length] + "...") if len(text) > length else text

    if score_dir is None:
        score_dir = SCORE_PATH
    else:
        score_dir = (PROJECT_ROOT / score_dir).resolve()
    # files = ["./score/data_non_live.csv", "./score/data_live.csv", "./score/data_overall.csv"]
    file = score_dir / "data_overall.csv"

    selected_columns = [
        "Rank",
        "Model",
        "Overall Acc",
        "Non-Live AST Acc",
        "Non-Live Exec Acc",
        "Live Acc",
        "Multi Turn Acc",
        "Relevance Detection",
        "Irrelevance Detection",
    ]

    if file.exists():
        with open(file, newline="") as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)  # Read the header row
            column_indices = [headers.index(col) for col in selected_columns]
            data = [
                [row[i] for i in column_indices] for row in reader
            ]  # Read the rest of the data
            selected_columns = selected_columns[:-2] + [
                "Relevance",
                "Irrelevance",
            ]  # Shorten the column names
            print(tabulate(data, headers=selected_columns, tablefmt="grid"))
    else:
        print(f"\nFile {file} not found.\n")


if __name__ == "__main__":
    cli()
