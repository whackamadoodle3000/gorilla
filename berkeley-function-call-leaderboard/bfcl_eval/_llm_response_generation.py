import argparse
import multiprocessing as mp
import os
import shutil
import traceback
from collections import defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import threading
import queue
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from bfcl_eval.ace.constants import (
    ACE_PLAYBOOK_SYSTEM_MARKER,
    DEFAULT_PLAYBOOK_PATH,
    DEFAULT_SPLIT_PATH,
)
from bfcl_eval.ace.playbook import PlaybookManager
from bfcl_eval.ace.utils import determine_tool_groups
from bfcl_eval.ace.split_manager import (
    TEST_PARTITION,
    TRAIN_PARTITION,
    ensure_split_exists,
    get_partition_ids,
)
from bfcl_eval.constants.eval_config import (
    PROJECT_ROOT,
    RESULT_PATH,
    TEST_IDS_TO_GENERATE_PATH,
    RESULT_FILE_PATTERN,
)
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
from bfcl_eval.eval_checker.eval_runner_helper import load_file
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.utils import *
from tqdm import tqdm

from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler


ACE_DATASET_SPLIT_CHOICES = {
    "full": None,
    "ace_train": TRAIN_PARTITION,
    "ace_test": TEST_PARTITION,
}


def get_args():
    parser = argparse.ArgumentParser()
    # Refer to model_choice for supported models.
    parser.add_argument("--model", type=str, default="gorilla-openfunctions-v2", nargs="+")
    # Refer to test_categories for supported categories.
    parser.add_argument("--test-category", type=str, default="all", nargs="+")

    # Parameters for the model that you want to test.
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--include-input-log", action="store_true", default=False)
    parser.add_argument("--exclude-state-log", action="store_true", default=False)
    parser.add_argument("--num-threads", required=False, type=int)
    parser.add_argument("--num-gpus", default=1, type=int)
    parser.add_argument("--backend", default="sglang", type=str, choices=["vllm", "sglang"])
    parser.add_argument("--gpu-memory-utilization", default=0.9, type=float)
    parser.add_argument("--result-dir", default=None, type=str)
    parser.add_argument("--run-ids", action="store_true", default=False)
    parser.add_argument("--allow-overwrite", "-o", action="store_true", default=False)
    parser.add_argument(
        "--skip-server-setup",
        action="store_true",
        default=False,
        help="Skip vLLM/SGLang server setup and use existing endpoint specified by the LOCAL_SERVER_ENDPOINT and LOCAL_SERVER_PORT environment variables.",
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="ace_test",
        choices=list(ACE_DATASET_SPLIT_CHOICES.keys()),
        help="Select which ACE dataset split partition to use. 'full' runs on the entire dataset.",
    )
    parser.add_argument(
        "--split-path",
        type=str,
        default=None,
        help="Path to the dataset split JSON relative to project root (defaults to ACE split path).",
    )
    parser.add_argument(
        "--ace",
        action="store_true",
        default=False,
        help="Enable ACE mode: prepend the playbook to prompts during generation.",
    )
    parser.add_argument(
        "--ace-playbook-path",
        type=str,
        default=None,
        help="Path to the ACE playbook JSON (defaults to bfcl_eval/data/ace/playbook.json).",
    )
    # Optional local model path
    parser.add_argument(
        "--local-model-path",
        type=str,
        default=None,
        help="Specify the path to a local directory containing the model's config/tokenizer/weights for fully offline inference. Use this only if the model weights are stored in a location other than the default HF_HOME directory.",
    )
    args = parser.parse_args()

    return args


def build_handler(model_name, temperature):
    config = MODEL_CONFIG_MAPPING[model_name]
    handler = config.model_handler(
        model_name=config.model_name,
        temperature=temperature,
        registry_name=model_name,
        is_fc_model=config.is_fc_model,
    )
    return handler


def get_involved_test_entries(test_category_args, run_ids):
    all_test_categories, all_test_entries_involved = [], []
    if run_ids:
        all_test_categories, all_test_entries_involved = load_test_entries_from_id_file(
            TEST_IDS_TO_GENERATE_PATH
        )

    else:
        all_test_categories = parse_test_category_argument(test_category_args)
        for test_category in all_test_categories:
            all_test_entries_involved.extend(load_dataset_entry(test_category))

    return (
        all_test_categories,
        all_test_entries_involved,
    )


def collect_test_cases(args, model_name, all_test_categories, all_test_entries_involved):
    model_name_dir = model_name.replace("/", "_")
    model_result_dir = args.result_dir / model_name_dir

    existing_result = []
    for test_category in all_test_categories:
        # TODO: Simplify the handling of memory prerequisite entries/categories
        result_file_paths = [
            model_result_dir
            / get_directory_structure_by_category(test_category)
            / get_file_name_by_category(test_category, is_result_file=True)
        ]
        if is_memory(test_category):
            # Memory test cases have the pre-requisite entries in a separate file
            result_file_paths.append(
                model_result_dir
                / get_directory_structure_by_category(test_category)
                / get_file_name_by_category(f"{test_category}_prereq", is_result_file=True)
            )

        for file_path in result_file_paths:
            if file_path.exists():
                # Not allowing overwrite, we will load the existing results
                if not args.allow_overwrite:
                    existing_result.extend(load_file(file_path))
                # Allow overwrite and not running specific test ids, we will delete the existing result file before generating new results
                elif not args.run_ids:
                    file_path.unlink()
                # Allow overwrite and running specific test ids, we will do nothing here
                else:
                    pass

        if is_memory(test_category):
            # We also need to special handle the pre-requisite entries and the snapshot result for memory test cases
            snapshot_folder = model_result_dir / "memory_snapshot" / test_category
            if snapshot_folder.exists():
                if not args.allow_overwrite:
                    pass
                elif not args.run_ids:
                    shutil.rmtree(snapshot_folder)
                else:
                    # TODO: If run_ids and id involes prereq entries, we should just delete those snapshot files
                    # It's not implemented yet, but it won't affect the accuracy, as those files will be overwritten anyway (assume generation success)
                    pass

    existing_ids = [entry["id"] for entry in existing_result]

    test_cases_to_generate = [
        test_case
        for test_case in all_test_entries_involved
        if test_case["id"] not in existing_ids
    ]

    # Skip format sensitivity test cases for FC models
    if (
        any(is_format_sensitivity(test_category) for test_category in all_test_categories)
        and MODEL_CONFIG_MAPPING[model_name].is_fc_model
    ):
        test_cases_to_generate = [
            test_case
            for test_case in test_cases_to_generate
            if not is_format_sensitivity(test_case["id"])
        ]

    test_cases_to_generate = clean_up_memory_prereq_entries(test_cases_to_generate)
    # TODO: Should we move these to the load_dataset_entry function?
    test_cases_to_generate = populate_initial_settings_for_memory_test_cases(
        test_cases_to_generate, model_result_dir
    )
    test_cases_to_generate = populate_initial_settings_for_web_search_test_cases(
        test_cases_to_generate
    )

    return sorted(test_cases_to_generate, key=sort_key)


def _inject_playbook_system_message(test_case: dict, playbook_text: str):
    if not playbook_text:
        return
    question_turns = test_case.get("question", [])
    if not question_turns:
        return
    first_turn = question_turns[0]
    if not isinstance(first_turn, list):
        return
    for message in first_turn:
        if (
            isinstance(message, dict)
            and message.get("role") == "system"
            and ACE_PLAYBOOK_SYSTEM_MARKER in message.get("content", "")
        ):
            return
    insert_idx = 0
    while insert_idx < len(first_turn) and first_turn[insert_idx].get("role") == "system":
        insert_idx += 1
    content = f"{ACE_PLAYBOOK_SYSTEM_MARKER}\n{playbook_text}"
    first_turn.insert(
        insert_idx,
        {
            "role": "system",
            "content": content,
        },
    )


def multi_threaded_inference(
    handler,
    test_case,
    include_input_log,
    exclude_state_log,
    playbook_text=None,
):

    assert type(test_case["function"]) is list

    try:
        test_case = deepcopy(test_case)
        if playbook_text:
            _inject_playbook_system_message(test_case, playbook_text)
        result, metadata = handler.inference(
            test_case, include_input_log, exclude_state_log
        )
    except Exception as e:
        # This is usually the case when the model getting stuck on one particular test case.
        # For example, timeout error or FC model returning invalid JSON response.
        # Since temperature is already set to 0.001, retrying the same test case will not help.
        # So we continue the generation process and record the error message as the model response
        error_block = (
            "-" * 100
            + "\n❗️❗️ Error occurred during inference. Continuing to next test case.\n"
            + f"❗️❗️ Test case ID: {test_case['id']}, Error: {str(e)}\n"
            + traceback.format_exc(limit=10)
            + "-" * 100
        )
        tqdm.write(error_block)

        result = f"Error during inference: {str(e)}"
        metadata = {"traceback": traceback.format_exc()}

    result_to_write = {
        "id": test_case["id"],
        "result": result,
        **metadata,
    }

    return result_to_write


def generate_results(args, model_name, test_cases_total):
    handler = build_handler(model_name, args.temperature)

    if isinstance(handler, OSSHandler):
        handler: OSSHandler
        is_oss_model = True
        # For OSS models, if the user didn't explicitly set the number of threads,
        # we default to 100 threads to speed up the inference.
        num_threads = (
            args.num_threads
            if args.num_threads is not None
            else LOCAL_SERVER_MAX_CONCURRENT_REQUEST
        )
    else:
        handler: BaseHandler
        is_oss_model = False
        num_threads = args.num_threads if args.num_threads is not None else 1

    # Use a separate thread to write the results to the file to avoid concurrent IO issues
    def _writer():
        """Consume result dicts from the queue and write them with exclusive access."""
        while True:
            item = write_queue.get()
            if item is None:
                break
            handler.write(item, result_dir=args.result_dir, update_mode=args.run_ids)
            write_queue.task_done()

    write_queue: queue.Queue = queue.Queue()

    writer_thread = threading.Thread(target=_writer, daemon=True)
    writer_thread.start()

    playbook_manager = None

    def render_playbook_text(test_case: dict) -> Optional[str]:
        if playbook_manager is None:
            return None
        tool_groups = list(dict.fromkeys(determine_tool_groups(test_case)))
        return playbook_manager.to_prompt_string(
            focus_sections=tool_groups or None,
            max_sections=len(tool_groups) if tool_groups else None,
        )

    if getattr(args, "ace", False):
        playbook_manager = PlaybookManager(args.ace_playbook_path)

    try:
        if is_oss_model:
            handler.spin_up_local_server(
                num_gpus=args.num_gpus,
                gpu_memory_utilization=args.gpu_memory_utilization,
                backend=args.backend,
                skip_server_setup=args.skip_server_setup,
                local_model_path=args.local_model_path,
            )

        # ───── dependency bookkeeping ──────────────────────────────
        dependencies = {
            test_case["id"]: set(test_case.get("depends_on", []))
            for test_case in test_cases_total
        }
        children_of = defaultdict(list)
        for test_case in test_cases_total:
            for dependency_id in test_case.get("depends_on", []):
                children_of[dependency_id].append(test_case["id"])

        id_to_test_case = {test_case["id"]: test_case for test_case in test_cases_total}

        ready_queue = deque(
            [
                test_case_id
                for test_case_id, dependency_ids in dependencies.items()
                if not dependency_ids
            ]
        )
        in_flight: dict[Future, str] = {}  # future -> test_case_id
        completed = set()

        with ThreadPoolExecutor(max_workers=num_threads) as pool, tqdm(
            total=len(test_cases_total),
            desc=f"Generating results for {model_name}",
            position=0,         
            leave=True,           
            dynamic_ncols=True,   
            mininterval=0.2,      
            smoothing=0.1,        
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:

            # seed initial ready tasks
            while ready_queue and len(in_flight) < num_threads:
                test_case_id = ready_queue.popleft()
                test_case = id_to_test_case[test_case_id]
                future = pool.submit(
                    multi_threaded_inference,
                    handler,
                    test_case,
                    args.include_input_log,
                    args.exclude_state_log,
                    render_playbook_text(test_case),
                )
                in_flight[future] = test_case_id

            # main scheduler loop
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    test_case_id = in_flight.pop(future)
                    result_dict = future.result()

                    # Enqueue the result for the writer thread to handle file IO
                    write_queue.put(result_dict)

                    # Update progress bar right after inference completes
                    pbar.update()
                    completed.add(test_case_id)

                    # unlock children
                    for child_id in children_of[test_case_id]:
                        dependencies[child_id].discard(test_case_id)
                        if not dependencies[child_id]:
                            ready_queue.append(child_id)

                # refill the pool up to max_workers
                while ready_queue and len(in_flight) < num_threads:
                    test_case_id = ready_queue.popleft()
                    test_case = id_to_test_case[test_case_id]
                    future = pool.submit(
                        multi_threaded_inference,
                        handler,
                        test_case,
                        args.include_input_log,
                        args.exclude_state_log,
                        render_playbook_text(test_case),
                    )
                    in_flight[future] = test_case_id

    finally:
        # Signal writer thread to finish and wait for it
        write_queue.put(None)
        writer_thread.join()

        if is_oss_model:
            handler.shutdown_local_server()


def main(args):

    # Note: The following environment variables are needed for the memory vector store implementation
    # Otherwise you get segfault or huggingface tokenizer warnings
    # disable HuggingFace tokenizers’ thread pool
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # limit all OpenMP/MKL threads to 1
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    # use spawn method for multiprocessing
    mp.set_start_method("spawn", force=True)

    if type(args.model) is not list:
        args.model = [args.model]
    if type(args.test_category) is not list:
        args.test_category = [args.test_category]

    (
        all_test_categories,
        all_test_entries_involved,
    ) = get_involved_test_entries(args.test_category, args.run_ids)

    for model_name in args.model:
        if model_name not in MODEL_CONFIG_MAPPING:
            raise ValueError(
                f"Unknown model_name '{model_name}'.\n"
                "• For officially supported models, please refer to `SUPPORTED_MODELS.md`.\n"
                "• For running new models, please refer to `README.md` and `CONTRIBUTING.md`."
            )
    tqdm.write(f"Generating results for {args.model}")
    if args.run_ids:
        tqdm.write("Running specific test cases. Ignoring `--test-category` argument.")
    else:
        tqdm.write(f"Running full test cases for categories: {all_test_categories}.")

    dataset_partition = ACE_DATASET_SPLIT_CHOICES.get(args.dataset_split)
    if args.split_path is not None:
        split_path = Path(args.split_path)
        if not split_path.is_absolute():
            split_path = PROJECT_ROOT / split_path
    else:
        split_path = DEFAULT_SPLIT_PATH
    args.split_path = split_path

    if args.ace_playbook_path is not None:
        playbook_path = Path(args.ace_playbook_path)
        if not playbook_path.is_absolute():
            playbook_path = PROJECT_ROOT / playbook_path
    else:
        playbook_path = DEFAULT_PLAYBOOK_PATH
    args.ace_playbook_path = playbook_path

    if dataset_partition:
        ensure_split_exists(output_path=split_path)
        allowed_ids = get_partition_ids(dataset_partition, path=split_path)
    else:
        allowed_ids = None

    if any(is_format_sensitivity(test_category) for test_category in all_test_categories):
        for model_name in args.model:
            if MODEL_CONFIG_MAPPING[model_name].is_fc_model:
                tqdm.write(
                    "⚠️ Warning: Format sensitivity test cases are only supported for prompting (non-FC) models. "
                    f"Since {model_name} is a FC model based on its config, the format sensitivity test cases will be skipped."
                )

    if args.result_dir is not None:
        args.result_dir = PROJECT_ROOT / args.result_dir
    else:
        args.result_dir = RESULT_PATH

    for model_name in args.model:
        test_cases_total = collect_test_cases(
            args,
            model_name,
            all_test_categories,
            deepcopy(all_test_entries_involved),
        )
        if allowed_ids is not None:
            test_cases_total = [
                test_case for test_case in test_cases_total if test_case["id"] in allowed_ids
            ]

        if len(test_cases_total) == 0:
            tqdm.write(
                f"✅ All selected test cases have been previously generated for {model_name}. No new test cases to generate."
            )
        else:
            generate_results(args, model_name, test_cases_total)
            # Sort the result files by id at the end
            for model_result_json in args.result_dir.rglob(RESULT_FILE_PATTERN):
                sort_file_content_by_id(model_result_json)
