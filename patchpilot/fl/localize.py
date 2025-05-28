import argparse
import concurrent.futures
import json
import os
from datasets import load_dataset

from patchpilot.fl.FL import LLMFL
from patchpilot.repair.repair import poc_info_prompt
from patchpilot.reproduce.task import parse_task_list_file
from patchpilot.util.preprocess_data import (
    filter_none_python,
    filter_out_test_files,
)
from patchpilot.util.utils import (
    load_existing_instance_ids,
    load_json,
    load_jsonl,
    setup_logger,
)
from get_repo_structure.get_repo_structure import (
    get_project_structure_from_scratch,
)

# SET THIS IF YOU WANT TO USE THE PREPROCESSED FILES
PROJECT_STRUCTURE = os.environ.get("PROJECT_STRUCTURE", None)


def store_reasoning_data(instance_id, prompt, trajs, args, id=-1, save_dir="", **kwargs):
    reasoning_output = os.path.join(args.output_folder, "related_reasoning.jsonl")

    reasoning_data = {
        "instance_id": instance_id,
        "prompt": prompt,
        "reasoning_content": [traj.get("reasoning_content", "") for traj in trajs],
        "response": [traj["response"] for traj in trajs],
        **kwargs
    }

    os.makedirs(os.path.dirname(reasoning_output), exist_ok=True)
    with open(reasoning_output, "a") as f:
        f.write(json.dumps(reasoning_data) + "\n")

    print("===================================================================")
    print(f"The reasoning output is at {reasoning_output}")
    print("===================================================================")


def localize_instance(
        bug, args, swe_bench_data, start_file_locs, existing_instance_ids
):
    instance_id = bug["instance_id"]

    log_file = os.path.join(
        args.output_folder, "localization_logs", f"{instance_id}.log"
    )
    if args.target_id is not None:
        if args.target_id != bug["instance_id"]:
            return

    logger = setup_logger(log_file)
    logger.info(f"Processing bug {instance_id}")

    if bug["instance_id"] in existing_instance_ids:
        logger.info(f"Skipping existing instance_id: {bug['instance_id']}")
        return

    if PROJECT_STRUCTURE is not None:
        project_file = os.path.join(PROJECT_STRUCTURE, bug["instance_id"] + ".json")
        file_json = load_json(project_file)

    else:
        # we need to get the project structure directly
        file_json = get_project_structure_from_scratch(
            bug["repo"], bug["base_commit"], bug["instance_id"], "playground", "structure", args
        )

    coverage_info = {
        "coverage_dict": {},
        "commit_info": {},
    }
    reproduce_info = ""

    logger.info(f"================ localize {instance_id} ================")

    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]
    structure = file_json["structure"]

    filter_none_python(structure)  # some basic filtering steps

    # filter out test files (unless its pytest)
    if not file_json["instance_id"].startswith("pytest"):
        filter_out_test_files(structure)

    found_files = []
    found_related_locs = []
    found_edit_locs = []
    additional_artifact_loc_file = None
    additional_artifact_loc_related = None
    additional_artifact_loc_edit_location = None
    file_traj, related_loc_traj, edit_loc_traj = {}, {}, {}

    # step 0: give llm a chance to search for a string in the problem statement
    search_str_with_file = dict()
    if args.search_level:
        fl = LLMFL(
            file_json["instance_id"],
            structure,
            problem_statement,
            args.model,
            args.backend,
            logger,
            args.match_partial_paths,
            args.temperature,
            port=args.port,
        )
        search_str_with_file = fl.search_in_problem_statement(reproduce_info)
    # file level localization
    if args.file_level:
        print(f"========== FILE LEVEL {instance_id} START ===========")
        fl = LLMFL(
            file_json["instance_id"],
            structure,
            problem_statement,
            args.model,
            args.backend,
            logger,
            args.match_partial_paths,
            args.temperature,
            port=args.port,
        )
        found_files, additional_artifact_loc_file, file_traj, raw_trajs, message = fl.localize(
            mock=args.mock,
            match_partial_paths=args.match_partial_paths,
            search_res_files=search_str_with_file,
            num_samples=args.num_samples,
            top_n=args.top_n,
            coverage_info=coverage_info,
            reasoning_mode=args.reasoning_mode
        )
        # store_reasoning_data(instance_id, message, raw_trajs, args, save_dir="reasoning_data/file_level", found_files=found_files)
        print(f"========== FILE LEVEL {instance_id} ENDS WITH RESULTS {found_files} ===========")

    # direct line level localization
    if args.direct_line_level:
        print(f"========== DIRECT LINE LEVEL {instance_id} Starts===========")
        pred_files = found_files[: args.top_n]
        fl = LLMFL(
            instance_id,
            structure,
            problem_statement,
            args.model,
            args.backend,
            logger,
            args.match_partial_paths,
            args.temperature,
            port=args.port
        )
        (
            found_edit_locs,
            additional_artifact_loc_edit_location,
            edit_loc_traj,
            raw_trajs,
            message,
            raw_trajs_list,
            message_list
        ) = fl.localize_line_from_files(
            pred_files,
            num_samples=args.num_samples,
            args=args,
        )
        additional_artifact_loc_edit_location = [additional_artifact_loc_edit_location]
        print(f"========== DIRECT LINE LEVEL {instance_id} ENDS WITH RESULTS {found_edit_locs} ===========")

    # skip file level localization if file level results is provided
    if not args.direct_line_level:
        # related class, functions, global var localization
        pred_files = []
        message = ""
        raw_trajs = {}
        if args.related_level:
            print(f"========== RELATED LEVEL {instance_id} STARTS===========")
            if len(found_files) != 0:
                pred_files = found_files[: args.top_n]
                fl = LLMFL(
                    file_json["instance_id"],
                    structure,
                    problem_statement,
                    args.model,
                    args.backend,
                    logger,
                    args.match_partial_paths,
                    args.temperature,
                    port=args.port,
                )
                additional_artifact_loc_related = []
                found_related_locs = []
                related_loc_traj = {}
                if args.compress:
                    (
                        found_related_locs,
                        additional_artifact_loc_related,
                        related_loc_traj,
                        raw_trajs,
                        message,
                    ) = fl.localize_function_from_compressed_files(
                        pred_files, mock=args.mock, num_samples=args.num_samples, coverage_info=coverage_info, reasoning_mode=args.reasoning_mode, args=args
                    )
                    additional_artifact_loc_related = [additional_artifact_loc_related]
                else:
                    assert False, "Not implemented yet."

        coarse_found_locs = {}
        for i, pred_file in enumerate(pred_files):
            if len(found_related_locs) > i:
                coarse_found_locs[pred_file] = found_related_locs[i]
        print(f"========== RELATED LEVEL {instance_id} ENDS WITH RESULTS {found_related_locs} ===========")

    if args.fine_grain_line_level and not args.direct_line_level:
        print(f"========== LINE LEVEL {instance_id} STARTS===========")
        # Only supports the following args for now
        code_graph_context = None
        pred_files = found_files[: args.top_n]
        fl = LLMFL(
            instance_id,
            structure,
            problem_statement,
            args.model,
            args.backend,
            logger,
            args.match_partial_paths,
            args.temperature,
            port=args.port,
        )
        (
            found_edit_locs,
            additional_artifact_loc_edit_location,
            edit_loc_traj,
            raw_trajs,
            message,
        ) = fl.localize_line_from_coarse_function_locs(
            pred_files,
            coarse_found_locs,
            context_window=args.context_window,
            add_space=args.add_space,
            code_graph=args.repo_graph,
            code_graph_context=code_graph_context,
            no_line_number=args.no_line_number,
            sticky_scroll=args.sticky_scroll,
            mock=args.mock,
            num_samples=args.num_samples,
            coverage_info=coverage_info,
            reasoning_mode=args.reasoning_mode
        )
        additional_artifact_loc_edit_location = [additional_artifact_loc_edit_location]
        print(f"========== LINE LEVEL {instance_id} ENDS WITH RESULTS {found_edit_locs} ===========")

    with open(args.output_file, "a") as f:
        f.write(
            json.dumps(
                {
                    "instance_id": file_json["instance_id"],
                    "found_files": found_files,
                    "search_result": search_str_with_file,
                    "additional_artifact_loc_file": additional_artifact_loc_file,
                    "file_traj": file_traj,
                    "found_related_locs": found_related_locs,
                    "additional_artifact_loc_related": additional_artifact_loc_related,
                    "related_loc_traj": related_loc_traj,
                    "found_edit_locs": found_edit_locs,
                    "additional_artifact_loc_edit_location": additional_artifact_loc_edit_location,
                    "edit_loc_traj": edit_loc_traj,
                }
            )
            + "\n"
        )


def localize(args):
    if args.benchmark == "lite":
        swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split=args.split)
    elif args.benchmark == "verified":
        swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Verified", split=args.split)
    elif args.benchmark == "full":
        swe_bench_data = load_dataset("princeton-nlp/SWE-bench", split=args.split)
    else:
        swe_bench_data = None
    start_file_locs = load_jsonl(args.start_file) if args.start_file else None
    existing_instance_ids = (
        load_existing_instance_ids(args.output_file) if args.skip_existing else set()
    )
    all_task_ids = []
    if args.task_list_file is not None:
        all_task_ids = parse_task_list_file(args.task_list_file)
    elif args.target_id is not None:
        all_task_ids = [args.target_id]
    else:
        for bug in swe_bench_data:
            all_task_ids.append(bug["instance_id"])

    if args.num_threads == 1:
        for bug in swe_bench_data:
            if bug["instance_id"] in all_task_ids:
                localize_instance(
                    bug, args, swe_bench_data, start_file_locs, existing_instance_ids
                )
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.num_threads
        ) as executor:
            futures = [
                executor.submit(
                    localize_instance,
                    bug,
                    args,
                    swe_bench_data,
                    start_file_locs,
                    existing_instance_ids,
                )
                for bug in swe_bench_data if bug["instance_id"] in all_task_ids
            ]
            concurrent.futures.wait(futures)


def merge(args):
    """Merge predicted locations."""
    start_file_locs = load_jsonl(args.start_file)

    # Dump each location sample.
    for st_id in range(args.num_samples):
        en_id = st_id
        merged_locs = []
        for locs in start_file_locs:
            merged_found_locs = []
            if "found_edit_locs" in locs and len(locs["found_edit_locs"]):
                merged_found_locs = [
                    "\n".join(x) for x in locs["found_edit_locs"][st_id]
                ]
            merged_locs.append({**locs, "found_edit_locs": merged_found_locs})
        with open(
                f"{args.output_folder}/loc_merged_{st_id}-{en_id}_outputs.jsonl", "w"
        ) as f:
            for data in merged_locs:
                f.write(json.dumps(data) + "\n")

    # Pair wise merge
    for st_id in range(0, args.num_samples - 1, 2):
        en_id = st_id + 1
        print(f"Merging sample {st_id} and {en_id}...")
        merged_locs = []
        for locs in start_file_locs:
            merged_found_locs = []
            if "found_edit_locs" in locs and len(locs["found_edit_locs"]):
                merged_found_locs = [
                    "\n".join(x) for x in locs["found_edit_locs"][st_id]
                ]
                for sample_found_locs in locs["found_edit_locs"][st_id + 1: en_id + 1]:
                    for i, file_found_locs in enumerate(sample_found_locs):
                        if isinstance(file_found_locs, str):
                            merged_found_locs[i] += "\n" + file_found_locs
                        else:
                            merged_found_locs[i] += "\n" + "\n".join(file_found_locs)
            merged_locs.append({**locs, "found_edit_locs": merged_found_locs})
        with open(
                f"{args.output_folder}/loc_merged_{st_id}-{en_id}_outputs.jsonl", "w"
        ) as f:
            for data in merged_locs:
                f.write(json.dumps(data) + "\n")

    ### Merge all
    all_merged_locs = []
    print("Merging all samples...")
    for locs in start_file_locs:
        merged_found_locs = []
        if "found_edit_locs" in locs and len(locs["found_edit_locs"]):
            merged_found_locs = ["\n".join(x) for x in locs["found_edit_locs"][0]]
            for sample_found_locs in locs["found_edit_locs"][1:]:
                for i, file_found_locs in enumerate(sample_found_locs):
                    if isinstance(file_found_locs, str):
                        merged_found_locs[i] += "\n" + file_found_locs
                    else:
                        merged_found_locs[i] += "\n" + "\n".join(file_found_locs)
        all_merged_locs.append({**locs, "found_edit_locs": merged_found_locs})
    with open(f"{args.output_folder}/loc_all_merged_outputs.jsonl", "w") as f:
        for data in all_merged_locs:
            f.write(json.dumps(data) + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="loc_outputs.jsonl")
    parser.add_argument(
        "--start_file",
        type=str,
        help="""previous output file to start with to reduce
        the work, should use in combination without --file_level""",
    )
    parser.add_argument("--search_level", action="store_true")
    parser.add_argument("--file_level", action="store_true")
    parser.add_argument("--direct_line_level", action="store_true")
    parser.add_argument("--related_level", action="store_true")
    parser.add_argument("--fine_grain_line_level", action="store_true")
    parser.add_argument("--review_level", action="store_true")
    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--add_space", action="store_true")
    parser.add_argument("--no_line_number", action="store_true")
    parser.add_argument("--sticky_scroll", action="store_true")
    parser.add_argument("--repo_graph", action="store_true")
    parser.add_argument("--code_graph_dir", type=str, default=None)
    parser.add_argument("--reproduce_folder", type=str)
    parser.add_argument(
        "--match_partial_paths",
        action="store_true",
        help="Whether to match model generated files based on subdirectories of original repository if no full matches can be found",
    )
    parser.add_argument("--context_window", type=int, default=10)
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for creating API requests",
    )
    parser.add_argument("--target_id", type=str)
    parser.add_argument(
        "--task_list_file",
        type=str,
        help="Path to the file that contains all tasks ids to be run.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip localization of instance id's which already contain a localization in the output file.",
    )
    parser.add_argument(
        "--mock", action="store_true", help="Mock run to compute prompt tokens."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="UCSB-SURFI/Co-PatcheR-Loc-Gen-14B",
    )
    parser.add_argument(
        "--backend", type=str, default="openai", choices=["openai", "deepseek", "claude", "opensource"]
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="lite",
        choices=["lite", "verified", "full"],
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "train"],
    )
    parser.add_argument(
        "--save_repo_clone",
        action="store_true",
        help="Whether preserve the repo clone or delete it after use"
    )
    parser.add_argument(
        "--save_structure",
        action="store_true",
        help="Whether or not save the structure in a file"
    )
    parser.add_argument("--reasoning_mode", action="store_true", default=False)
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2951)

    parser.add_argument("--allow_in_other_files", action="store_true")
    parser.add_argument("--allow_more_files", action="store_true")

    args = parser.parse_args()

    import os

    args.output_file = os.path.join(args.output_folder, args.output_file)

    assert (
            not os.path.exists(args.output_file) or args.skip_existing
    ), "Output file already exists and not set to skip existing localizations"

    assert not (
            args.file_level and args.start_file
    ), "Cannot use both file_level and start_file"

    assert not (
            args.file_level and args.fine_grain_line_level and not args.related_level
    ), "Cannot use both file_level and fine_grain_line_level without related_level"

    assert not (
            (not args.file_level) and (not args.start_file)
    ), "Must use either file_level or start_file"

    os.makedirs(os.path.join(args.output_folder, "localization_logs"), exist_ok=True)
    os.makedirs(args.output_folder, exist_ok=True)

    # write the arguments
    with open(f"{args.output_folder}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    if args.merge:
        merge(args)
    else:
        localize(args)


if __name__ == "__main__":
    main()