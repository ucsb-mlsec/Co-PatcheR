#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import json
import os
import sys
from typing import Dict, List
import concurrent.futures
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
from unidiff import PatchSet
from collections import Counter

from patchpilot.util.postprocess_data import normalize_patch
from patchpilot.util.utils import load_jsonl, setup_logger
from patchpilot.reproduce.reproduce import ensure_directory_exists
from patchpilot.repair.bfs import vote_outputs_unwrap
from patchpilot.util.model import make_model
from patchpilot.util.preprocess_data import (
    get_full_file_paths_and_classes_and_functions,
    get_repo_structure,
)
from patchpilot.reproduce.task import parse_task_list_file


vote_patch_prompt = """
We are currently addressing the following issue in our repository. Several candidate patches have been proposed to resolve this issue. Your task is to evaluate each patch in detail and select the one that offers the most effective and general solution.

Analyze the issue and provided patchs according to the following guidelines, you should look each patch at give each patch a score, output the score for each patch:

## Reason about the Scope (5 points):
Reason about the scope of the critical variable, considering the values that should and should not be affected. What situations should the patch handle, and what should it avoid? Ensure the patch correctly targets the issue without impacting unrelated code or values. Score based on the accuracy of the scope.
You should always explicitly infer the scope of the critical variable, output the exact scope of values that should and should not be affected.
It is not a negative factor if the patch introduces complexity logic.

Example:
For instance, if the issue can be triggered by an empty string, you need to explicitly consider whether it can also be triggered by None, an empty list, or other similar values. Prefer patches that only modify the variable triggering the issue. If None does not trigger the issue, the patch should not alter the behavior of None. 
Similarly, if an integer in the issue causes the problem, explicitly evaluate whether other integers can also trigger the issue. Prioritize patches that adjust the scope of the variable in a way that matches the specific values capable of triggering the issue, without impacting unrelated cases.

## Correctness (5 points):
Infer the logical root cause of the issue. Ensure the proposed patch fixes the issue as described in the problem statement and behaves as expected. 

## Reusability of Existing Functions (2 points):
Favor patches that reuse existing functions or utilities.

## Logic Changes(5 points):
If a patch reorders checks, it should get 0 points for this criteria.
You should always explicitly infer whether the checks are reordered and output the result.
If a patch broaden the scope of checks unnecessarily, it should get 0 points for this criteria. 
You should always explicitly infer whether the checks are broadened and output the result.
If a patch doesn't fix the issue completely, it should get 0 points for this criteria.

## Consideration of Structural Interactions (5 points):
Ensure that the patch handles interactions between different parts of the code, such as nested structures, function calls, or data dependencies.
The patch should maintain the integrity of the overall structure while addressing the issue, ensuring that changes in one part do not inadvertently affect other parts. 
Prefer solutions that are robust against changes in the structure or future extensions.

# Minimal Patch (2 points):
The patch should be minimal, only addressing the specific issue described in the problem statement. Avoid making unnecessary changes or introducing new functionality.

## Type (2 points):
If the patch involves checking or modifying the type of the variable, you should consider the context, and prefer types specific to the python project over general ones.

After evaluating each patch based on these criteria, conclude your analysis by stating:
"The best choice is s," where s is the integer ID of the patch you believe is the best option.

Your analysis should not involve copying any code from the patches.
Your analysis should not have any code snippets.
You should compare each patch and score each patch.

Here is the issue text:
--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

Below are some code segments, each from a relevant file. One or more of these files may contain bugs.
--- BEGIN FILE ---
{content}
--- END FILE ---

Here are the candidate patches:
--- BEGIN PATCHES ---
{patches}
--- END PATCHES ---
"""

REPO_ROOT = Path("/scr/playground")


def _apply_patch_on_text(original_text: str, file_patch) -> str:
    try:
        orig_lines = original_text.splitlines(keepends=True)
        res_lines  = []
        orig_idx   = 0

        for hunk in file_patch:
            hunk_start = hunk.source_start - 1

            res_lines.extend(orig_lines[orig_idx:hunk_start])
            orig_idx = hunk_start

            for line in hunk:
                if line.is_context:
                    res_lines.append(orig_lines[orig_idx])
                    orig_idx += 1
                elif line.is_removed:
                    orig_idx += 1
                elif line.is_added:
                    res_lines.append(line.value)

        res_lines.extend(orig_lines[orig_idx:])

        return "".join(res_lines)
    except Exception:
        return ""


def _extract_file_blobs(instance_id: str, patch_text: str):
    """
    parse diff，return:
        edited_files            -> list[str]
        original_file_content   -> list[str]
        new_file_content        -> list[str]
    the length should be the same
    """
    patchset = PatchSet(patch_text)
    edited_files, old_blobs, new_blobs = [], [], []

    repo_dir = REPO_ROOT / instance_id
    for file_patch in patchset:
        fp = file_patch.path
        edited_files.append(fp)
        abs_fp = repo_dir / fp

        # original text
        if abs_fp.exists():
            original_text = abs_fp.read_text(encoding="utf-8", errors="ignore")
        else:
            original_text = ""        # add new file

        old_blobs.append(original_text)

        if file_patch.is_removed_file:
            new_text = ""             # delete file
        else:
            new_text = _apply_patch_on_text(original_text, file_patch)
        new_blobs.append(new_text)

    return edited_files, old_blobs, new_blobs


def normalize_patches(args, instance_id, min_patches):
    normalized_patches = {}
    for min_patch in min_patches:
        patch_text = min_patch

        # 1. parse diff → get the information of diff
        edited_files, orig_blobs, new_blobs = _extract_file_blobs(
            instance_id, patch_text
        )

        # 2. normalize_patch
        normalized_patches[min_patch]= normalize_patch(
            instance_id,
            patch_text,
            orig_blobs,
            new_blobs,
            edited_files,
        )

    return normalized_patches


def majority_voting(args, normalized_patches):
    vote = Counter()
    first_seen = {}
    winner_patch = ""
    for idx, (patch_text, norm_key) in enumerate(normalized_patches.items()):
        key = norm_key.strip()
        if not key:        # empty do not go into voting
            continue
        vote[key] += 1
        first_seen.setdefault(key, idx)   # log first seen

    # get winner
    if vote:
        winner_key = max(vote.keys(), key=lambda k: (vote[k], -first_seen[k]))
        for ptxt, nkey in normalized_patches.items():
            if nkey.strip() == winner_key:
                winner_patch = ptxt
                break

    for key in vote:
        print(vote[key])

    return winner_patch


def get_rank_from_verify_info(args, verify_info, model_patch) -> float:
    if model_patch.strip() == "":
        return sys.maxsize
    poc_test_succeed_llm = verify_info["result"].get('poc_test_succeed_llm', [])
    poc_test_succeed_rule = verify_info["result"].get('poc_test_succeed_rule', [])
    num_failed_poc_llm = len([x for x in poc_test_succeed_llm if not x])
    num_failed_poc_rule = len([x for x in poc_test_succeed_rule if not x])
    increased_failed_tests = 0
    increased_failed_tests = verify_info["result"]["functionality_test_fail_num"][
                                 "new_failed_tests_num"] - verify_info["result"]["functionality_test_fail_num"][
                                 "old_failed_tests_num"]
    rank = num_failed_poc_llm + max(0, increased_failed_tests)*0.1
    return rank


def break_tie(args, loc, instance_id, final_patches, min_patches):
    log_file = os.path.join(
        args.output_folder, "rerank_by_verification", instance_id, f"break_tie.log"
    )
    ensure_directory_exists(os.path.join(
        args.output_folder, "rerank_by_verification", instance_id
    ))
    logger = setup_logger(log_file)

    # get all the information needed
    if args.benchmark == "lite":
        swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    elif args.benchmark == "verified":
        swe_bench_data = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    elif args.benchmark == "full":
        swe_bench_data = load_dataset("princeton-nlp/SWE-bench", split="test")
    else:
        raise ValueError(f"benchmark {args.benchmark} not supported")
    bench_data = [x for x in swe_bench_data if x["instance_id"] == instance_id][0]
    problem_statement = bench_data["problem_statement"]
    pred_files = loc["found_files"][: args.top_n]
    structure = get_repo_structure(
        instance_id, bench_data["repo"], bench_data["base_commit"], "/scr/playground", "/scr/structure", args
    )
    files, _, _ = get_full_file_paths_and_classes_and_functions(structure)
    file_contents = dict()
    for i, pred_file in enumerate(pred_files):
        content = None

        for file_content in files:
            if file_content[0] == pred_file:
                content = "\n".join(file_content[1])
                file_contents[pred_file] = content
                break

        assert content is not None, f"{pred_file} file not found"

    # Construct top-n file context
    file_to_edit_locs = dict()
    for i, pred_file in enumerate(pred_files):
        if "found_edit_locs" in loc and len(loc["found_edit_locs"]) > i:
            file_to_edit_locs[pred_file] = loc["found_edit_locs"][i]

    # let llm vote
    patch_candidate_prompt = ''
    # Limit to maximum 20 candidates to prevent LLM inference issues
    candidates_to_use = min_patches[:20] if len(min_patches) > 20 else min_patches
    for i, patch_candidate in enumerate(candidates_to_use):
        patch_candidate_prompt += f"--- BEGIN PATCH {i + 1} ---\n{patch_candidate}\n--- END PATCH {i + 1} ---\n"

    model = make_model(
        model=args.model,
        logger=logger,
        max_tokens=8192,
        backend=args.backend,
        temperature=0.8,
        batch_size=len(min_patches),
    )
    message = vote_patch_prompt.format(
        problem_statement=problem_statement,
        content="",  # topn_content.rstrip(),
        patches=patch_candidate_prompt,
    ).strip()
    logger.info(f"Instance_id {instance_id}: voting with breaktie message:\n{message}")
    vote_results = [0] * len(min_patches)
    while sum(vote_results) == 0:
        if args.reasoning_mode:
            vote_traj = model.codegen(message, num_samples=1, reasoning_mode=args.reasoning_mode, port=args.port, ip=args.ip)
        else:
            vote_traj = model.codegen(message, num_samples=1, port=args.port, ip=args.ip)
        vote_outputs = [vote_traj["response"] for vote_traj in vote_traj]
        # Update to consider only the candidates that were sent to the LLM
        tmp_vote_results = vote_outputs_unwrap(vote_outputs, len(candidates_to_use))
        
        # Map the votes back to the original indices
        if len(candidates_to_use) < len(min_patches):
            vote_results = [0] * len(min_patches)
            for i, vote in enumerate(tmp_vote_results):
                vote_results[i] = vote
        else:
            vote_results = tmp_vote_results

        if args.reasoning_mode:
            verify_folder = args.verify_folder
            reasoning_output = os.path.join(verify_folder, f"reasoning_data.jsonl")
            reasoning_data = {
                "instance_id": instance_id,
                "prompt": message,
                "reasoning_content": vote_traj[0]["reasoning_content"],
                "response": vote_traj[0]["response"],
            }

            with open(reasoning_output, "a") as f:
                f.write(json.dumps(reasoning_data) + "\n")

    logger.info(f"Instance_id {instance_id}: voting results:\n{vote_results}")
    best_one_idx = sorted(range(len(vote_results)), key=lambda i: vote_results[i], reverse=True)[0]
    final_patches[instance_id] = min_patches[best_one_idx]
    logger.info(f"Instance_id {instance_id}: best one patch:\n{final_patches[instance_id]}")


def get_final_patch_instance(args, instance_id, locs, final_patches, final_ranks, all_predictions):
    patches = all_predictions[instance_id]
    min_value = min(patches.values())
    final_ranks[instance_id] = min_value
    min_patches = [patch for patch in patches if patches[patch] == min_value]
    if len(min_patches) == 1:
        final_patches[instance_id] = min_patches[0]
    elif len(min_patches) > 1:
        if args.llm_voting:
            # use llm to vote
            loc = [loc for loc in locs if loc["instance_id"] == instance_id][0]
            break_tie(args, loc, instance_id, final_patches, min_patches)
        else:
            # use structure consistency
            normalized_patch = normalize_patches(args, instance_id, min_patches)
            final_patches[instance_id] = majority_voting(args, normalized_patch)


def rerank_by_verification(args, num_generated_sample_before, num_generated_sample, best_patch_file=None):
    # rerank the sampled patches based on the verification results

    # key is the instance_id, value is also a dict, with key being the patch and value being the rank (listed above)
    all_predictions = dict()

    # key is the instance_id, value is the patch
    final_patches = dict()
    # key is the instance_id, value is the rank
    final_ranks = dict()

    # Track the number of min_patches for each instance
    min_patches_counts = dict()

    # also output a file containing the final patch passing all verifications (for debugging)

    patches_passed_all_verifications = dict()

    # also output a file containing the final patch passing all functionality tests (for debugging)

    patches_passed_all_functionality_tests_no_poc = dict()

    global reloca_ids
    reloca_ids = []

    for i in range(num_generated_sample_before, num_generated_sample):
        with open(args.raw_output_file.replace(
                ".jsonl", f"_{i}_processed.jsonl"), "r") as f:
            for line in f:
                result = json.loads(line)
                instance_id = result["instance_id"]
                if instance_id not in args.task_ids_to_repair:
                    continue
                if "model_patch" in result:
                    verify_file = args.verify_folder + os.path.join(f"/samples_{i}", instance_id, "verify_outputs.json")
                    if os.path.exists(verify_file):
                        print("checking the verification file", verify_file)
                        with open(verify_file, "r") as f:
                            verify_info = json.load(f)
                        model_patch = result["model_patch"]
                        rank = get_rank_from_verify_info(args, verify_info, model_patch)
                    else:
                        rank = sys.maxsize
                if instance_id not in all_predictions:
                    all_predictions[instance_id] = {result["model_patch"]: rank}
                else:
                    # if there are multiple patches that are exactly the same, we only keep the one with the smallest rank, since the verification res may be unstable
                    all_predictions[instance_id][result["model_patch"]] = min(rank, all_predictions[instance_id].get(
                        result["model_patch"], sys.maxsize))

    # decide the final patches and break ties
    ensure_directory_exists(os.path.join(args.result_folder, "rerank_by_verification"))
    
    # Calculate min_patches counts for each instance before reranking
    min_patches_counts = {}
    for instance_id in all_predictions:
        patches = all_predictions[instance_id]
        min_value = min(patches.values())
        min_patches = [patch for patch in patches if patches[patch] == min_value]
        min_patches_counts[instance_id] = len(min_patches)
    
    # Save min_patches counts to file
    with open(os.path.join(args.result_folder, "min_patches_counts.json"), "w") as f:
        json.dump(min_patches_counts, f, indent=2)
    
    if args.num_threads == 1:
        for instance_id in tqdm(all_predictions, total=len(args.task_ids_to_repair)):
            get_final_patch_instance(args, instance_id, locs_global, final_patches, final_ranks, all_predictions)
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.num_threads
        ) as executor:
            futures = {
                executor.submit(get_final_patch_instance, args, instance_id, locs_global, final_patches, final_ranks, all_predictions): instance_id
                for instance_id in all_predictions
            }
            for future in tqdm(
                    concurrent.futures.as_completed(futures), total=len(args.task_ids_to_repair)
            ):
                result = future.result()

    # save the patch that pass checks
    for instance_id in all_predictions:
        # we need to get the verify_info to check if the poc is executed
        verify_file = args.verify_folder + os.path.join(f"/samples_0", instance_id, "verify_outputs.json")
        has_poc = False
        if os.path.exists(verify_file):
            verify_info = json.load(open(verify_file, "r"))
            if verify_info["result"]["poc_is_executed"]:
                has_poc = True
        if final_ranks[instance_id] == 0:
            patches_passed_all_verifications[instance_id] = final_patches[instance_id]
        if final_ranks[instance_id] == 0 and not has_poc:
            patches_passed_all_functionality_tests_no_poc[instance_id] = final_patches[instance_id]
    # get the indices of the final patches (which sample they are from)
    final_patch_indices = dict()

    for i in range(num_generated_sample):
        with open(args.raw_output_file.replace(".jsonl", f"_{i}_processed.jsonl"), "r") as f:
            for line in f:
                result = json.loads(line)
                if result["instance_id"] in final_patch_indices:
                    continue
                instance_id = result["instance_id"]
                if "model_patch" in result and instance_id in final_patches:
                    if result["model_patch"] == final_patches[instance_id]:
                        final_patch_indices[instance_id] = i

    return final_patches, patches_passed_all_verifications, patches_passed_all_functionality_tests_no_poc, final_ranks, final_patch_indices


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_folder", required=True)
    p.add_argument("--verify_folder", required=True)
    p.add_argument("--loc_file", required=True)
    p.add_argument("--setup_map", required=True)
    p.add_argument("--tasks_map", required=True)
    p.add_argument("--top_n", type=int, default=1)
    p.add_argument("--task_list_file")
    p.add_argument("--target_id")
    p.add_argument("--benchmark", default="lite", choices=["lite", "verified", "full"])
    p.add_argument("--backend", default="openai",
                   choices=["openai", "deepseek", "claude", "opensource"])
    p.add_argument("--model", default="o3-2025-04-03")
    p.add_argument("--num_threads", type=int, default=1)
    p.add_argument("--sample_mod", action="store_true", default=True)
    p.add_argument("--raw_output_file", default="output.jsonl")
    p.add_argument("--llm_voting", action="store_true", default=False)
    p.add_argument("--majority_voting", action="store_true", default=False)
    p.add_argument("--reasoning_mode", action="store_true", default=False)
    p.add_argument("--ip", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=2951)
    p.add_argument(
        "--save_structure",
        action="store_true",
        help="Whether or not save the structure in a file"
    )
    return p.parse_args()


def main():
    args = parse_args()

    global locs_global
    locs_global = load_jsonl(args.loc_file)

    if args.task_list_file:
        task_ids = parse_task_list_file(args.task_list_file)
    elif args.target_id:
        task_ids = [args.target_id]
    else:
        task_ids = [l["instance_id"] for l in locs_global]
    args.task_ids_to_repair = task_ids
    args.raw_output_file = os.path.join(args.output_folder, args.raw_output_file)

    patt = os.path.join(
        args.output_folder,
        args.raw_output_file.replace(".jsonl", "_*_processed.jsonl"))
    num_generated_sample = len(glob.glob(patt))
    if args.llm_voting:
        args.result_folder = os.path.join(args.verify_folder, "llm_voting")
    else:
        args.result_folder = os.path.join(args.verify_folder, "majority_voting")
    ensure_directory_exists(args.result_folder)
    if num_generated_sample == 0:
        print("no *_processed.jsonl files found")
        sys.exit(1)

    if args.sample_mod:
        if os.path.exists(os.path.join(args.result_folder, "final_patches.jsonl")):
            print("final_patches.jsonl already exists, skip reranking")
        else:
            (final_patches,
             patches_passed_all_verifications,
             patches_passed_all_functionality_tests_no_poc,
             final_ranks,
             final_patch_indices) = rerank_by_verification(
                args, 0, num_generated_sample)
            with open(os.path.join(args.result_folder, "final_patches.jsonl"), "w") as f:
                for ins in final_patches:
                    f.write(json.dumps({
                        "model_name_or_path": "patchingagent",
                        "instance_id": ins,
                        "model_patch": final_patches[ins],
                        "rank": final_ranks[ins],
                        "sample_idx": final_patch_indices[ins],
                    }) + "\n")
            with open(os.path.join(args.result_folder, "patches_passed_all_verifications.jsonl"), "w") as f:
                for ins in patches_passed_all_verifications:
                    f.write(json.dumps({
                        "model_name_or_path": "patchingagent",
                        "instance_id": ins,
                        "model_patch": patches_passed_all_verifications[ins],
                    }) + "\n")
            with open(os.path.join(args.result_folder, "patches_passed_all_functionality_tests.jsonl"), "w") as f:
                for ins in patches_passed_all_functionality_tests_no_poc:
                    f.write(json.dumps({
                        "model_name_or_path": "patchingagent",
                        "instance_id": ins,
                        "model_patch": patches_passed_all_functionality_tests_no_poc[ins],
                    }) + "\n")


if __name__ == "__main__":
    main()
