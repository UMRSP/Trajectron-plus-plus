import sys
import os
import dill
import json
import argparse
import time

import torch
import numpy as np
import pandas as pd

sys.path.append("../../trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--output_tag", help="name tag for output file", type=str)
parser.add_argument("--node_type", help="node type to evaluate", type=str)
args = parser.parse_args()


def load_model(model_dir, env, ts=100):
    """Load model weights, configuration, and initialize Trajectron on CPU."""
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)

    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')
    trajectron.set_environment(env)
    trajectron.set_annealing_params()

    return trajectron, hyperparams


def create_timing_stats():
    """Create counters for one evaluation mode."""
    return {
        'prediction_calls': 0,          # Every eval_stg.predict() call, including empty outputs.
        'nonempty_batches': 0,          # Prediction calls that returned at least one trajectory.
        'total_trajectories': 0,        # Agent/timestep trajectories, excluding Monte-Carlo samples.
        'inference_time': 0.0,          # Time spent inside eval_stg.predict().
        'metrics_time': 0.0,            # Time spent in evaluation.compute_batch_statistics().
    }


def count_predicted_trajectories(predictions):
    """
    Count predicted agent trajectories in Trajectron++ output.

    predictions has the usual Trajectron++ structure:
        {timestep: {node: prediction_array}}

    Each node entry represents one agent trajectory for one prediction timestep.
    This deliberately does NOT multiply by num_samples (20, 2000, etc.).
    """
    if not predictions:
        return 0

    trajectory_count = 0

    for timestep_predictions in predictions.values():
        if timestep_predictions is None:
            continue

        try:
            trajectory_count += len(timestep_predictions)
        except TypeError:
            # Defensive fallback for an unexpected output container.
            trajectory_count += 1

    return trajectory_count


def timed_predict(model, scene, timesteps, prediction_horizon, stats, **predict_kwargs):
    """Run one prediction call and update inference/batch/trajectory counters."""
    inference_start = time.perf_counter()

    predictions = model.predict(
        scene,
        timesteps,
        prediction_horizon,
        **predict_kwargs
    )

    elapsed_inference_time = time.perf_counter() - inference_start
    trajectory_count = count_predicted_trajectories(predictions)

    stats['prediction_calls'] += 1
    stats['inference_time'] += elapsed_inference_time
    stats['total_trajectories'] += trajectory_count

    if trajectory_count > 0:
        stats['nonempty_batches'] += 1

    return predictions


def timed_batch_statistics(predictions, scene, max_hl, ph, env, stats, **statistics_kwargs):
    """Compute evaluation metrics and separately measure metric/postprocessing time."""
    metrics_start = time.perf_counter()

    batch_error_dict = evaluation.compute_batch_statistics(
        predictions,
        scene.dt,
        max_hl=max_hl,
        ph=ph,
        node_type_enum=env.NodeType,
        map=None,
        **statistics_kwargs
    )

    stats['metrics_time'] += time.perf_counter() - metrics_start

    return batch_error_dict


def print_mode_summary(mode_name, stats):
    """Print timing and workload statistics for one evaluation mode."""
    prediction_calls = stats['prediction_calls']
    nonempty_batches = stats['nonempty_batches']
    total_trajectories = stats['total_trajectories']
    inference_time = stats['inference_time']
    metrics_time = stats['metrics_time']

    print("\n" + "=" * 68)
    print(f"{mode_name.upper()} SUMMARY")
    print("=" * 68)
    print(f"Total prediction calls / batches:       {prediction_calls}")
    print(f"Non-empty prediction batches:           {nonempty_batches}")
    print(f"Total predicted agent trajectories:     {total_trajectories}")

    if prediction_calls > 0:
        print(f"Average trajectories / batch:           {total_trajectories / prediction_calls:.2f}")
        print(f"Total inference time:                   {inference_time:.6f} s")
        print(f"Average inference time / batch:         {inference_time / prediction_calls:.6f} s")
    else:
        print("Total inference time:                   0.000000 s")
        print("Average inference time / batch:         N/A")

    if nonempty_batches > 0:
        print(f"Average trajectories / non-empty batch: {total_trajectories / nonempty_batches:.2f}")
    else:
        print("Average trajectories / non-empty batch: N/A")

    if total_trajectories > 0:
        print(f"Average inference time / trajectory:    {(inference_time / total_trajectories) * 1000.0:.6f} ms")
    else:
        print("Average inference time / trajectory:    N/A")

    print(f"Metric/postprocessing time:             {metrics_time:.6f} s")
    print(f"Inference + metric processing time:     {inference_time + metrics_time:.6f} s")


def combine_timing_stats(all_stats):
    """Combine workload/timing values across all evaluation modes."""
    combined_stats = create_timing_stats()

    for stats in all_stats.values():
        for key in combined_stats:
            combined_stats[key] += stats[key]

    return combined_stats


if __name__ == "__main__":
    program_start = time.perf_counter()

    # ============================================================
    # Dataset loader time
    # ============================================================
    dataset_load_start = time.perf_counter()

    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    dataset_load_time = time.perf_counter() - dataset_load_start

    # ============================================================
    # Model loading / model build time
    # ============================================================
    model_load_start = time.perf_counter()

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    model_load_time = time.perf_counter() - model_load_start

    # ============================================================
    # Dataset preprocessing time
    # Includes attention-radius overrides and node graph construction.
    # ============================================================
    dataset_preprocessing_start = time.perf_counter()

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(
            env.attention_radius,
            hyperparams['edge_addition_filter'],
            hyperparams['edge_removal_filter']
        )

    ph = hyperparams['prediction_horizon']
    max_hl = hyperparams['maximum_history_length']

    dataset_preprocessing_time = time.perf_counter() - dataset_preprocessing_start

    print("\n" + "=" * 68)
    print("INITIALIZATION / DATASET PREPARATION TIMING")
    print("=" * 68)
    print(f"Dataset loader time:                   {dataset_load_time:.6f} s")
    print(f"Model load/build time:                 {model_load_time:.6f} s")
    print(f"Dataset preprocessing time:            {dataset_preprocessing_time:.6f} s")
    print("=" * 68)

    mode_stats = {}

    with torch.no_grad():
        # ============================================================
        # MOST LIKELY
        # ============================================================
        ml_stats = create_timing_stats()
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])

        print("-- Evaluating GMM Grid Sampled (Most Likely)")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
            timesteps = np.arange(scene.timesteps)

            predictions = timed_predict(
                eval_stg,
                scene,
                timesteps,
                ph,
                ml_stats,
                num_samples=1,
                min_history_timesteps=7,
                min_future_timesteps=12,
                z_mode=False,
                gmm_mode=True,
                full_dist=True
            )  # This will trigger grid sampling.

            if not predictions:
                continue

            batch_error_dict = timed_batch_statistics(
                predictions,
                scene,
                max_hl,
                ph,
                env,
                ml_stats,
                prune_ph_to_future=True,
                kde=False
            )

            eval_ade_batch_errors = np.hstack(
                (eval_ade_batch_errors, batch_error_dict[args.node_type]['ade'])
            )
            eval_fde_batch_errors = np.hstack(
                (eval_fde_batch_errors, batch_error_dict[args.node_type]['fde'])
            )

        if eval_fde_batch_errors.size > 0:
            print(f"Most Likely mean FDE: {np.mean(eval_fde_batch_errors):.6f}")
        else:
            print("Most Likely mean FDE: N/A (no valid predictions)")

        pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'ml'}).to_csv(
            os.path.join(args.output_path, args.output_tag + '_ade_most_likely.csv')
        )
        pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'ml'}).to_csv(
            os.path.join(args.output_path, args.output_tag + '_fde_most_likely.csv')
        )

        mode_stats['Most Likely'] = ml_stats

        # ============================================================
        # MODE Z
        # ============================================================
        z_mode_stats = create_timing_stats()
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        eval_kde_nll = np.array([])

        print("-- Evaluating Mode Z")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t, t + 10)

                predictions = timed_predict(
                    eval_stg,
                    scene,
                    timesteps,
                    ph,
                    z_mode_stats,
                    num_samples=2000,
                    min_history_timesteps=7,
                    min_future_timesteps=12,
                    z_mode=True,
                    full_dist=False
                )

                if not predictions:
                    continue

                batch_error_dict = timed_batch_statistics(
                    predictions,
                    scene,
                    max_hl,
                    ph,
                    env,
                    z_mode_stats,
                    prune_ph_to_future=True
                )

                eval_ade_batch_errors = np.hstack(
                    (eval_ade_batch_errors, batch_error_dict[args.node_type]['ade'])
                )
                eval_fde_batch_errors = np.hstack(
                    (eval_fde_batch_errors, batch_error_dict[args.node_type]['fde'])
                )
                eval_kde_nll = np.hstack(
                    (eval_kde_nll, batch_error_dict[args.node_type]['kde'])
                )

        pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'z_mode'}).to_csv(
            os.path.join(args.output_path, args.output_tag + '_ade_z_mode.csv')
        )
        pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'z_mode'}).to_csv(
            os.path.join(args.output_path, args.output_tag + '_fde_z_mode.csv')
        )
        pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'z_mode'}).to_csv(
            os.path.join(args.output_path, args.output_tag + '_kde_z_mode.csv')
        )

        mode_stats['Mode Z'] = z_mode_stats

        # ============================================================
        # BEST OF 20
        # ============================================================
        best_of_20_stats = create_timing_stats()
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        eval_kde_nll = np.array([])

        print("-- Evaluating Best of 20")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t, t + 10)

                predictions = timed_predict(
                    eval_stg,
                    scene,
                    timesteps,
                    ph,
                    best_of_20_stats,
                    num_samples=20,
                    min_history_timesteps=7,
                    min_future_timesteps=12,
                    z_mode=False,
                    gmm_mode=False,
                    full_dist=False
                )

                if not predictions:
                    continue

                batch_error_dict = timed_batch_statistics(
                    predictions,
                    scene,
                    max_hl,
                    ph,
                    env,
                    best_of_20_stats,
                    best_of=True,
                    prune_ph_to_future=True
                )

                eval_ade_batch_errors = np.hstack(
                    (eval_ade_batch_errors, batch_error_dict[args.node_type]['ade'])
                )
                eval_fde_batch_errors = np.hstack(
                    (eval_fde_batch_errors, batch_error_dict[args.node_type]['fde'])
                )
                eval_kde_nll = np.hstack(
                    (eval_kde_nll, batch_error_dict[args.node_type]['kde'])
                )

        pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'best_of'}).to_csv(
            os.path.join(args.output_path, args.output_tag + '_ade_best_of.csv')
        )
        pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'best_of'}).to_csv(
            os.path.join(args.output_path, args.output_tag + '_fde_best_of.csv')
        )
        pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'best_of'}).to_csv(
            os.path.join(args.output_path, args.output_tag + '_kde_best_of.csv')
        )

        mode_stats['Best of 20'] = best_of_20_stats

        # ============================================================
        # FULL
        # ============================================================
        full_stats = create_timing_stats()
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        eval_kde_nll = np.array([])

        print("-- Evaluating Full")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t, t + 10)

                predictions = timed_predict(
                    eval_stg,
                    scene,
                    timesteps,
                    ph,
                    full_stats,
                    num_samples=2000,
                    min_history_timesteps=7,
                    min_future_timesteps=12,
                    z_mode=False,
                    gmm_mode=False,
                    full_dist=False
                )

                if not predictions:
                    continue

                batch_error_dict = timed_batch_statistics(
                    predictions,
                    scene,
                    max_hl,
                    ph,
                    env,
                    full_stats,
                    prune_ph_to_future=True
                )

                eval_ade_batch_errors = np.hstack(
                    (eval_ade_batch_errors, batch_error_dict[args.node_type]['ade'])
                )
                eval_fde_batch_errors = np.hstack(
                    (eval_fde_batch_errors, batch_error_dict[args.node_type]['fde'])
                )
                eval_kde_nll = np.hstack(
                    (eval_kde_nll, batch_error_dict[args.node_type]['kde'])
                )

        pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'full'}).to_csv(
            os.path.join(args.output_path, args.output_tag + '_ade_full.csv')
        )
        pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'full'}).to_csv(
            os.path.join(args.output_path, args.output_tag + '_fde_full.csv')
        )
        pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'full'}).to_csv(
            os.path.join(args.output_path, args.output_tag + '_kde_full.csv')
        )

        mode_stats['Full'] = full_stats

    # ============================================================
    # Final timing / workload summaries
    # ============================================================
    total_program_time = time.perf_counter() - program_start
    overall_stats = combine_timing_stats(mode_stats)

    for mode_name, stats in mode_stats.items():
        print_mode_summary(mode_name, stats)

    print_mode_summary("Overall (all evaluation modes combined)", overall_stats)

    print("\n" + "=" * 68)
    print("COMPLETE PROGRAM RUNTIME SUMMARY")
    print("=" * 68)
    print(f"Dataset loader time:                   {dataset_load_time:.6f} s")
    print(f"Model load/build time:                 {model_load_time:.6f} s")
    print(f"Dataset preprocessing time:            {dataset_preprocessing_time:.6f} s")
    print(f"All inference time:                    {overall_stats['inference_time']:.6f} s")
    print(f"All metric/postprocessing time:        {overall_stats['metrics_time']:.6f} s")
    print(f"Total program wall-clock time:         {total_program_time:.6f} s")
    print("=" * 68)
