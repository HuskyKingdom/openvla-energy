#!/bin/bash

# VEL v2 (P1) · eval run: vel_gal_multiscale / 50000 step
# - Base VLA & action head & proprio projector: HF hub (training was freeze-mode,
#   so these are identical to the upstream moojink checkpoint).
# - Energy head: loaded from our training run_root_dir via --energy_ckpt.
# - energy_alpha=0.2 = α_max; line-search grid becomes (0.2, 0.1, 0.05, 0.0).

echo "Running Evaluations Automatically ------------------------------"

PRETRAINED_CKPT=moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
ENERGY_CKPT=ckpoints/energy_model--50000_checkpoint.pt
ENERGY_ALPHA=0.2
E_DECODING=True
# Path A (docs/VEL_v2_progress.md 2026-04-27): zero gripper dim of energy gradient
# in line-search. True = default, fixes the catastrophic SR collapse on
# gripper-heavy suites (object/long). Set False to reproduce the
# "w/o gripper-skip" ablation row.
ENERGY_SKIP_GRIPPER=False

# Path B (docs/VEL_v2_progress.md 2026-04-27): acceptance gate.
#   ENERGY_ACCEPT_MODE ∈ {always, monotonic, slope, both}
#     - always    : reproduce P1 baseline (no gate, naive argmin) — known broken
#     - monotonic : default. Accept only if energies strictly decrease along
#                   the α-grid AND argmin is at α_max. Rejects spurious basins.
#     - slope     : accept only if (E_BC - E_best) > tau · mean_abs(best - BC)
#     - both      : require both conditions
#   ENERGY_TAU            : slope criterion threshold (only matters in slope/both)
#   ENERGY_MONOTONIC_TOL  : tolerance for "monotonic" check (in energy units)
ENERGY_ACCEPT_MODE=monotonic
ENERGY_TAU=4.0
ENERGY_MONOTONIC_TOL=0.0
RUN_TAG=v2_p1_alpha_0.2_skipgripper

# Timing profile switch — 1 = print rolling VLA / energy / total latency stats.
# Leave at 0 for real SR runs (sync() kills GPU pipelining and inflates wall time).
export VEL_TIMING_PROFILE=0
export VEL_TIMING_LOG_EVERY=50
export VEL_TIMING_WINDOW=200

echo "Evaluating spatial ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint $PRETRAINED_CKPT \
    --energy_ckpt           $ENERGY_CKPT \
    --task_suite_name       libero_spatial \
    --unnorm_key            libero_spatial_no_noops \
    --e_decoding            $E_DECODING \
    --energy_alpha          $ENERGY_ALPHA \
    --energy_skip_gripper   $ENERGY_SKIP_GRIPPER \
    --energy_accept_mode    $ENERGY_ACCEPT_MODE \
    --energy_tau            $ENERGY_TAU \
    --energy_monotonic_tol  $ENERGY_MONOTONIC_TOL \
    --task_label            ${RUN_TAG}_spatial

echo "Evaluating object ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint $PRETRAINED_CKPT \
    --energy_ckpt           $ENERGY_CKPT \
    --task_suite_name       libero_object \
    --unnorm_key            libero_object_no_noops \
    --e_decoding            $E_DECODING \
    --energy_alpha          $ENERGY_ALPHA \
    --energy_skip_gripper   $ENERGY_SKIP_GRIPPER \
    --energy_accept_mode    $ENERGY_ACCEPT_MODE \
    --energy_tau            $ENERGY_TAU \
    --energy_monotonic_tol  $ENERGY_MONOTONIC_TOL \
    --task_label            ${RUN_TAG}_object

echo "Evaluating goal ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint $PRETRAINED_CKPT \
    --energy_ckpt           $ENERGY_CKPT \
    --task_suite_name       libero_goal \
    --unnorm_key            libero_goal_no_noops \
    --e_decoding            $E_DECODING \
    --energy_alpha          $ENERGY_ALPHA \
    --energy_skip_gripper   $ENERGY_SKIP_GRIPPER \
    --energy_accept_mode    $ENERGY_ACCEPT_MODE \
    --energy_tau            $ENERGY_TAU \
    --energy_monotonic_tol  $ENERGY_MONOTONIC_TOL \
    --task_label            ${RUN_TAG}_goal

echo "Evaluating long ------------------------------"
echo N | python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint $PRETRAINED_CKPT \
    --energy_ckpt           $ENERGY_CKPT \
    --task_suite_name       libero_10 \
    --unnorm_key            libero_10_no_noops \
    --e_decoding            $E_DECODING \
    --energy_alpha          $ENERGY_ALPHA \
    --energy_skip_gripper   $ENERGY_SKIP_GRIPPER \
    --energy_accept_mode    $ENERGY_ACCEPT_MODE \
    --energy_tau            $ENERGY_TAU \
    --energy_monotonic_tol  $ENERGY_MONOTONIC_TOL \
    --task_label            ${RUN_TAG}_long
