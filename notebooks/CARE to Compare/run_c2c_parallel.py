"""Parallel driver for the CARE-to-Compare paper evaluation.

Trains the 95 events x n_runs models for a CAREScore evaluation in parallel.
Processes and writes one result CSV per (event, run) into results/run_<run>/events/.
Afterwards the per-run shards are concatenated into results/run_<run>/results_all.csv so the
remaining notebook cells work unchanged.

Run from a terminal with e.g. 32 workers:

    python run_c2c_parallel.py --workers 32 --n-runs 10

Re-running skips every (event, run) whose result is already stored (shard or results_all.csv),
so the run is resumable. After results_all.csv has been written successfully for a run, its
per-event shards are deleted.

The seed for each (event, run) is derived deterministically from --base-seed, so running with
the same base seed and the same number of runs reproduces the results exactly, independent of
the number of workers or how runs are grouped into work units.

For a multi-node setup, run the same command per node with
--num-shards N --shard <index>, then once with --gather-only.
"""

import os

# Must be set before tensorflow is imported in this process and in the spawned workers.
# One TF thread per process; parallelism comes from running many processes.
for _key, _value in {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "-1",
    "TF_CPP_MIN_LOG_LEVEL": "3",
}.items():
    os.environ[_key] = _value

import argparse
import multiprocessing as mp
import time
from pathlib import Path

WIND_FARMS = ("A", "B", "C")
IGNORE_NORMAL = {"A": True, "B": False, "C": False}
DEFAULT_BASE_SEED = 42
SEED_STRIDE = 1000  # one seed block per event, large enough for any number of runs


def _seed_for(base_seed: int, event_id: int, run: int) -> int:
    """Deterministic seed for one (event, run) pair."""
    return base_seed + SEED_STRIDE * event_id + run


_STATE = {}


def _event_rows(data_dir: Path):
    """Return a list of (event_id, wind_farm, dataset_file_size) tuples."""
    import pandas as pd

    rows = []
    for wf in WIND_FARMS:
        info = pd.read_csv(data_dir / f"Wind Farm {wf}" / "event_info.csv", sep=";")
        for event_id in info["event_id"].astype(int):
            path = data_dir / f"Wind Farm {wf}" / "datasets" / f"{event_id}.csv"
            rows.append((event_id, wf, path.stat().st_size))
    return rows


def _init_worker(data_dir, config_dir, results_dir, base_seed):
    """Initialise a worker: thread limits, seeds, configs and the dataset loader."""
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    # Seed the process at startup; every fit re-seeds deterministically per (event, run).
    tf.keras.utils.set_random_seed(base_seed)

    from energy_fault_detector.config import Config
    from energy_fault_detector.evaluation import Care2CompareDataset
    from energy_fault_detector.fault_detector import FaultDetector

    c2c = Care2CompareDataset(path=data_dir)
    configs = {wf: Config(str(Path(config_dir) / f"windfarm_{wf}.yaml")) for wf in WIND_FARMS}
    for wf in WIND_FARMS:
        c2c.update_c2c_config(configs[wf], wf)

    _STATE.update(
        c2c=c2c,
        configs=configs,
        FaultDetector=FaultDetector,
        results_dir=Path(results_dir),
        base_seed=base_seed,
        cache={},
        completed={},
    )


def _prepare(event_id, wf):
    """Load and format the train/test data of one event (cached per worker)."""
    cached = _STATE["cache"].get(event_id)
    if cached is not None:
        return cached

    c2c = _STATE["c2c"]
    x_train, x_test = c2c.load_event_dataset(event_id=event_id, index_column="time_stamp")

    y_train = x_train["status_type_id"] == 0
    y_test = x_test["status_type_id"] == 0
    if wf == "C":
        # Apply status filtering based on wind speed and active power for Wind Farm C, as described in the paper.
        in_operation = (
            x_train["Wind speed 1 (avg)"].between(3, 25)
            & (x_train["Active power HV grid (avg)"] >= 0.01)
        )
        y_train = y_train & in_operation

    x_train = x_train.drop(["id", "asset_id", "status_type_id"], axis=1, errors="ignore")
    x_test = x_test.drop(["id", "asset_id", "status_type_id"], axis=1, errors="ignore")

    info = c2c.event_info_all[c2c.event_info_all["event_id"] == event_id].iloc[0]
    data = (x_train, y_train, x_test, y_test, info)
    _STATE["cache"] = {event_id: data}  # keep only the current event in memory
    return data


def _out_path(run, event_id):
    return _STATE["results_dir"] / f"run_{run}" / "events" / f"event_{event_id}.csv"


def _completed_event_ids(run):
    """Event IDs already present in results_all.csv (empty if the file does not exist)."""
    if run not in _STATE["completed"]:
        import pandas as pd

        target = _STATE["results_dir"] / f"run_{run}" / "results_all.csv"
        if target.exists():
            ids = set(pd.read_csv(target, usecols=["event_id"])["event_id"].astype(int))
        else:
            ids = set()
        _STATE["completed"][run] = ids
    return _STATE["completed"][run]


def _is_done(run, event_id):
    return _out_path(run, event_id).exists() or event_id in _completed_event_ids(run)


def _run_unit(task):
    """Train and evaluate a chunk of runs for a single event."""
    import pandas as pd
    import tensorflow as tf
    from energy_fault_detector.evaluation import CAREScore

    event_id, wf, runs = task
    todo = [run for run in runs if not _is_done(run, event_id)]
    if not todo:
        return event_id, 0, 0.0

    x_train, y_train, x_test, y_test, info = _prepare(event_id, wf)
    started = time.time()

    for run in todo:
        seed = _seed_for(_STATE["base_seed"], event_id, run)
        tf.keras.utils.set_random_seed(seed)

        model = _STATE["FaultDetector"](_STATE["configs"][wf])
        model.fit(sensor_data=x_train, normal_index=y_train, save_models=False)
        prediction = model.predict(x_test)

        score = CAREScore(
            coverage_beta=0.5,
            reliability_beta=0.5,
            anomaly_detection_method="criticality",
            ws_start_of_descend=(1, 2)
        )
        evaluation = score.evaluate_event(
            event_id=event_id,
            event_start=info["event_start"],
            event_end=info["event_end"],
            event_label=info["event_label"],
            normal_index=y_test,
            predicted_anomalies=prediction.predicted_anomalies,
            ignore_normal_index=IGNORE_NORMAL[wf],
        )
        evaluation["run"] = run
        evaluation["seed"] = seed

        out = _out_path(run, event_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        pd.DataFrame([evaluation]).to_csv(tmp, index=False)
        os.replace(tmp, out)
        del model, prediction

    return event_id, len(todo), round(time.time() - started, 1)


def _gather(results_dir: Path, n_runs: int):
    """Merge the per-event shards of each run into results_all.csv and delete the shards."""
    import pandas as pd

    for run in range(1, n_runs + 1):
        run_dir = results_dir / f"run_{run}"
        events_dir = run_dir / "events"
        target = run_dir / "results_all.csv"
        files = sorted(events_dir.glob("event_*.csv")) if events_dir.exists() else []
        if not files:
            continue

        frames = [pd.read_csv(target)] if target.exists() else []
        frames.extend(pd.read_csv(f) for f in files)
        df = pd.concat(frames, ignore_index=True).drop_duplicates("event_id", keep="last")

        tmp = target.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, target)  # only delete shards once results_all.csv was written

        for f in files:
            f.unlink()
        try:
            events_dir.rmdir()
        except OSError:
            pass
        print(f"run {run}: {len(df)} events -> results_all.csv ({len(files)} shards removed)")


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=here.parents[1] / "CARE_To_Compare")
    parser.add_argument("--config-dir", type=Path, default=here / "c2c_configs")
    parser.add_argument("--results-dir", type=Path, default=here / "results")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument(
        "--base-seed", type=int, default=DEFAULT_BASE_SEED,
        help="Base seed; each (event, run) gets base_seed + 1000*event_id + run.",
    )
    parser.add_argument(
        "--runs-per-task", type=int, default=None,
        help="Runs of one event per work unit. Default: auto (~2 units per worker).",
    )
    parser.add_argument(
        "--recycle", type=int, default=50,
        help="Restart a worker after this many work units (0 = never).",
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--gather-only", action="store_true")
    args = parser.parse_args()

    if args.gather_only:
        _gather(args.results_dir, args.n_runs)
        return

    rows = _event_rows(args.data_dir)
    total_runs = len(rows) * args.n_runs
    runs_per_task = args.runs_per_task or max(
        1, min(args.n_runs, -(-total_runs // (2 * args.workers)))
    )

    tasks = []
    sizes = {}
    for event_id, wf, size in rows:
        sizes[event_id] = size
        for start in range(1, args.n_runs + 1, runs_per_task):
            runs = list(range(start, min(start + runs_per_task, args.n_runs + 1)))
            tasks.append((event_id, wf, runs))

    # Longest-processing-time-first scheduling: big datasets start first.
    tasks.sort(key=lambda task: sizes[task[0]] * len(task[2]), reverse=True)
    if args.num_shards > 1:
        tasks = tasks[args.shard::args.num_shards]

    print(
        f"{len(tasks)} work units, {args.workers} workers, "
        f"{runs_per_task} run(s)/unit, {args.n_runs} runs, {len(rows)} events, "
        f"base seed {args.base_seed}",
        flush=True,
    )

    started = time.time()
    context = mp.get_context("spawn")
    with context.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(str(args.data_dir), str(args.config_dir), str(args.results_dir), args.base_seed),
        maxtasksperchild=args.recycle or None,
    ) as pool:
        for index, (event_id, n_done, seconds) in enumerate(
            pool.imap_unordered(_run_unit, tasks, chunksize=1), 1
        ):
            print(
                f"[{index}/{len(tasks)}] event {event_id}: "
                f"{n_done} run(s) in {seconds}s",
                flush=True,
            )

    print(f"training done in {(time.time() - started) / 60:.1f} min", flush=True)

    if args.num_shards == 1:
        _gather(args.results_dir, args.n_runs)


if __name__ == "__main__":
    main()
