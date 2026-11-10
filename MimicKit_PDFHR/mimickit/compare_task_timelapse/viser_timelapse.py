import argparse

try:
    from .viser_dodgeball_timelapse import ViserDodgeballTimelapse
    from .viser_dribble_timelapse import ViserDribbleTimelapse
    from .viser_location_timelapse import ViserLocationTimelapse
    from .viser_steering_timelapse import ViserSteeringTimelapse
    from .viser_timelapse_base import (
        ViserTimelapseBase,
        load_rollout_metadata,
    )
    from .viser_traj_timelapse import ViserTrajTimelapse
except ImportError:
    from viser_dodgeball_timelapse import ViserDodgeballTimelapse
    from viser_dribble_timelapse import ViserDribbleTimelapse
    from viser_location_timelapse import ViserLocationTimelapse
    from viser_steering_timelapse import ViserSteeringTimelapse
    from viser_timelapse_base import ViserTimelapseBase, load_rollout_metadata
    from viser_traj_timelapse import ViserTrajTimelapse


TIMELAPSE_ENVS = {
    "task_location": ViserLocationTimelapse,
    "task_steering": ViserSteeringTimelapse,
    "task_dodgeball": ViserDodgeballTimelapse,
    "task_dribble": ViserDribbleTimelapse,
    "task_traj": ViserTrajTimelapse,
}


def build_timelapse_viewer(baseline_file, ours_file):
    baseline_path = ViserTimelapseBase.resolve_rollout_file(baseline_file)
    ours_path = ViserTimelapseBase.resolve_rollout_file(ours_file)
    baseline_metadata = load_rollout_metadata(baseline_path)
    ours_metadata = load_rollout_metadata(ours_path)
    baseline_seed = baseline_metadata.get("seed")
    ours_seed = ours_metadata.get("seed")
    assert baseline_seed == ours_seed, (
        "Baseline seed {} does not match ours seed {}".format(
            baseline_seed, ours_seed
        )
    )
    env_name = baseline_metadata.get("env_name")
    assert env_name == ours_metadata.get("env_name"), (
        "Baseline task {!r} does not match ours task {!r}".format(
            env_name, ours_metadata.get("env_name")
        )
    )
    viewer_type = TIMELAPSE_ENVS.get(env_name)
    if viewer_type is None:
        supported = ", ".join(sorted(TIMELAPSE_ENVS.keys()))
        raise ValueError(
            "Unsupported rollout metadata.env_name={!r}; supported: {}".format(
                env_name, supported
            )
        )
    print("Selected compare timelapse environment: {}".format(env_name))
    return viewer_type(baseline_path, ours_path)


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline and ours task rollouts in Viser."
    )
    parser.add_argument(
        "--baseline_file",
        required=True,
        help="Path to the baseline rollout NPZ file",
    )
    parser.add_argument(
        "--ours_file",
        required=True,
        help="Path to our rollout NPZ file",
    )
    args = parser.parse_args()
    viewer = build_timelapse_viewer(args.baseline_file, args.ours_file)
    viewer.run_forever()


if __name__ == "__main__":
    main()
