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


def build_timelapse_viewer(rollout_file):
    rollout_path = ViserTimelapseBase.resolve_rollout_file(rollout_file)
    metadata = load_rollout_metadata(rollout_path)
    env_name = metadata.get("env_name")
    viewer_type = TIMELAPSE_ENVS.get(env_name)
    if viewer_type is None:
        supported = ", ".join(sorted(TIMELAPSE_ENVS.keys()))
        raise ValueError(
            "Unsupported rollout metadata.env_name={!r}; supported: {}".format(
                env_name, supported
            )
        )
    print("Selected timelapse environment: {}".format(env_name))
    return viewer_type(rollout_path)


def main():
    parser = argparse.ArgumentParser(
        description="Open a rollout with its task-specific Viser timelapse viewer."
    )
    parser.add_argument(
        "--rollout_file",
        required=True,
        help="Path to a rollout NPZ file",
    )
    args = parser.parse_args()
    viewer = build_timelapse_viewer(args.rollout_file)
    viewer.run_forever()


if __name__ == "__main__":
    main()
