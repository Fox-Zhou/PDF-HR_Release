import copy
import importlib

from envs.viser_base_env import ViserBaseEnv


_VISER_MIXIN_REGISTRY = {
    "deepmimic": ("envs.viser_tracking_env", "ViserTrackingEnv"),
    "amp": ("envs.viser_tracking_env", "ViserTrackingEnv"),
    "add": ("envs.viser_tracking_env", "ViserTrackingEnv"),
    "smp": ("envs.viser_tracking_env", "ViserTrackingEnv"),
    "task_location": ("envs.viser_location_env", "ViserLocationEnv"),
    "task_steering": ("envs.viser_steering_env", "ViserSteeringEnv"),
    "task_dodgeball": ("envs.viser_dodgeball_env", "ViserDodgeballEnv"),
    "task_traj": ("envs.viser_traj_env", "ViserTrajEnv"),
    "task_dribble": ("envs.viser_dribble_env", "ViserDribbleEnv"),
}


def ViserEnvBuilder(env_name, env_config, engine_config, num_envs, device,
                    visualize=False):
    """Build a simulation env with its task-specific Viser mixin."""
    base_env_name = _get_base_env_name(env_name)
    mixin_spec = _VISER_MIXIN_REGISTRY.get(base_env_name)
    if mixin_spec is None:
        raise ValueError("Viser is not implemented for env: {}".format(base_env_name))

    mixin_module_name, mixin_class_name = mixin_spec
    try:
        mixin_module = importlib.import_module(mixin_module_name)
    except ModuleNotFoundError as err:
        if err.name == mixin_module_name:
            raise ModuleNotFoundError(
                "Missing Viser mixin module for {}: {}".format(
                    base_env_name, mixin_module_name
                )
            ) from err
        raise

    mixin_cls = getattr(mixin_module, mixin_class_name)
    base_env_cls = _get_env_class(base_env_name)
    dynamic_cls = type(
        "Viser{}".format(base_env_cls.__name__),
        (mixin_cls, base_env_cls),
        {},
    )

    base_env_config = copy.deepcopy(env_config)
    base_env_config["env_name"] = base_env_name
    is_tracking_env = base_env_name in {"deepmimic", "amp", "add", "smp"}
    if is_tracking_env:
        base_env_config["visualize_ref_char"] = True

    env_args = dict(env_config=base_env_config,
                    engine_config=engine_config,
                    num_envs=num_envs,
                    device=device,
                    visualize=visualize)
    env_args["record_video"] = False

    return dynamic_cls(**env_args)


def _get_env_class(env_name):
    if env_name == "deepmimic":
        from envs.deepmimic_env import DeepMimicEnv
        return DeepMimicEnv
    elif env_name == "amp":
        from envs.amp_env import AMPEnv
        return AMPEnv
    elif env_name == "add":
        from envs.add_env import ADDEnv
        return ADDEnv
    elif env_name == "smp":
        from envs.smp_env import SMPEnv
        return SMPEnv
    elif env_name == "task_location":
        from envs.task_location_env import TaskLocationEnv
        return TaskLocationEnv
    elif env_name == "task_steering":
        from envs.task_steering_env import TaskSteeringEnv
        return TaskSteeringEnv
    elif env_name == "task_dodgeball":
        from envs.task_dodgeball_env import TaskDodgeballEnv
        return TaskDodgeballEnv
    elif env_name == "task_dribble":
        from envs.task_dribble_env import TaskDribbleEnv
        return TaskDribbleEnv
    elif env_name == "task_traj":
        from envs.task_traj_env import TaskTrajEnv
        return TaskTrajEnv
    raise ValueError("Viser is not implemented for env: {}".format(env_name))


def _get_base_env_name(env_name):
    suffix = "_viser"
    if env_name.endswith(suffix):
        return env_name[:-len(suffix)]
    return env_name


class ViserMixinTemplate(ViserBaseEnv):
    """Minimal template for a task visualizer.

    New task mixins should inherit directly from :class:`ViserBaseEnv` and
    implement these two hooks. This template remains as executable guidance.
    """

    def _build_task_viser(self):
        raise NotImplementedError

    def _update_task_viser(self):
        raise NotImplementedError
