import os
import sys
import time

import numpy as np

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.arg_parser as arg_parser
from util.logger import Logger
import util.mp_util as mp_util
import util.util as util


def load_args(argv):
    args = arg_parser.ArgParser()
    args.load_args(argv[1:])

    arg_file = args.parse_string("arg_file", "")
    if arg_file != "":
        succ = args.load_file(arg_file)
        assert succ, "Failed to load args from: {:s}".format(arg_file)

    return args


def set_np_formatting():
    np.set_printoptions(edgeitems=30,
                        infstr="inf",
                        linewidth=4000,
                        nanstr="nan",
                        precision=2,
                        suppress=False,
                        threshold=10000,
                        formatter=None)
    return


def set_rand_seed(args):
    if args.has_key("rand_seed"):
        rand_seed = args.parse_int("rand_seed")
    else:
        rand_seed = np.uint64(time.time() * 256)

    Logger.print("Setting seed: {}".format(rand_seed))
    util.set_rand_seed(rand_seed)
    return int(rand_seed)


def resolve_model_file(model_folder, model_file):
    if model_file == "":
        model_file = os.path.join(model_folder, "model.pt")
    elif model_file.isdigit():
        model_file = os.path.join(
            model_folder,
            "int_models",
            "model_{:010d}.pt".format(int(model_file)),
        )
    elif not os.path.isabs(model_file) and not os.path.isfile(model_file):
        model_file = os.path.join(model_folder, model_file)

    model_file = os.path.abspath(model_file)
    assert os.path.isfile(model_file), "Model file does not exist: {:s}".format(model_file)
    return model_file


def build_viser_env(model_folder, model_file, device, rand_seed, record_rollout):
    env_file = os.path.join(model_folder, "env_config.yaml")
    engine_file = os.path.join(model_folder, "engine_config.yaml")

    assert os.path.isfile(env_file), "Environment config does not exist: {:s}".format(env_file)
    if not os.path.isfile(engine_file):
        engine_file = None

    env_config, engine_config = env_builder.load_configs(env_file, engine_file)
    record_config = env_config.setdefault("viser_record", {})
    record_config["output_dir"] = get_record_dir(model_folder, model_file)
    rollout_config = env_config.setdefault("viser_rollout", {})
    rollout_config["enabled"] = bool(record_rollout)
    rollout_config["output_file"] = get_rollout_file(model_folder, model_file)
    rollout_config["seed"] = int(rand_seed)
    rollout_config["metadata"] = {
        "model_folder": os.path.basename(os.path.normpath(model_folder)),
        "model_file": os.path.basename(model_file),
    }
    env_name = env_config["env_name"]
    Logger.print("Building Viser env for {}".format(env_name))

    import envs.viser_env as viser_env
    env = viser_env.ViserEnvBuilder(env_name=env_name,
                                    env_config=env_config,
                                    engine_config=engine_config,
                                    num_envs=1,
                                    device=device,
                                    visualize=False)
    return env


def get_record_dir(model_folder, model_file):
    """Build MimicKit/record/img/<run>_<checkpoint> for Viser output."""
    model_stem = os.path.splitext(os.path.basename(model_file))[0]
    if model_stem == "model":
        checkpoint_name = "model"
    elif model_stem.startswith("model_") and model_stem[6:].isdigit():
        checkpoint_name = str(int(model_stem[6:]))
    else:
        checkpoint_name = model_stem

    run_name = os.path.basename(os.path.normpath(model_folder))
    record_name = "{}_{}".format(run_name, checkpoint_name)
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_dir, "record", "img", record_name)


def get_rollout_file(model_folder, model_file):
    """Build MimicKit/record/data/<run>_<checkpoint>_rollout.npz."""
    record_name = os.path.basename(get_record_dir(model_folder, model_file))
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(
        project_dir, "record", "data", record_name + "_rollout.npz"
    )


def test(agent, test_episodes):
    result = agent.test_model(num_episodes=test_episodes)
    Logger.print("Mean Return: {}".format(result["mean_return"]))
    Logger.print("Mean Episode Length: {}".format(result["mean_ep_len"]))
    Logger.print("Episodes: {}".format(result["num_eps"]))
    return result


def main(argv):
    args = load_args(argv)

    model_folder = args.parse_string("model_folder", "")
    assert model_folder != "", "Must specify --model_folder"
    model_folder = os.path.abspath(model_folder)
    assert os.path.isdir(model_folder), "Model folder does not exist: {:s}".format(model_folder)

    device = args.parse_string("device", "cuda:0")
    model_file = resolve_model_file(model_folder, args.parse_string("model_file", ""))
    agent_file = os.path.join(model_folder, "agent_config.yaml")
    assert os.path.isfile(agent_file), "Agent config does not exist: {:s}".format(agent_file)

    mp_util.init(rank=0, num_procs=1, device=device, master_port=None)
    rand_seed = set_rand_seed(args)
    set_np_formatting()

    record_rollout = args.parse_bool("record", False)
    env = build_viser_env(
        model_folder, model_file, device, rand_seed, record_rollout
    )
    agent = agent_builder.build_agent(agent_file, env, device)
    agent.load(model_file)
    Logger.print("Resetting seed before test: {}".format(rand_seed))
    util.set_rand_seed(rand_seed)

    test_episodes = args.parse_int("test_episodes", np.iinfo(np.int64).max)
    if record_rollout:
        test_episodes = 1
        Logger.print("Rollout recording enabled; forcing test_episodes=1")
    try:
        test(agent, test_episodes)
    finally:
        if record_rollout:
            env.save_viser_rollout()
    return


if __name__ == "__main__":
    main(sys.argv)
