import numpy as np
import torch

import engines.engine as engine
import envs.smp_env as smp_env
import envs.base_env as base_env
import util.torch_util as torch_util


class _TrajGenerator:
    """
    Lightweight 2D piecewise-linear trajectory generator.

    For each env, it stores a trajectory:
        self._verts: [num_envs, num_verts, 3]

    calc_pos(env_ids, times) returns linearly interpolated target positions.
    """

    def __init__(
        self,
        num_envs,
        episode_dur,
        num_verts,
        device,
        speed_min,
        speed_max,
        accel_max,
        sharp_turn_prob,
        sharp_turn_angle,
        dtheta_max=2.0,
    ):
        self._num_envs = int(num_envs)
        self._episode_dur = float(episode_dur)
        self._num_verts = int(num_verts)
        self._device = device

        self._speed_min = float(speed_min)
        self._speed_max = float(speed_max)
        self._accel_max = float(accel_max)
        self._sharp_turn_prob = float(sharp_turn_prob)
        self._sharp_turn_angle = float(sharp_turn_angle)
        self._dtheta_max = float(dtheta_max)

        self._dt = self._episode_dur / float(max(self._num_verts - 1, 1))
        self._verts = torch.zeros(
            [self._num_envs, self._num_verts, 3],
            device=self._device,
            dtype=torch.float32,
        )
        return

    def reset(self, env_ids, root_pos):
        if env_ids is None or len(env_ids) == 0:
            return

        env_ids = env_ids.to(device=self._device, dtype=torch.long)
        root_pos = root_pos.to(device=self._device, dtype=torch.float32)

        n = int(env_ids.shape[0])
        num_seg = max(self._num_verts - 1, 1)

        verts = torch.zeros(
            [n, self._num_verts, 3],
            device=self._device,
            dtype=torch.float32,
        )
        verts[:, 0, 0:2] = root_pos[:, 0:2]
        verts[:, 0, 2] = 0.0

        theta = 2.0 * np.pi * torch.rand(
            [n],
            device=self._device,
            dtype=torch.float32,
        )

        speed = self._speed_min + (self._speed_max - self._speed_min) * torch.rand(
            [n],
            device=self._device,
            dtype=torch.float32,
        )

        for i in range(num_seg):
            if self._sharp_turn_prob > 0.0 and self._sharp_turn_angle > 0.0:
                sharp_mask = torch.rand([n], device=self._device) < self._sharp_turn_prob

                sharp_sign = torch.where(
                    torch.rand([n], device=self._device) < 0.5,
                    -torch.ones([n], device=self._device),
                    torch.ones([n], device=self._device),
                )
                sharp_delta = sharp_sign * self._sharp_turn_angle
            else:
                sharp_mask = torch.zeros([n], device=self._device, dtype=torch.bool)
                sharp_delta = torch.zeros([n], device=self._device, dtype=torch.float32)

            smooth_delta = (
                2.0 * torch.rand([n], device=self._device, dtype=torch.float32) - 1.0
            ) * self._dtheta_max * self._dt

            dtheta = torch.where(sharp_mask, sharp_delta, smooth_delta)
            theta = theta + dtheta

            if self._accel_max > 0.0:
                dv = (
                    2.0 * torch.rand([n], device=self._device, dtype=torch.float32) - 1.0
                ) * self._accel_max * self._dt
                speed = torch.clamp(speed + dv, min=self._speed_min, max=self._speed_max)

            vel_xy = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
            vel_xy = vel_xy * speed.unsqueeze(-1)

            verts[:, i + 1, 0:2] = verts[:, i, 0:2] + vel_xy * self._dt
            verts[:, i + 1, 2] = 0.0

        self._verts[env_ids] = verts
        return

    def calc_pos(self, env_ids, times):
        env_ids = env_ids.to(device=self._device, dtype=torch.long)
        times = times.to(device=self._device, dtype=torch.float32)

        flat_env_ids = env_ids.reshape(-1)
        flat_times = times.reshape(-1)
        flat_times = torch.clamp(flat_times, min=0.0, max=self._episode_dur)

        if self._num_verts <= 1:
            out = self._verts[flat_env_ids, 0]
            return out.reshape(times.shape + (3,))

        u = flat_times / max(self._dt, 1e-6)

        idx0 = torch.floor(u).to(dtype=torch.long)
        idx0 = torch.clamp(idx0, min=0, max=self._num_verts - 1)
        idx1 = torch.clamp(idx0 + 1, min=0, max=self._num_verts - 1)

        alpha = (u - idx0.to(dtype=torch.float32)).unsqueeze(-1)

        p0 = self._verts[flat_env_ids, idx0]
        p1 = self._verts[flat_env_ids, idx1]
        out = (1.0 - alpha) * p0 + alpha * p1

        return out.reshape(times.shape + (3,))

    def get_traj_verts(self, env_id):
        return self._verts[int(env_id)]


class TaskTrajEnv(smp_env.SMPEnv):
    """
    Trajectory-following version of TaskLocationEnv.

    Difference from TaskLocationEnv:
      - TaskLocationEnv has one target location, task obs dim = 2.
      - TaskTrajEnv has a sequence of future target locations, task obs dim = 2 * K.

    The task obs is appended to the end of policy obs:
        obs = [base_obs, task_obs]

    The trajectory samples are consumed through the regular policy
    observation used by MimicKit's native agents.
    """

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        # ------------------------------------------------------------
        # Trajectory generation config
        # ------------------------------------------------------------
        self._enable_task_obs = bool(env_config.get("enable_task_obs", True))

        self._num_traj_samples = int(env_config.get("num_traj_samples", 10))
        self._traj_sample_timestep = float(env_config.get("traj_sample_timestep", 0.5))
        self._traj_num_verts = int(env_config.get("traj_num_verts", 101))

        self._speed_min = float(env_config.get("speed_min", 0.8))
        self._speed_max = float(env_config.get("speed_max", 1.6))
        self._accel_max = float(env_config.get("accel_max", 0.5))
        self._sharp_turn_prob = float(env_config.get("sharp_turn_prob", 0.2))
        self._sharp_turn_angle = float(env_config.get("sharp_turn_angle", 1.57))
        self._dtheta_max = float(env_config.get("dtheta_max", 2.0))

        # ------------------------------------------------------------
        # Reward config
        # ------------------------------------------------------------
        self._dist_threshold = float(env_config.get("dist_threshold", 0.5))

        self._pos_err_scale = float(env_config.get("pos_err_scale", env_config.get("traj_reward_pos_scale", 2.0)))
        self._vel_err_scale = float(env_config.get("vel_err_scale", 4.0))

        self._pos_reward_w = float(env_config.get("pos_reward_w", 0.6))
        self._vel_reward_w = float(env_config.get("vel_reward_w", 0.3))
        self._face_reward_w = float(env_config.get("face_reward_w", 0.1))

        # Harder tracking terms. Unlike the previous reward, velocity is
        # matched as a full 2D vector, so moving sideways or overshooting no
        # longer receives a high velocity reward. The overspeed term is an
        # optional additional regularizer for cutting corners too fast.
        self._overspeed_reward_w = float(env_config.get("overspeed_reward_w", 0.0))
        self._overspeed_err_scale = float(env_config.get("overspeed_err_scale", 2.0))
        self._overspeed_slack = float(env_config.get("overspeed_slack", 0.25))

        self._reward_lookahead_time = float(
            env_config.get("reward_lookahead_time", self._traj_sample_timestep)
        )

        # Optional failure if the character drifts too far from the current target.
        self._enable_traj_fail = bool(env_config.get("enable_traj_fail", True))
        self._traj_fail_dist = float(env_config.get("traj_fail_dist", env_config.get("fail_dist", 4.0)))

        # ------------------------------------------------------------
        # Visualization config
        # ------------------------------------------------------------
        self._traj_marker_asset_file = str(
            env_config.get("traj_marker_asset_file", env_config.get("marker_asset_file", "data/assets/objects/location_marker.xml"))
        )
        self._traj_marker_height = float(env_config.get("traj_marker_height", 0.02))
        self._traj_marker_color = env_config.get("traj_marker_color", [0.8, 0.0, 0.0])
        self._traj_line_color = np.array([[0.0, 0.0, 1.0, 0.5]], dtype=np.float32)

        assert self._num_traj_samples > 0
        assert self._traj_sample_timestep > 0.0
        assert self._traj_num_verts >= 2
        assert self._speed_min >= 0.0
        assert self._speed_max >= self._speed_min
        assert self._accel_max >= 0.0

        super().__init__(
            env_config=env_config,
            engine_config=engine_config,
            num_envs=num_envs,
            device=device,
            visualize=visualize,
            record_video=record_video,
        )
        return

    # ============================================================
    # Build env / markers
    # ============================================================
    def _build_envs(self, config, num_envs):
        if self._visualize:
            self._marker_ids = []

        super()._build_envs(config, num_envs)
        return

    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)

        if self._visualize:
            for marker_idx in range(self._num_traj_samples):
                marker_id = self._build_marker(env_id, marker_idx)

                if env_id == 0:
                    self._marker_ids.append(marker_id)
                else:
                    assert self._marker_ids[marker_idx] == marker_id

        return

    def _build_marker(self, env_id, marker_idx):
        marker_id = self._engine.create_obj(
            env_id=env_id,
            obj_type=engine.ObjType.rigid,
            asset_file=self._traj_marker_asset_file,
            name="traj_marker_{:02d}".format(marker_idx),
            is_visual=True,
            fix_root=True,
            color=self._traj_marker_color,
        )
        return marker_id

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)

        num_envs = self.get_num_envs()
        self._prev_root_pos = torch.zeros(
            [num_envs, 3],
            device=self._device,
            dtype=torch.float32,
        )

        episode_dur = float(
            getattr(
                self,
                "_episode_length",
                config.get("episode_length", config.get("episodeLength", 20.0)),
            )
        )

        self._traj_gen = _TrajGenerator(
            num_envs=num_envs,
            episode_dur=episode_dur,
            num_verts=self._traj_num_verts,
            device=self._device,
            speed_min=self._speed_min,
            speed_max=self._speed_max,
            accel_max=self._accel_max,
            sharp_turn_prob=self._sharp_turn_prob,
            sharp_turn_angle=self._sharp_turn_angle,
            dtheta_max=self._dtheta_max,
        )

        env_ids = torch.arange(num_envs, device=self._device, dtype=torch.long)
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        self._traj_gen.reset(env_ids, root_pos)
        return

    def _get_marker_id(self, marker_idx):
        return self._marker_ids[marker_idx]

    # ============================================================
    # Step / reset / task update
    # ============================================================
    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        self._record_char_root()
        return

    def _record_char_root(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        self._prev_root_pos[:] = root_pos
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)

        if len(env_ids) > 0:
            self._reset_task(env_ids)
            self._record_char_root()
        return

    def _reset_task(self, env_ids):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        self._traj_gen.reset(env_ids, root_pos[env_ids])

        if self._visualize:
            self._update_marker(env_ids)
        return

    def _get_task_time(self, env_ids):
        if hasattr(self, "_time_buf"):
            return self._time_buf[env_ids]

        return torch.zeros(
            [env_ids.shape[0]],
            device=self._device,
            dtype=torch.float32,
        )

    def _fetch_traj_samples(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(
                self.get_num_envs(),
                device=self._device,
                dtype=torch.long,
            )

        timestep_beg = self._get_task_time(env_ids)

        sample_steps = torch.arange(
            self._num_traj_samples,
            device=self._device,
            dtype=torch.float32,
        )
        sample_steps = sample_steps * self._traj_sample_timestep

        traj_timesteps = timestep_beg.unsqueeze(-1) + sample_steps
        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)

        traj_samples_flat = self._traj_gen.calc_pos(
            env_ids_tiled.reshape(-1),
            traj_timesteps.reshape(-1),
        )

        traj_samples = traj_samples_flat.reshape(
            env_ids.shape[0],
            self._num_traj_samples,
            3,
        )

        return traj_samples

    def _fetch_curr_traj_target(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(
                self.get_num_envs(),
                device=self._device,
                dtype=torch.long,
            )

        return self._traj_gen.calc_pos(env_ids, self._get_task_time(env_ids))

    def _fetch_next_traj_target(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(
                self.get_num_envs(),
                device=self._device,
                dtype=torch.long,
            )

        t = self._get_task_time(env_ids) + self._reward_lookahead_time
        return self._traj_gen.calc_pos(env_ids, t)

    def _update_marker(self, env_ids):
        if not self._visualize:
            return

        if not hasattr(self, "_marker_ids"):
            return

        traj_samples = self._fetch_traj_samples(env_ids).clone()
        traj_samples[..., 2] = self._traj_marker_height

        rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device, dtype=torch.float32)

        for marker_idx, marker_id in enumerate(self._marker_ids):
            self._engine.set_root_pos(env_ids, marker_id, traj_samples[:, marker_idx, :])
            self._engine.set_root_rot(env_ids, marker_id, rot)
            self._engine.set_root_vel(env_ids, marker_id, 0.0)
            self._engine.set_root_ang_vel(env_ids, marker_id, 0.0)

        return

    def _update_misc(self):
        super()._update_misc()

        if self._visualize:
            env_ids = torch.arange(
                self.get_num_envs(),
                device=self._device,
                dtype=torch.long,
            )
            self._update_marker(env_ids)

        return

    # ============================================================
    # Task obs module
    # ============================================================
    def _get_task_obs_dim(self):
        if not self._enable_task_obs:
            return 0
        return 2 * int(self._num_traj_samples)

    def _compute_task_obs(self, env_ids=None):
        if not self._enable_task_obs:
            num_envs = self.get_num_envs() if env_ids is None else int(env_ids.shape[0])
            return torch.zeros(
                [num_envs, 0],
                device=self._device,
                dtype=torch.float32,
            )

        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)

        if env_ids is not None:
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]

        traj_samples = self._fetch_traj_samples(env_ids)

        task_obs = compute_traj_observations(
            root_pos=root_pos,
            root_rot=root_rot,
            traj_samples=traj_samples,
        )

        assert task_obs.shape[-1] == self._get_task_obs_dim(), (
            f"task_obs dim mismatch: got {task_obs.shape[-1]}, "
            f"expected {self._get_task_obs_dim()}"
        )

        return task_obs

    # ============================================================
    # Observation / reward / done
    # ============================================================
    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)

        task_obs = self._compute_task_obs(env_ids)
        if task_obs.shape[-1] > 0:
            obs = torch.cat([obs, task_obs], dim=-1)

        return obs

    def _update_reward(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)

        env_ids = torch.arange(
            self.get_num_envs(),
            device=self._device,
            dtype=torch.long,
        )

        tar_pos = self._fetch_curr_traj_target(env_ids)
        tar_pos_next = self._fetch_next_traj_target(env_ids)

        self._reward_buf[:] = compute_traj_reward(
            root_pos=root_pos,
            prev_root_pos=self._prev_root_pos,
            root_rot=root_rot,
            tar_pos=tar_pos,
            tar_pos_next=tar_pos_next,
            dt=self._engine.get_timestep(),
            reward_lookahead_time=self._reward_lookahead_time,
            dist_threshold=self._dist_threshold,
            pos_err_scale=self._pos_err_scale,
            vel_err_scale=self._vel_err_scale,
            pos_reward_w=self._pos_reward_w,
            vel_reward_w=self._vel_reward_w,
            face_reward_w=self._face_reward_w,
            overspeed_reward_w=self._overspeed_reward_w,
            overspeed_err_scale=self._overspeed_err_scale,
            overspeed_slack=self._overspeed_slack,
        )
        return

    def _update_done(self):
        super()._update_done()

        if not (getattr(self, "_enable_early_termination", False) and self._enable_traj_fail):
            return

        if self._traj_fail_dist <= 0.0:
            return

        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)

        env_ids = torch.arange(
            self.get_num_envs(),
            device=self._device,
            dtype=torch.long,
        )
        tar_pos = self._fetch_curr_traj_target(env_ids)

        dist = torch.linalg.norm(tar_pos[..., 0:2] - root_pos[..., 0:2], dim=-1)
        fail = dist > self._traj_fail_dist

        # Avoid terminating immediately at reset.
        task_time = self._get_task_time(env_ids)
        fail = torch.logical_and(fail, task_time > 0.5)

        self._done_buf[fail] = base_env.DoneFlags.FAIL.value
        return

    # ============================================================
    # Rendering
    # ============================================================
    def _render_scene(self):
        super()._render_scene()
        self._render_traj_lines()
        return

    def _render_traj_lines(self):
        if not hasattr(self._engine, "draw_lines"):
            return

        line_width = 2.0
        col = self._traj_line_color

        num_envs = self.get_num_envs()
        for i in range(num_envs):
            verts = self._traj_gen.get_traj_verts(i).detach().cpu().numpy()
            verts = verts.copy()
            verts[..., 2] = self._traj_marker_height

            if verts.shape[0] < 2:
                continue

            starts = verts[:-1]
            ends = verts[1:]
            curr_cols = np.broadcast_to(col, [starts.shape[0], col.shape[-1]])

            self._engine.draw_lines(i, starts, ends, curr_cols, line_width)

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_traj_observations(root_pos, root_rot, traj_samples):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    heading_rot = torch_util.calc_heading_quat_inv(root_rot)

    heading_rot_exp = torch.broadcast_to(
        heading_rot.unsqueeze(1),
        (traj_samples.shape[0], traj_samples.shape[1], 4),
    )

    root_pos_exp = torch.broadcast_to(
        root_pos.unsqueeze(1),
        (traj_samples.shape[0], traj_samples.shape[1], 3),
    )

    local_traj_samples = torch_util.quat_rotate(
        heading_rot_exp.reshape(-1, 4),
        traj_samples.reshape(-1, 3) - root_pos_exp.reshape(-1, 3),
    )

    obs = local_traj_samples[..., 0:2].reshape(root_pos.shape[0], -1)
    return obs


@torch.jit.script
def compute_traj_reward(
    root_pos,
    prev_root_pos,
    root_rot,
    tar_pos,
    tar_pos_next,
    dt,
    reward_lookahead_time,
    dist_threshold,
    pos_err_scale,
    vel_err_scale,
    pos_reward_w,
    vel_reward_w,
    face_reward_w,
    overspeed_reward_w,
    overspeed_err_scale,
    overspeed_slack,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, float, float, float) -> Tensor

    # ------------------------------------------------------------
    # 1. Strict position tracking to the time-indexed trajectory point.
    # ------------------------------------------------------------
    pos_diff = tar_pos - root_pos
    pos_err = pos_diff[..., 0] * pos_diff[..., 0] + pos_diff[..., 1] * pos_diff[..., 1]
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    # ------------------------------------------------------------
    # 2. Target tangent direction and full 2D target velocity.
    #    Previous version only rewarded projected speed along tar_dir and
    #    clamped missing speed. That made sideways motion / corner cutting too
    #    easy. Here we match the full 2D velocity vector.
    # ------------------------------------------------------------
    tar_delta = tar_pos_next - tar_pos
    tar_vel = tar_delta[..., 0:2] / max(reward_lookahead_time, 1e-6)
    tar_speed = torch.linalg.norm(tar_vel, dim=-1)

    tar_dir = tar_delta[..., 0:2]
    fallback_dir = pos_diff[..., 0:2]
    tar_dir_norm = torch.linalg.norm(tar_dir, dim=-1, keepdim=True)
    fallback_norm = torch.linalg.norm(fallback_dir, dim=-1, keepdim=True)

    tar_dir = torch.where(
        tar_dir_norm > 1e-6,
        tar_dir / torch.clamp(tar_dir_norm, min=1e-6),
        fallback_dir / torch.clamp(fallback_norm, min=1e-6),
    )

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    root_vel_xy = root_vel[..., 0:2]

    vel_err = torch.sum(torch.square(tar_vel - root_vel_xy), dim=-1)
    vel_reward = torch.exp(-vel_err_scale * vel_err)

    # If the character moves opposite to the trajectory tangent, give no
    # velocity credit even if the vector error is accidentally small.
    proj_speed = torch.sum(tar_dir * root_vel_xy, dim=-1)
    wrong_dir = proj_speed <= 0.0
    vel_reward[wrong_dir] = 0.0

    # ------------------------------------------------------------
    # 3. Facing should align with the local trajectory tangent.
    # ------------------------------------------------------------
    heading_rot = torch_util.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = torch_util.quat_rotate(heading_rot, facing_dir)

    facing_dot = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_dot, 0.0)

    # ------------------------------------------------------------
    # 4. Optional overspeed regularizer.
    #    Full velocity matching already penalizes overspeed. This term makes
    #    cutting corners by sprinting even less attractive.
    # ------------------------------------------------------------
    root_speed = torch.linalg.norm(root_vel_xy, dim=-1)
    speed_limit = tar_speed + overspeed_slack
    overspeed = torch.clamp_min(root_speed - speed_limit, 0.0)
    overspeed_reward = torch.exp(-overspeed_err_scale * overspeed * overspeed)

    # No free velocity/facing reward near the target. The previous version set
    # these terms to 1 inside dist_threshold, which made the task much easier.
    reward = (
        pos_reward_w * pos_reward
        + vel_reward_w * vel_reward
        + face_reward_w * facing_reward
        + overspeed_reward_w * overspeed_reward
    )

    return reward
