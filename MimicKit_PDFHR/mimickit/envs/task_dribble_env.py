import numpy as np
import torch

import engines.engine as engine
import envs.smp_env as smp_env
import envs.base_env as base_env
import util.torch_util as torch_util


class TaskDribbleEnv(smp_env.SMPEnv):
    """
    Velocity-direction dribble task for MimicKit.

    The humanoid no longer pushes the ball to a sampled target location.
    Instead, each environment samples a desired ball velocity:

        v*_ball = tar_speed * tar_dir

    The policy is rewarded for keeping the ball near the feet while making
    the ball move along the sampled direction at the desired speed.

    Task obs layout:
        local_ball_pos      : 3
        local_ball_vel      : 3
        local_tar_dir_xy    : 2
        tar_speed           : 1
    Total:
        task_obs_dim = 9
    """

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        # ------------------------------------------------------------
        # Asset / geometry
        # ------------------------------------------------------------
        self._ball_asset_file = env_config.get(
            "ball_asset_file",
            "data/assets/objects/dodgeball.xml",
        )

        # Direction marker. Prefer steering_marker.xml because this task is
        # direction/velocity based, not location-target based.
        self._velocity_marker_asset_file = env_config.get(
            "velocity_marker_asset_file",
            env_config.get("target_marker_asset_file", "data/assets/objects/steering_marker.xml"),
        )

        self._ball_radius = float(env_config.get("ball_radius", 0.1))
        self._ball_height = float(env_config.get("ball_height", self._ball_radius))
        self._velocity_marker_height = float(env_config.get("velocity_marker_height", 0.02))
        self._velocity_marker_dist = float(env_config.get("velocity_marker_dist", 1.5))

        # ------------------------------------------------------------
        # Reset distribution
        # ------------------------------------------------------------
        self._ball_start_forward = float(env_config.get("ball_start_forward", 0.75))
        self._ball_start_lateral_noise = float(env_config.get("ball_start_lateral_noise", 0.15))

        # Desired ball velocity, steering-style.
        # Backward compatible: if tar_speed_min/max are not provided, use tar_speed.
        default_tar_speed = float(env_config.get("tar_speed", 1.5))
        self._tar_speed_min = float(env_config.get("tar_speed_min", default_tar_speed))
        self._tar_speed_max = float(env_config.get("tar_speed_max", default_tar_speed))
        assert self._tar_speed_max >= self._tar_speed_min

        self._tar_change_time_min = float(env_config.get("tar_change_time_min", 3.0))
        self._tar_change_time_max = float(env_config.get("tar_change_time_max", 6.0))
        assert self._tar_change_time_max >= self._tar_change_time_min

        self._rand_tar_dir = bool(env_config.get("rand_tar_dir", True))

        # If True, target velocity direction is sampled around humanoid heading.
        # If False, it is sampled uniformly in the world frame when rand_tar_dir=True.
        self._target_forward_bias = bool(env_config.get("target_forward_bias", True))
        self._target_angle_noise = float(env_config.get("target_angle_noise", np.pi / 3.0))

        # ------------------------------------------------------------
        # Ball physics helpers
        # ------------------------------------------------------------
        self._ball_linear_damping = float(env_config.get("ball_linear_damping", 0.02))
        self._ball_angular_damping = float(env_config.get("ball_angular_damping", 0.05))
        self._ball_max_speed = float(env_config.get("ball_max_speed", 4.5))

        # ------------------------------------------------------------
        # Reward config
        # ------------------------------------------------------------
        self._char_ball_err_scale = float(env_config.get("char_ball_err_scale", 4.0))
        self._ball_vel_err_scale = float(env_config.get("ball_vel_err_scale", 2.5))
        self._ball_height_err_scale = float(env_config.get("ball_height_err_scale", 8.0))

        self._near_ball_reward_w = float(env_config.get("near_ball_reward_w", 0.35))
        self._ball_vel_reward_w = float(env_config.get("ball_vel_reward_w", 0.55))
        self._ball_height_reward_w = float(env_config.get("ball_height_reward_w", 0.10))

        # Feet used for near-ball reward.
        self._dribble_foot_bodies = env_config.get(
            "dribble_foot_bodies",
            env_config.get("contact_bodies", ["right_foot", "left_foot"]),
        )

        # ------------------------------------------------------------
        # Early termination
        # ------------------------------------------------------------
        self._enable_dribble_fail = bool(env_config.get("enable_dribble_fail", True))
        self._char_ball_fail_dist = float(env_config.get("char_ball_fail_dist", 6.0))
        self._ball_fail_height = float(env_config.get("ball_fail_height", 1.2))
        self._fail_grace_time = float(env_config.get("fail_grace_time", 0.5))

        # ------------------------------------------------------------
        # Colors
        # ------------------------------------------------------------
        self._ball_color = env_config.get("ball_color", [0.9725, 0.42, 0.1137])
        self._velocity_marker_color = env_config.get("velocity_marker_color", [0.8, 0.0, 0.0])

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
    # Ball physics helper
    # ============================================================
    def _apply_ball_damping(self):
        ball_id = self._get_ball_id()

        ball_vel = self._engine.get_root_vel(ball_id)
        ball_ang_vel = self._engine.get_root_ang_vel(ball_id)

        if self._ball_linear_damping > 0.0:
            ball_vel = ball_vel.clone()
            ball_vel[:, 0:2] *= (1.0 - self._ball_linear_damping)

        if self._ball_angular_damping > 0.0:
            ball_ang_vel = ball_ang_vel.clone()
            ball_ang_vel *= (1.0 - self._ball_angular_damping)

        if self._ball_max_speed > 0.0:
            xy_vel = ball_vel[:, 0:2]
            speed = torch.linalg.norm(xy_vel, dim=-1, keepdim=True)
            scale = torch.clamp(self._ball_max_speed / (speed + 1e-6), max=1.0)
            ball_vel[:, 0:2] = xy_vel * scale

        env_ids = torch.arange(
            self.get_num_envs(),
            device=self._device,
            dtype=torch.long,
        )

        self._engine.set_root_vel(env_ids, ball_id, ball_vel)
        self._engine.set_root_ang_vel(env_ids, ball_id, ball_ang_vel)
        return

    # ============================================================
    # Build env / objects
    # ============================================================
    def _build_envs(self, config, num_envs):
        self._ball_ids = []

        if self._visualize:
            self._velocity_marker_ids = []

        super()._build_envs(config, num_envs)
        return

    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)

        ball_id = self._build_ball(env_id)
        if env_id == 0:
            self._ball_ids.append(ball_id)
        else:
            assert self._ball_ids[0] == ball_id

        if self._visualize:
            marker_id = self._build_velocity_marker(env_id)
            if env_id == 0:
                self._velocity_marker_ids.append(marker_id)
            else:
                assert self._velocity_marker_ids[0] == marker_id

        return

    def _build_ball(self, env_id):
        start_pos = np.array([10.0, 10.0, self._ball_height], dtype=np.float32)
        start_rot = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        ball_id = self._engine.create_obj(
            env_id=env_id,
            obj_type=engine.ObjType.rigid,
            asset_file=self._ball_asset_file,
            name="dribble_ball",
            start_pos=start_pos,
            start_rot=start_rot,
            color=self._ball_color,
        )
        return ball_id

    def _build_velocity_marker(self, env_id):
        marker_id = self._engine.create_obj(
            env_id=env_id,
            obj_type=engine.ObjType.rigid,
            asset_file=self._velocity_marker_asset_file,
            name="dribble_velocity_marker",
            is_visual=True,
            fix_root=True,
            color=self._velocity_marker_color,
        )
        return marker_id

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)

        num_envs = self.get_num_envs()

        self._tar_speed = torch.ones(
            [num_envs],
            device=self._device,
            dtype=torch.float,
        ) * self._tar_speed_min

        self._tar_dir = torch.zeros(
            [num_envs, 2],
            device=self._device,
            dtype=torch.float,
        )
        self._tar_dir[:, 0] = 1.0

        self._tar_change_times = torch.zeros(
            [num_envs],
            device=self._device,
            dtype=torch.float,
        )

        self._prev_ball_pos = torch.zeros(
            [num_envs, 3],
            device=self._device,
            dtype=torch.float,
        )

        self._prev_root_pos = torch.zeros(
            [num_envs, 3],
            device=self._device,
            dtype=torch.float,
        )

        self._dribble_foot_body_ids = self._build_body_ids_tensor(
            self._dribble_foot_bodies
        ).to(device=self._device, dtype=torch.long)

        return

    def _get_ball_id(self):
        return self._ball_ids[0]

    def _get_velocity_marker_id(self):
        return self._velocity_marker_ids[0]

    # ============================================================
    # Step / reset
    # ============================================================
    def _pre_physics_step(self, actions):
        self._apply_ball_damping()
        super()._pre_physics_step(actions)
        self._record_prev_states()
        return

    def _record_prev_states(self):
        char_id = self._get_char_id()
        ball_id = self._get_ball_id()

        self._prev_root_pos[:] = self._engine.get_root_pos(char_id)
        self._prev_ball_pos[:] = self._engine.get_root_pos(ball_id)
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)

        if len(env_ids) > 0:
            self._reset_task(env_ids)
            self._record_prev_states()
        return

    def _reset_task(self, env_ids):
        self._reset_ball(env_ids)
        self._reset_tar(env_ids)

        if self._visualize:
            self._update_marker(env_ids)

        return

    def _reset_ball(self, env_ids):
        n = env_ids.shape[0]
        char_id = self._get_char_id()
        ball_id = self._get_ball_id()

        root_pos = self._engine.get_root_pos(char_id)[env_ids]
        root_rot = self._engine.get_root_rot(char_id)[env_ids]

        heading_rot = torch_util.calc_heading_quat(root_rot)

        local_offset = torch.zeros([n, 3], device=self._device, dtype=torch.float)
        local_offset[:, 0] = self._ball_start_forward
        local_offset[:, 1] = (
            2.0 * torch.rand([n], device=self._device, dtype=torch.float) - 1.0
        ) * self._ball_start_lateral_noise
        local_offset[:, 2] = 0.0

        world_offset = torch_util.quat_rotate(heading_rot, local_offset)

        ball_pos = root_pos + world_offset
        ball_pos[:, 2] = self._ball_height

        ball_rot = torch.zeros([n, 4], device=self._device, dtype=torch.float)
        ball_rot[:, 3] = 1.0

        zero_vel = torch.zeros([n, 3], device=self._device, dtype=torch.float)

        self._engine.set_root_pos(env_ids, ball_id, ball_pos)
        self._engine.set_root_rot(env_ids, ball_id, ball_rot)
        self._engine.set_root_vel(env_ids, ball_id, zero_vel)
        self._engine.set_root_ang_vel(env_ids, ball_id, zero_vel)

        self._prev_ball_pos[env_ids] = ball_pos
        return

    def _reset_tar(self, env_ids):
        n = env_ids.shape[0]
        char_id = self._get_char_id()

        root_rot = self._engine.get_root_rot(char_id)[env_ids]

        if self._target_forward_bias:
            heading_rot = torch_util.calc_heading_quat(root_rot)
            forward = torch.zeros([n, 3], device=self._device, dtype=torch.float)
            forward[:, 0] = 1.0
            forward = torch_util.quat_rotate(heading_rot, forward)
            base_theta = torch.atan2(forward[:, 1], forward[:, 0])

            if self._rand_tar_dir:
                rand_theta = (
                    base_theta
                    + (2.0 * torch.rand([n], device=self._device, dtype=torch.float) - 1.0)
                    * self._target_angle_noise
                )
            else:
                rand_theta = base_theta
        else:
            if self._rand_tar_dir:
                rand_theta = 2.0 * np.pi * torch.rand([n], device=self._device, dtype=torch.float) - np.pi
            else:
                rand_theta = torch.zeros([n], device=self._device, dtype=torch.float)

        tar_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)

        rand_speed = torch.rand([n], device=self._device, dtype=torch.float)
        tar_speed = (self._tar_speed_max - self._tar_speed_min) * rand_speed + self._tar_speed_min

        rand_dt = torch.rand([n], device=self._device, dtype=torch.float)
        rand_dt = (self._tar_change_time_max - self._tar_change_time_min) * rand_dt + self._tar_change_time_min

        self._tar_dir[env_ids] = tar_dir
        self._tar_speed[env_ids] = tar_speed
        self._tar_change_times[env_ids] = self._time_buf[env_ids] + rand_dt
        return

    def _update_marker(self, env_ids):
        marker_id = self._get_velocity_marker_id()
        ball_id = self._get_ball_id()

        ball_pos = self._engine.get_root_pos(ball_id)[env_ids]
        tar_dir = self._tar_dir[env_ids]
        tar_speed = self._tar_speed[env_ids]

        # Visual length scales with target speed, like steering marker.
        if self._tar_speed_max > self._tar_speed_min:
            speed_alpha = (tar_speed - self._tar_speed_min) / (self._tar_speed_max - self._tar_speed_min)
        else:
            speed_alpha = torch.zeros_like(tar_speed)
        marker_dist = self._velocity_marker_dist * (0.75 + 0.5 * speed_alpha)
        marker_dist = marker_dist.unsqueeze(-1)

        marker_pos = ball_pos.clone()
        marker_pos[:, 0:2] += marker_dist * tar_dir
        marker_pos[:, 2] = self._velocity_marker_height

        theta = torch.atan2(tar_dir[:, 1], tar_dir[:, 0])
        axis = torch.zeros_like(marker_pos)
        axis[:, 2] = 1.0
        marker_rot = torch_util.axis_angle_to_quat(axis, theta)

        zero_vel = torch.zeros([env_ids.shape[0], 3], device=self._device, dtype=torch.float)

        self._engine.set_root_pos(env_ids, marker_id, marker_pos)
        self._engine.set_root_rot(env_ids, marker_id, marker_rot)
        self._engine.set_root_vel(env_ids, marker_id, zero_vel)
        self._engine.set_root_ang_vel(env_ids, marker_id, zero_vel)
        return

    def _update_task(self):
        reset_task_mask = self._time_buf >= self._tar_change_times
        reset_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()

        if len(reset_env_ids) > 0:
            self._reset_tar(reset_env_ids)

        if self._visualize:
            self._update_marker(self._env_ids)

        return

    def _update_misc(self):
        super()._update_misc()
        self._update_task()
        return

    # ============================================================
    # Task obs module
    # ============================================================
    def _get_task_obs_dim(self):
        return 9

    def _compute_task_obs(self, env_ids=None):
        char_id = self._get_char_id()
        ball_id = self._get_ball_id()

        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        ball_pos = self._engine.get_root_pos(ball_id)
        ball_vel = self._engine.get_root_vel(ball_id)
        tar_dir = self._tar_dir
        tar_speed = self._tar_speed

        if env_ids is not None:
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            ball_pos = ball_pos[env_ids]
            ball_vel = ball_vel[env_ids]
            tar_dir = tar_dir[env_ids]
            tar_speed = tar_speed[env_ids]

        task_obs = compute_dribble_observations(
            root_pos=root_pos,
            root_rot=root_rot,
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            tar_dir=tar_dir,
            tar_speed=tar_speed,
        )

        assert task_obs.shape[-1] == self._get_task_obs_dim(), (
            f"task_obs dim mismatch: got {task_obs.shape[-1]}, "
            f"expected {self._get_task_obs_dim()}"
        )

        return task_obs

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        task_obs = self._compute_task_obs(env_ids)
        obs = torch.cat([obs, task_obs], dim=-1)
        return obs

    # ============================================================
    # Reward / done
    # ============================================================
    def _update_reward(self):
        char_id = self._get_char_id()
        ball_id = self._get_ball_id()

        root_pos = self._engine.get_root_pos(char_id)
        body_pos = self._engine.get_body_pos(char_id)
        foot_pos = body_pos[:, self._dribble_foot_body_ids, :]

        ball_pos = self._engine.get_root_pos(ball_id)
        ball_vel = self._engine.get_root_vel(ball_id)

        self._reward_buf[:] = compute_dribble_reward(
            foot_pos=foot_pos,
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            tar_dir=self._tar_dir,
            tar_speed=self._tar_speed,
            ball_height=self._ball_height,
            char_ball_err_scale=self._char_ball_err_scale,
            ball_vel_err_scale=self._ball_vel_err_scale,
            ball_height_err_scale=self._ball_height_err_scale,
            near_ball_reward_w=self._near_ball_reward_w,
            ball_vel_reward_w=self._ball_vel_reward_w,
            ball_height_reward_w=self._ball_height_reward_w,
        )
        return

    def _update_done(self):
        super()._update_done()

        if not (self._enable_early_termination and self._enable_dribble_fail):
            return

        char_id = self._get_char_id()
        ball_id = self._get_ball_id()

        root_pos = self._engine.get_root_pos(char_id)
        ball_pos = self._engine.get_root_pos(ball_id)

        fail = compute_dribble_fail_flags(
            root_pos=root_pos,
            ball_pos=ball_pos,
            char_ball_fail_dist=self._char_ball_fail_dist,
            ball_fail_height=self._ball_fail_height,
        )

        if hasattr(self, "_time_buf"):
            fail = torch.logical_and(fail, self._time_buf > self._fail_grace_time)

        self._done_buf[fail] = base_env.DoneFlags.FAIL.value
        return

    # ============================================================
    # Rendering
    # ============================================================
    def _render_scene(self):
        super()._render_scene()
        self._render_dribble_lines()
        return

    def _render_dribble_lines(self):
        if not hasattr(self._engine, "draw_lines"):
            return

        ball_id = self._get_ball_id()
        ball_pos = self._engine.get_root_pos(ball_id)

        starts = ball_pos.detach().cpu().numpy().copy()
        dirs = self._tar_dir.detach().cpu().numpy().copy()
        speed = self._tar_speed.detach().cpu().numpy().copy()

        if self._tar_speed_max > self._tar_speed_min:
            alpha = (speed - self._tar_speed_min) / (self._tar_speed_max - self._tar_speed_min)
        else:
            alpha = np.zeros_like(speed)
        dist = self._velocity_marker_dist * (0.75 + 0.5 * alpha)

        ends = starts.copy()
        ends[:, 0:2] += dist[:, None] * dirs
        starts[:, 2] = self._velocity_marker_height
        ends[:, 2] = self._velocity_marker_height

        col = np.array([[1.0, 0.0, 0.0, 0.5]], dtype=np.float32)
        line_width = 2.0

        num_envs = self.get_num_envs()
        for i in range(num_envs):
            self._engine.draw_lines(i, starts[i:i + 1], ends[i:i + 1], col, line_width)

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_dribble_observations(root_pos, root_rot, ball_pos, ball_vel, tar_dir, tar_speed):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)

    local_ball_pos = torch_util.quat_rotate(
        heading_inv_rot,
        ball_pos - root_pos,
    )

    local_ball_vel = torch_util.quat_rotate(
        heading_inv_rot,
        ball_vel,
    )

    tar_dir3d = torch.cat(
        [tar_dir, torch.zeros_like(tar_dir[..., 0:1])],
        dim=-1,
    )
    local_tar_dir = torch_util.quat_rotate(
        heading_inv_rot,
        tar_dir3d,
    )[..., 0:2]

    obs = torch.cat(
        [
            local_ball_pos,
            local_ball_vel,
            local_tar_dir,
            tar_speed.unsqueeze(-1),
        ],
        dim=-1,
    )

    return obs


@torch.jit.script
def compute_dribble_reward(
    foot_pos,
    ball_pos,
    ball_vel,
    tar_dir,
    tar_speed,
    ball_height,
    char_ball_err_scale,
    ball_vel_err_scale,
    ball_height_err_scale,
    near_ball_reward_w,
    ball_vel_reward_w,
    ball_height_reward_w,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float) -> Tensor

    # ------------------------------------------------------------
    # Feet close to ball reward
    # ------------------------------------------------------------
    foot_ball_diff = foot_pos[..., 0:2] - ball_pos.unsqueeze(1)[..., 0:2]
    foot_ball_dist = torch.linalg.norm(foot_ball_diff, dim=-1)
    min_foot_ball_dist = torch.min(foot_ball_dist, dim=-1).values
    near_ball_reward = torch.exp(
        -char_ball_err_scale * min_foot_ball_dist * min_foot_ball_dist
    )

    # ------------------------------------------------------------
    # Ball target velocity reward: v_ball -> tar_speed * tar_dir
    # ------------------------------------------------------------
    tar_vel = tar_speed.unsqueeze(-1) * tar_dir
    ball_vel_xy = ball_vel[..., 0:2]
    ball_vel_err = torch.sum(torch.square(tar_vel - ball_vel_xy), dim=-1)
    ball_vel_reward = torch.exp(-ball_vel_err_scale * ball_vel_err)

    # No reward if the ball moves opposite to the desired direction.
    proj_speed = torch.sum(tar_dir * ball_vel_xy, dim=-1)
    wrong_dir = proj_speed <= 0.0
    ball_vel_reward[wrong_dir] = 0.0

    # ------------------------------------------------------------
    # Keep ball near ground / foot height
    # ------------------------------------------------------------
    height_err = ball_pos[..., 2] - ball_height
    height_reward = torch.exp(-ball_height_err_scale * height_err * height_err)

    reward = (
        near_ball_reward_w * near_ball_reward
        + ball_vel_reward_w * ball_vel_reward
        + ball_height_reward_w * height_reward
    )

    return reward


@torch.jit.script
def compute_dribble_fail_flags(root_pos, ball_pos, char_ball_fail_dist, ball_fail_height):
    # type: (Tensor, Tensor, float, float) -> Tensor

    char_ball_dist = torch.linalg.norm(
        root_pos[..., 0:2] - ball_pos[..., 0:2],
        dim=-1,
    )

    too_far = char_ball_dist > char_ball_fail_dist
    too_high = ball_pos[..., 2] > ball_fail_height

    fail = torch.logical_or(too_far, too_high)
    return fail
