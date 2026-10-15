"""Closed-loop LIBERO evaluation with research strategies, plugged into the unchanged run_libero_eval.py loop.

Modes (`--research_mode`):
  none       identical to run_libero_eval.py
  oracle_l3  depth oracle: every call runs `oracle_F` loops, decodes each, and returns the action from
             k* = min{k : max_{j>=k} mean|a_j - a_F| <= oracle_eps} (normalized action units). k* is reported as
             the call's loop count, so `iters_per_call` in the JSON is the oracle depth. Execution horizon as in
             the base eval (num_exec_actions=5 unless the adaptive-execution flags are given).
  shadow     horizon oracle: query the policy at EVERY step ("shadow query"), keep executing the current chunk
             (up to 8 actions) while the fresh query's first action stays within `shadow_theta` of the planned
             action (mean |.| over the 7 action dims, env units), and adopt the fresh chunk otherwise or when the
             chunk is used up. `num_predictions` in the JSON = adopted calls (what an oracle would have needed);
             per-step disagreements go to <json_log_file>.shadow.json. theta=inf reproduces horizon 8.

Usage: same flags as run_libero_eval.py plus the research flags, e.g.
  python research/run_eval_research.py --pretrained_checkpoint <ckpt> --task_suite_name libero_spatial \
     --task_id 0 --research_mode oracle_l3 --oracle_eps 0.01 --json_log_file out.json
"""
import json
import math
from collections import deque
from dataclasses import dataclass

import draccus
import numpy as np
import torch

import experiments.robot.libero.run_libero_eval as R
from research.rdvla_tools import NUM_TASK_TOKENS, Features, LoopRunner, head_layers


@dataclass
class ResearchConfig(R.GenerateConfig):
    research_mode: str = "none"  # none | oracle_l3 | shadow
    oracle_eps: float = 0.01
    oracle_F: int = 20
    shadow_theta: float = math.inf


class ResearchHead:
    """Wraps ActionHeadRecurrent; vla.predict_action calls `predict_action` with the VLM hidden states."""

    def __init__(self, head, cfg: ResearchConfig):
        self.head, self.cfg = head, cfg
        self.runner = LoopRunner(head.model, kv_cache=True)
        self.layers = head_layers(head.model)

    def predict_action(self, ahs, proprio=None, proprio_projector=None, **kw):
        if self.cfg.research_mode != "oracle_l3":
            return self.head.predict_action(ahs, proprio=proprio, proprio_projector=proprio_projector, **kw)
        B = ahs.shape[0]
        p = proprio_projector(proprio.reshape(B, -1).to(torch.bfloat16)).unsqueeze(1)
        hs = ahs[:, self.layers]
        f = Features(hs[:, :, NUM_TASK_TOKENS:], hs[:, :, :NUM_TASK_TOKENS], p, self.layers)
        out, k = self.runner.oracle_l3(f, self.cfg.oracle_eps, self.cfg.oracle_F)
        return out, int(k[0]), None


SHADOW_LOG = []  # one dict per episode, in episode order


def run_episode_shadow(cfg, env, task_description, model, resize_size, processor=None, action_head=None,
                       proprio_projector=None, initial_state=None, log_file=None, global_iters=None):
    """run_libero_eval.run_episode with a shadow query at every control step (see module docstring)."""
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()
    t, replay_images, max_steps = 0, [], R.TASK_MAX_STEPS[cfg.task_suite_name]
    queue, replay_stats, episode_iters = deque(), [], []
    deltas, positions, adopted_at = [], [], []
    executed_from_chunk, n_queries, success = 0, 0, False
    try:
        while t < max_steps + cfg.num_steps_wait:
            if t < cfg.num_steps_wait:
                obs, _, _, _ = env.step(R.get_libero_dummy_action(cfg.model_family))
                t += 1
                continue
            observation, img = R.prepare_observation(obs, resize_size)
            replay_images.append(img)
            actions, iters, _ = R.get_action(cfg, model, observation, task_description, processor=processor,
                                             action_head=action_head, proprio_projector=proprio_projector,
                                             use_film=cfg.use_film, use_minivlm=cfg.use_minivlm)
            n_queries += 1
            fresh = [np.asarray(a) for a in actions[:8]]
            adopt = len(queue) == 0
            if not adopt:
                d = float(np.abs(queue[0] - fresh[0]).mean())
                deltas.append(round(d, 5))
                positions.append(executed_from_chunk)
                adopt = d > cfg.shadow_theta
            if adopt:
                if replay_stats:
                    replay_stats[-1] = (replay_stats[-1][0], executed_from_chunk)
                queue = deque(fresh)
                replay_stats.append((iters or 0, 0))
                adopted_at.append(len(replay_images) - 1)
                executed_from_chunk = 0
            action = queue.popleft()
            executed_from_chunk += 1
            obs, _, done, _ = env.step(R.process_action(action, cfg.model_family).tolist())
            if done:
                success = True
                break
            t += 1
    except Exception as e:
        R.log_message(f"Episode error: {e}", log_file)
    if replay_stats:
        replay_stats[-1] = (replay_stats[-1][0], executed_from_chunk)
    SHADOW_LOG.append({"success": success, "queries": n_queries, "adopted": len(replay_stats),
                       "deltas": deltas, "pos_in_chunk": positions, "adopted_at_step": adopted_at})
    return success, replay_images, episode_iters, replay_stats


@draccus.wrap()
def main(cfg: ResearchConfig):
    orig_init = R.initialize_model

    def initialize_model(c):
        model, head, proj, processor = orig_init(c)
        return model, ResearchHead(head, cfg), proj, processor

    R.initialize_model = initialize_model
    if cfg.research_mode == "shadow":
        R.run_episode = run_episode_shadow
    rate = R.eval_libero.__wrapped__(cfg)
    if cfg.research_mode == "shadow" and cfg.json_log_file:
        json.dump({"theta": cfg.shadow_theta, "episodes": SHADOW_LOG}, open(cfg.json_log_file + ".shadow.json", "w"))
    return rate


if __name__ == "__main__":
    main()
