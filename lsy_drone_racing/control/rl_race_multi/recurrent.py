"""Recurrent rollout and sequence forward adapter for the vendored Brax PPO.

Brax still owns minibatch shuffling, clipping, optimizer epochs, pmap and logging.
No physics derivatives are taken. Carries are sampled constants at each TBPTT boundary.
"""

from typing import Any

import jax
import jax.numpy as jnp
from brax.training import types
from brax.training.agents.ppo import losses as brax_losses

from .network import reset_carry


def generate_unroll(
    env: Any, state: Any, policy: Any, key: Any, unroll_length: int, extra_fields: tuple = ()
) -> tuple:
    def step(carry: Any, _: Any) -> tuple:
        state, key = carry
        key, action_key = jax.random.split(key)
        hidden = state.info["recurrent"]
        next_hidden, action, extras = policy(state.obs, hidden, action_key)
        extras = dict(
            extras,
            actor_c=hidden[0][0],
            actor_h=hidden[0][1],
            critic_c=hidden[1][0],
            critic_h=hidden[1][1],
        )
        next_state = env.step(state, action)
        info = dict(next_state.info)
        info["recurrent"] = reset_carry(
            next_hidden, next_state.info.get("learner_reset", next_state.done) > 0
        )
        next_state = next_state.replace(info=info)
        selected = {name: next_state.info[name] for name in extra_fields}
        selected["terminal_obs"] = next_state.info["terminal_obs"]
        transition = types.Transition(
            state.obs,
            action,
            next_state.reward,
            1 - next_state.done,
            next_state.obs,
            {"policy_extras": extras, "state_extras": selected},
        )
        return (next_state, key), transition

    (state, _), data = jax.lax.scan(step, (state, key), None, length=unroll_length)
    return state, jax.tree.map(jax.lax.stop_gradient, data)


def sequence_apply(
    network: Any, stats: Any, params: Any, observations: Any, carry: Any, done: Any
) -> tuple:
    """Time-major sequence; resets happen after consuming the terminal step."""

    def forward(carry: Any, item: Any) -> tuple:
        obs, ended = item
        next_carry, output = network.apply(stats, params, obs, carry)
        return reset_carry(next_carry, ended), (output, next_carry)

    return jax.lax.scan(forward, jax.tree.map(jax.lax.stop_gradient, carry), (observations, done))


def compute_ppo_loss(
    params: Any, normalizer_params: Any, data: Any, rng: Any, ppo_network: Any, **kwargs: Any
) -> tuple:
    """Recompute continuous LSTM sequences, then delegate PPO math to Brax's loss.

    Timeout transitions retain the un-reset terminal observation and recurrent carry;
    reward bootstraps there, and GAE never crosses a reset boundary.
    """
    time = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), data)
    extra = time.extras["policy_extras"]
    actor0 = (extra["actor_c"][0], extra["actor_h"][0])
    critic0 = (extra["critic_c"][0], extra["critic_h"][0])
    done = time.discount == 0
    _, (logits, _) = sequence_apply(
        ppo_network.policy_network, normalizer_params, params.policy, time.observation, actor0, done
    )
    critic_final, (values, critic_after) = sequence_apply(
        ppo_network.value_network, normalizer_params, params.value, time.observation, critic0, done
    )
    # Evaluate each physical terminal observation with its pre-reset carry.
    _, terminal_values = ppo_network.value_network.apply(
        normalizer_params, params.value, time.extras["state_extras"]["terminal_obs"], critic_after
    )
    _, bootstrap = ppo_network.value_network.apply(
        normalizer_params, params.value, time.next_observation[-1], critic_final
    )
    truncation = time.extras["state_extras"]["truncation"]
    gamma = kwargs.get("discounting", 0.9)
    scale = kwargs.get("reward_scaling", 1.0)
    adjusted_reward = time.reward * scale + gamma * truncation * jax.lax.stop_gradient(
        terminal_values
    )
    kwargs["reward_scaling"] = 1.0
    # Brax's GAE treats termination as the trajectory boundary. Timeouts are
    # bootstrapped above and then terminate GAE, rather than using reset-state V.
    state_extras = dict(data.extras["state_extras"], truncation=jnp.zeros_like(data.reward))
    adapted = data._replace(
        reward=jnp.swapaxes(adjusted_reward, 0, 1),
        extras=dict(data.extras, state_extras=state_extras),
    )
    from brax.training.agents.ppo.networks import PPONetworks
    from brax.training.networks import FeedForwardNetwork

    # Brax calls apply on [T,B,57] and then [B,57]. Captured outputs remain
    # differentiable wrt new params; recurrent initial carries are stop_gradient.
    policy_net = FeedForwardNetwork(init=ppo_network.policy_network.init, apply=lambda *a: logits)
    value_net = FeedForwardNetwork(
        init=ppo_network.value_network.init,
        apply=lambda *a: values if a[-1].ndim == 3 else bootstrap,
    )
    compatible = PPONetworks(policy_net, value_net, ppo_network.parametric_action_distribution)
    return brax_losses.compute_ppo_loss(
        params, normalizer_params, adapted, rng, ppo_network=compatible, **kwargs
    )


class Evaluator:
    """Fixed-seed first-episode evaluation with exactly the rollout carry semantics."""

    def __init__(
        self,
        eval_env: Any,
        eval_policy_fn: Any,
        num_eval_envs: int,
        episode_length: int,
        action_repeat: int,
        key: Any,
    ):
        if action_repeat != 1:
            raise ValueError("Recurrent racing requires action_repeat=1")
        self.env, self.policy_fn = eval_env, eval_policy_fn
        self.count, self.length, self.key = num_eval_envs, episode_length, key
        self._run = jax.jit(self._evaluate)

    def _evaluate(self, params: Any, key: Any) -> dict:
        env = self.env
        state = env.reset(jax.random.split(key, self.count))
        seen = jnp.zeros(self.count, bool)
        sums = {k: jnp.zeros(self.count) for k in state.metrics}
        sums["reward"] = jnp.zeros(self.count)
        policy = self.policy_fn(params)

        def step(carry: Any, _: Any) -> tuple:
            state, seen, sums, key = carry
            key, subkey = jax.random.split(key)
            hidden, action, _ = policy(state.obs, state.info["recurrent"], subkey)
            next_state = env.step(state, action)
            info = dict(
                next_state.info,
                recurrent=reset_carry(
                    hidden, next_state.info.get("learner_reset", next_state.done) > 0
                ),
            )
            next_state = next_state.replace(info=info)
            metrics = dict(next_state.metrics, reward=next_state.reward)
            sums = {k: sums[k] + jnp.where(~seen, metrics[k], 0.0) for k in sums}
            return (next_state, seen | (next_state.done > 0), sums, key), None

        (_, seen, sums, _), _ = jax.lax.scan(
            step, (state, seen, sums, key), None, length=self.length
        )
        return {
            **{f"eval/episode_{k}": jnp.mean(v) for k, v in sums.items()},
            "eval/all_finished": jnp.all(seen).astype(jnp.float32),
        }

    def run_evaluation(self, params: Any, training_metrics: dict, **kwargs: Any) -> dict:
        # Re-use fixed episode seeds across calls to make thresholds comparable.
        values = self._run(params, self.key)
        if not bool(values["eval/all_finished"]):
            raise RuntimeError("Evaluation horizon did not finish every first episode")
        return {**training_metrics, **{k: float(v) for k, v in values.items()}}
