"""CPU-only checks of the real champion wrapper and websocket request handler."""

import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from openpi_client import msgpack_numpy

from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper, B1KWrapperConfig
from b1k.shared.eval_compat import CAMERA_KEYS, PROPRIO_KEY
from b1k.shared.websocket_policy_server import WebsocketPolicyServer


def make_wrapper():
    # No checkpoint or model is constructed. Only the expensive infer function
    # is fake; wrapper state, transforms and request dispatch are production code.
    policy = Mock()
    policy._rng = object()
    policy.infer.return_value = {"actions": np.full((3, 23), 0.25, dtype=np.float32)}
    wrapper = B1KPolicyWrapper(policy, task_id=0, config=B1KWrapperConfig(
        actions_to_execute=2, actions_to_keep=1, execute_in_n_steps=2,
        apply_eval_tricks=False,
    ))
    wrapper.current_stage = 3
    wrapper.prediction_history.extend([3, 4])
    wrapper.step_count = 57
    wrapper.prediction_count = 9
    wrapper.last_actions = np.full((2, 23), 999.0)
    wrapper.action_index = 1
    wrapper.next_initial_actions = np.full((1, 23), 999.0)
    return wrapper, policy


def observation():
    return {
        PROPRIO_KEY: np.zeros((1, 61), dtype=np.float32),
        **{key: np.zeros((1, 8, 8, 3), dtype=np.uint8) for key in CAMERA_KEYS},
        "task_id": np.asarray([0]),
    }


class PolicyCacheTests(unittest.TestCase):
    def test_invalidation_clears_only_stale_actions_and_preserves_stage_state(self):
        wrapper, policy = make_wrapper()
        history = wrapper.prediction_history
        rng = policy._rng
        wrapper.invalidate_action_cache()
        self.assertIsNone(wrapper.last_actions)
        self.assertIsNone(wrapper.next_initial_actions)
        self.assertEqual(wrapper.action_index, 0)
        self.assertEqual(wrapper.current_stage, 3)
        self.assertIs(wrapper.prediction_history, history)
        self.assertEqual(list(history), [3, 4])
        self.assertEqual((wrapper.task_id, wrapper.step_count, wrapper.prediction_count), (0, 57, 9))
        self.assertIs(policy._rng, rng)
        policy.reset.assert_not_called()
        policy.infer.assert_not_called()

    def test_next_action_replans_with_current_observation_and_preserved_stage(self):
        wrapper, policy = make_wrapper()
        wrapper.invalidate_action_cache()
        action = wrapper.act(observation())
        policy.infer.assert_called_once()
        model_input = policy.infer.call_args.args[0]
        np.testing.assert_array_equal(model_input["tokenized_prompt"], [0, 3])
        self.assertEqual(int(model_input["subtask_state"]), 3)
        self.assertNotIn("initial_actions", model_input)
        self.assertEqual(policy.infer.call_args.kwargs, {})
        np.testing.assert_allclose(action.numpy(), 0.25)
        self.assertEqual(wrapper.current_stage, 3)
        self.assertEqual(list(wrapper.prediction_history), [3, 4])
        self.assertEqual((wrapper.step_count, wrapper.prediction_count), (58, 10))
        # A following VLA call uses the normal cache, without another inference.
        wrapper.act(observation())
        policy.infer.assert_called_once()
        policy.reset.assert_not_called()

    def test_stage_voting_can_continue_after_handoff(self):
        wrapper, policy = make_wrapper()
        wrapper.prediction_history.clear()
        wrapper.prediction_history.extend([4, 4])
        logits = np.zeros(15)
        logits[4] = 1
        policy.infer.return_value["subtask_logits"] = logits
        wrapper.invalidate_action_cache()
        wrapper.act(observation())
        np.testing.assert_array_equal(policy.infer.call_args.args[0]["tokenized_prompt"], [0, 3])
        self.assertEqual(wrapper.current_stage, 4)

    def test_full_reset_keeps_original_new_episode_behavior(self):
        wrapper, policy = make_wrapper()
        wrapper.reset()
        policy.reset.assert_called_once()
        self.assertEqual(wrapper.current_stage, 0)
        self.assertEqual(list(wrapper.prediction_history), [])
        self.assertEqual((wrapper.step_count, wrapper.prediction_count), (0, 0))
        self.assertIsNone(wrapper.last_actions)
        self.assertIsNone(wrapper.next_initial_actions)

    def test_task_change_still_resets_stage_as_in_baseline(self):
        wrapper, _ = make_wrapper()
        wrapper.invalidate_action_cache()
        wrapper._handle_task_change(1)
        self.assertEqual((wrapper.task_id, wrapper.current_stage), (1, 0))
        self.assertEqual(list(wrapper.prediction_history), [])


class FakeSocket:
    remote_address = ("test", 0)

    def __init__(self, requests):
        packer = msgpack_numpy.Packer()
        self.requests = iter(packer.pack(request) for request in requests)
        self.sent = []
        self.closed = None

    async def send(self, value):
        self.sent.append(value)

    async def recv(self):
        try:
            return next(self.requests)
        except StopIteration:
            # Cancel after the last request rather than constructing a fake
            # websockets connection; cancellation is not a handler error.
            import asyncio
            raise asyncio.CancelledError

    async def close(self, **kwargs):
        self.closed = kwargs


class PolicyControlProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def run_requests(self, server, requests):
        import asyncio
        socket = FakeSocket(requests)
        with self.assertRaises(asyncio.CancelledError):
            await server._handler(socket)
        return socket

    async def test_invalidation_and_next_observation_preserve_stage_without_extra_reply(self):
        wrapper, policy = make_wrapper()
        metadata = {"model": "test", "emerge_capabilities": {"other": True}}
        server = WebsocketPolicyServer(wrapper, metadata=metadata)
        socket = await self.run_requests(server, [{"invalidate_action_cache": True}, observation()])
        self.assertEqual(len(socket.sent), 2)  # metadata, then one action only
        advertised = msgpack_numpy.unpackb(socket.sent[0])
        self.assertTrue(advertised["emerge_capabilities"]["invalidate_action_cache"])
        self.assertTrue(advertised["emerge_capabilities"]["other"])
        self.assertNotIn("invalidate_action_cache", metadata["emerge_capabilities"])
        response = msgpack_numpy.unpackb(socket.sent[1])
        np.testing.assert_allclose(response["action"], 0.25)
        self.assertEqual(wrapper.current_stage, 3)
        policy.reset.assert_not_called()

    async def test_official_reset_still_resets_stage_and_sends_no_action(self):
        wrapper, policy = make_wrapper()
        socket = await self.run_requests(WebsocketPolicyServer(wrapper), [{"reset": True}])
        self.assertEqual(len(socket.sent), 1)
        policy.reset.assert_called_once()
        self.assertEqual(wrapper.current_stage, 0)

    async def test_unknown_or_mixed_cache_control_fails_without_resetting_policy(self):
        for request in ({"invalidate_action_cache": False},
                        {"invalidate_action_cache": True, "reset": True},
                        {"invalidate_action_cache": True, "observation": 123}):
            with self.subTest(request=request):
                wrapper, policy = make_wrapper()
                socket = FakeSocket([request])
                with self.assertRaisesRegex(ValueError, "isolated true"):
                    await WebsocketPolicyServer(wrapper)._handler(socket)
                self.assertEqual(socket.closed["code"], 1011)
                self.assertEqual(wrapper.current_stage, 3)
                policy.reset.assert_not_called()

    async def test_unsupported_policy_does_not_advertise_or_fall_back_to_reset(self):
        policy = type("LegacyPolicy", (), {"reset": Mock(), "act": Mock()})()
        server = WebsocketPolicyServer(policy)
        self.assertFalse(server._metadata["emerge_capabilities"]["invalidate_action_cache"])
        socket = FakeSocket([{"invalidate_action_cache": True}])
        with self.assertRaisesRegex(RuntimeError, "refusing full reset"):
            await server._handler(socket)
        policy.reset.assert_not_called()
        policy.act.assert_not_called()


@unittest.skipUnless(
    (Path(__file__).resolve().parents[3] / "robot/behavior/gateway.py").is_file(),
    "EMERGE checkout is needed for the gateway transport check",
)
class GatewayPolicyTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_vln_to_vla_transport_preserves_stage_until_official_reset(self):
        import asyncio
        from websockets.asyncio.client import connect
        from websockets.asyncio.server import serve
        from robot.behavior.gateway import B1KProxyGateway

        wrapper, policy = make_wrapper()
        server = WebsocketPolicyServer(wrapper)
        packer = msgpack_numpy.Packer()
        async with serve(server._handler, "127.0.0.1", 0, compression=None) as upstream:
            port = upstream.sockets[0].getsockname()[1]
            gateway = B1KProxyGateway(
                upstream_url=f"ws://127.0.0.1:{port}", listen_port=0, controlled=True,
            )
            gateway.start()
            try:
                async with connect(f"ws://127.0.0.1:{gateway.listen_port}", compression=None) as evaluator:
                    metadata = msgpack_numpy.unpackb(await evaluator.recv())
                    self.assertTrue(metadata["emerge_capabilities"]["invalidate_action_cache"])
                    local_id = gateway.begin_local_execution("navigate_vln", 1, [0.0] * 23)
                    await evaluator.send(packer.pack(observation()))
                    local = msgpack_numpy.unpackb(await asyncio.wait_for(evaluator.recv(), 3))
                    np.testing.assert_allclose(local["action"], 0.0)
                    await asyncio.to_thread(gateway.wait_for_execution, local_id, progress_timeout_s=3)

                    for _ in range(2):
                        vla_id = gateway.begin_execution("original task instruction", 1)
                        await evaluator.send(packer.pack(observation()))
                        response = msgpack_numpy.unpackb(await asyncio.wait_for(evaluator.recv(), 3))
                        np.testing.assert_allclose(response["action"], 0.25)
                        result = await asyncio.to_thread(gateway.wait_for_execution, vla_id, progress_timeout_s=3)
                        self.assertEqual(result["status"], "step_completed")
                    self.assertEqual(wrapper.current_stage, 3)
                    self.assertEqual(list(wrapper.prediction_history), [3, 4])
                    self.assertEqual((wrapper.step_count, wrapper.prediction_count), (59, 10))
                    policy.infer.assert_called_once()
                    policy.reset.assert_not_called()
                    self.assertEqual(gateway.snapshot()["upstream_cache_invalidations_sent"], 1)
                    self.assertEqual(gateway.snapshot()["evaluator_resets_forwarded"], 0)

                    await evaluator.send(packer.pack({"reset": True}))
                    vla_id = gateway.begin_execution("original task instruction", 1)
                    await evaluator.send(packer.pack(observation()))
                    await asyncio.wait_for(evaluator.recv(), 3)
                    await asyncio.to_thread(gateway.wait_for_execution, vla_id, progress_timeout_s=3)
                    self.assertEqual(wrapper.current_stage, 0)
                    self.assertEqual(list(wrapper.prediction_history), [])
                    self.assertEqual((wrapper.step_count, wrapper.prediction_count), (1, 1))
                    policy.reset.assert_called_once()
                    self.assertEqual(gateway.snapshot()["evaluator_resets_forwarded"], 1)
                    self.assertEqual(gateway.snapshot()["upstream_cache_invalidations_sent"], 1)
            finally:
                await asyncio.to_thread(gateway.stop)


if __name__ == "__main__":
    unittest.main()
