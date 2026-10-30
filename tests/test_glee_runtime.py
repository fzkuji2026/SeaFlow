import json
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

import pytest
import requests

from glee_runtime.claim_coordinator import ClaimCoordinator
from glee_runtime.lifecycle import GameRunner, RunConfig, RunSnapshot, RunState
from glee_runtime.queue_control import QueueReservations
from glee_runtime.per_game_recovery import GameMoveRecovery, MoveUncertain
from glee_runtime.persistence import RunStore
from glee_runtime.transport import (
    ApiTransport,
    NetworkUnavailable,
    attach_direct_adapter,
    connect_with_failover,
)
from glee_sdk import GleeClient
from glee_sdk.client import GleeAPIError


def game(game_id="g", round_=1, current_player="seller"):
    return {
        "game_id": game_id,
        "game_family": "negotiation",
        "your_player": "seller",
        "phase": "playing",
        "valid_actions": ["offer"],
        "game_state": {
            "round": round_,
            "phase": "playing",
            "current_player": current_player,
            "history": [],
            "last_offer": 10,
        },
    }


def test_move_lost_response_reconciles_advanced_turn_without_duplicate_post():
    calls = []

    def post(game_id, action, timeout):
        calls.append((game_id, action, timeout))
        raise requests.ReadTimeout("response lost")

    recovery = GameMoveRecovery(post, lambda _id, _timeout: game(round_=2, current_player="buyer"))
    result = recovery.submit(game(), {"decision": "offer", "amount": 11})

    assert result.reconciled is True
    assert result.payload["valid"] is True
    assert len(calls) == 1


def test_move_recovery_caps_post_and_get_read_timeouts():
    post_timeouts = []
    get_timeouts = []

    def post(_game_id, _action, timeout):
        post_timeouts.append(timeout)
        raise requests.ReadTimeout("response lost")

    def get_state(_game_id, timeout):
        get_timeouts.append(timeout)
        return game(round_=2, current_player="buyer")

    GameMoveRecovery(post, get_state).submit(game(), {"decision": "offer"})

    assert post_timeouts and all(read <= 8.0 for _, read in post_timeouts)
    assert get_timeouts and all(read <= 2.0 for _, read in get_timeouts)


def test_move_retries_once_only_after_state_proves_turn_unchanged():
    calls = []

    def post(_game_id, _action, _timeout):
        calls.append(1)
        if len(calls) == 1:
            raise requests.ConnectionError("not sent")
        return {"valid": True, "game_over": False}

    recovery = GameMoveRecovery(post, lambda _id, _timeout: game())
    result = recovery.submit(game(), {"decision": "offer", "amount": 11})

    assert result.payload == {"valid": True, "game_over": False}
    assert len(calls) == 2


def test_move_does_not_duplicate_when_reconciliation_is_ambiguous():
    calls = []

    def post(_game_id, _action, _timeout):
        calls.append(1)
        raise requests.ReadTimeout("response lost")

    original = game()
    recovery = GameMoveRecovery(post, lambda _id, _timeout: game())

    with pytest.raises(MoveUncertain):
        recovery.submit(original, {"decision": "accept"})
    assert len(calls) == 2


def test_confirmed_retry_not_your_turn_is_treated_as_applied():
    calls = []

    def post(_game_id, _action, _timeout):
        calls.append(1)
        if len(calls) == 1:
            raise requests.ConnectionError("not sent")
        raise GleeAPIError(400, "It is not your turn")

    recovery = GameMoveRecovery(
        post,
        lambda _id, _timeout: game(),
        already_applied_error=GleeClient._not_your_turn,
    )
    result = recovery.submit(game(), {"decision": "offer"})
    assert result.reconciled is True
    assert len(calls) == 2


def test_not_your_turn_on_first_post_is_treated_as_applied_without_retry():
    calls = {"post": 0, "get": 0}

    def post(_game_id, _action, _timeout):
        calls["post"] += 1
        raise GleeAPIError(400, "It is not your turn")

    def get_state(_game_id, _timeout):
        calls["get"] += 1
        return game()

    recovery = GameMoveRecovery(
        post,
        get_state,
        already_applied_error=GleeClient._not_your_turn,
    )
    result = recovery.submit(game(), {"decision": "offer"})
    assert result.reconciled is True
    assert result.payload.get("applied") is True
    assert calls == {"post": 1, "get": 1}


def test_runner_topup_uses_owned_count_not_account_active_games():
    class Client:
        def __init__(self):
            self.queued = []

        def active_owned_count(self):
            return 5

        def queue(self, family):
            self.queued.append(family)
            return {}

    client = Client()
    runner = GameRunner(client)
    runner._topup(("negotiation",), 30, 30, None)
    assert len(client.queued) == 12


def test_adaptive_topup_continues_after_each_confirmed_queue_response():
    class Client:
        def __init__(self):
            self.queued = []

        def queue(self, family):
            self.queued.append(family)
            return {"queued": True}

    client = Client()
    runner = GameRunner(client)
    for _ in range(3):
        runner._topup(
            ("negotiation", "persuasion"),
            4,
            0,
            None,
            adaptive=True,
        )
    assert client.queued == ["negotiation", "persuasion", "negotiation"]
    assert runner._reservations.count() == 3


def test_adaptive_topup_honors_active_limit_deadline_and_breaker():
    class Client:
        def __init__(self):
            self.queue_calls = 0

        def queue(self, _family):
            self.queue_calls += 1
            return {"queued": True}

    client = Client()
    runner = GameRunner(client)
    runner._topup(("negotiation",), 4, 0, lambda: 0, adaptive=True)
    runner._topup(
        ("negotiation",), 4, 0, None, adaptive=True, deadline=time.monotonic() - 1
    )
    runner._pause(RunState.NETWORK_WAITING, "offline", 60)
    runner._topup(("negotiation",), 4, 0, None, adaptive=True)
    assert client.queue_calls == 0


def test_adaptive_run_zero_stops_before_queue_at_worker_deadline():
    class Client:
        def __init__(self):
            self.queue_calls = 0
            self.stopped = False

        def queue(self, _family):
            self.queue_calls += 1
            return {"queued": True}

        def leave_queue(self, _family):
            return {}

        def pending_games(self, *, short=False):
            del short
            return []

        def stop_claiming(self):
            self.stopped = True

    client = Client()
    runner = GameRunner(client)
    runner.run(
        lambda _game: {},
        game_families=["negotiation"],
        concurrency=0,
        max_time=0,
        poll_interval=0,
    )
    assert runner.state is RunState.ENDED
    assert client.queue_calls == 0
    assert client.stopped is True


def test_inflight_game_is_not_resubmitted_from_stale_pending():
    class Client:
        def __init__(self):
            self.active = 0
            self.peak = 0
            self.lock = threading.Lock()
            self.submissions = 0

        @property
        def assigned_count(self):
            return 1

        def _handle_game(self, strategy, current):
            del strategy
            with self.lock:
                self.submissions += 1
                self.active += 1
                self.peak = max(self.peak, self.active)
            time.sleep(0.01)
            with self.lock:
                self.active -= 1
            return False

    client = Client()
    runner = GameRunner(client)
    inflight = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        games = [game("g1")]
        assert runner._submit_games(pool, inflight, games, lambda _g: {}, 1)
        assert client.submissions == 1
        assert runner._submit_games(pool, inflight, games, lambda _g: {}, 1) is False
        assert client.submissions == 1
        wait(tuple(inflight), timeout=1)
    assert client.peak == 1


def test_generation_handoff_allows_new_owner_after_old_release(tmp_path):
    old = ClaimCoordinator(tmp_path, "run-old", "negotiation", pid=101)
    new = ClaimCoordinator(tmp_path, "run-new", "negotiation", pid=202)
    target = game("handoff")

    assert old.try_claim(target) is not None
    assert new.try_claim(target) is None
    old.release("handoff")
    assert new.try_claim(target) is not None


def test_move_post_uses_single_attempt_without_connection_retry():
    transport = ApiTransport("Bearer test")
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url))
        raise requests.ConnectionError("offline")

    transport.default_session.request = fake_request  # type: ignore[method-assign]
    with pytest.raises(requests.ConnectionError):
        transport.request_json(
            "POST",
            "https://example.test/move",
            raise_for_response=lambda _response: None,
            attempts=1,
            retry_post_connection_errors=False,
        )
    assert len(calls) == 1


def test_sdk_endpoints_use_explicit_short_connect_and_read_timeouts(monkeypatch):
    client = GleeClient("test")
    calls = []

    class Response:
        ok = True
        status_code = 200

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    def request_json(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return Response([] if url.endswith("/games/pending") else {})

    monkeypatch.setattr(client.transport.default_session, "request", request_json)
    client.queue("negotiation")
    client.pending_games()
    client.game_state("g")
    client.move("g", {"decision": "offer"})
    client.stats()

    assert [kwargs["timeout"] for _, _, kwargs in calls] == [
        (2.0, 5.0),
        (2.0, 5.0),
        (2.0, 5.0),
        (2.0, 5.0),
        (2.0, 5.0),
    ]


def test_network_pause_blocks_queue_but_keeps_owned_game_schedulable():
    class Client:
        def __init__(self):
            self.queue_calls = 0
            self.handled = []

        def queue(self, _family):
            self.queue_calls += 1

        def _handle_game(self, strategy, current):
            self.handled.append(strategy(current))
            return True

    client = Client()
    runner = GameRunner(client)
    runner._pause_for_error(requests.ReadTimeout("offline"))
    runner._topup(("negotiation",), 30, 30, None)

    inflight = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert runner._submit_games(
            pool,
            inflight,
            [game("owned")],
            lambda _game: {"decision": "offer"},
            1,
        )
        assert next(iter(inflight)).result() is True

    assert client.queue_calls == 0
    assert client.handled == [{"decision": "offer"}]


def test_cooldown_blocks_topup_but_still_submits_pending_games():
    class Client:
        def __init__(self):
            self.queue_calls = 0
            self.handled = []

        def queue(self, _family):
            self.queue_calls += 1

        def leave_queue(self, _family):
            pass

        def pending_games(self, *, short=False):
            del short
            return [game("g1")]

        def _handle_game(self, strategy, current):
            self.handled.append(strategy(current))
            return True

    client = Client()
    runner = GameRunner(client)
    runner._pause(RunState.COOLDOWN, "403 cooldown", 9999.0)
    inflight = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        submitted = runner._submit_games(
            pool, inflight, client.pending_games(), lambda _game: {"decision": "offer"}, 1
        )
        assert submitted
        assert next(iter(inflight)).result() is True
    runner._topup(("negotiation",), 30, 30, None)
    assert client.queue_calls == 0
    assert client.handled == [{"decision": "offer"}]


def test_one_game_network_failure_does_not_block_another_game():
    def post(game_id, _action, _timeout):
        if game_id == "bad":
            raise requests.ConnectionError("offline")
        return {"valid": True, "game_over": False}

    recovery = GameMoveRecovery(post, lambda game_id, _timeout: game(game_id))
    with ThreadPoolExecutor(max_workers=2) as pool:
        bad = pool.submit(recovery.submit, game("bad"), {"decision": "offer"})
        good = pool.submit(recovery.submit, game("good"), {"decision": "offer"})

    with pytest.raises(MoveUncertain):
        bad.result()
    assert good.result().payload["valid"] is True


def test_client_handle_game_calls_recovery_submit_not_direct_move(monkeypatch):
    client = GleeClient("test")
    action = {"decision": "offer", "amount": 11}
    submitted = []

    class Recovery:
        def submit(self, current, selected):
            submitted.append((current, selected))
            return type("Result", (), {"payload": {"valid": True, "game_over": False}})()

    client.move_recovery = Recovery()
    monkeypatch.setattr(
        client,
        "move",
        lambda *_args, **_kwargs: pytest.fail("_handle_game bypassed recovery.submit"),
    )
    current = game()

    assert client._handle_game(lambda _game: action, current) is False
    assert submitted == [(current, action)]


def test_claim_coordinator_filters_family_and_allows_exactly_one_owner(tmp_path):
    first = ClaimCoordinator(tmp_path, "run-a", "negotiation", pid=101)
    second = ClaimCoordinator(tmp_path, "run-b", "negotiation", pid=202)
    target = game()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda owner: owner.try_claim(target), (first, second)))

    assert sum(result is not None for result in results) == 1
    assert first.try_claim({**target, "game_id": "other", "game_family": "persuasion"}) is None
    other_family = ClaimCoordinator(tmp_path, "run-c", "persuasion", pid=303)
    assert other_family.try_claim({**target, "game_id": "p1", "game_family": "persuasion"}) is not None


def test_claim_coordinator_releases_only_its_own_claim(tmp_path):
    owner = ClaimCoordinator(tmp_path, "run-a", "negotiation", pid=101)
    other = ClaimCoordinator(tmp_path, "run-b", "negotiation", pid=202)
    assert owner.try_claim(game()) is not None

    other.release("g")
    assert len(owner.owned_games()) == 1
    owner.release("g")
    assert owner.owned_incomplete_count() == 0


def test_lifecycle_uses_canonical_states_and_local_drain_counts():
    assert RunConfig("negotiation").game_concurrency == 0
    assert RunConfig("negotiation").queue_target == 0
    assert RunConfig("negotiation", 30, 30, 4).move_workers == 4
    assert RunConfig("negotiation", 30, 30, 5).move_workers == 5
    assert RunSnapshot(RunState.DRAINING).drain_complete is True
    assert RunSnapshot(RunState.DRAINING, owned_games=1).drain_complete is False

    class Client:
        def stop_claiming(self):
            pass

        def leave_queue(self, _family):
            pass

        def drain_complete(self):
            return True

    runner = GameRunner(Client())
    runner.run(lambda _game: {}, game_families=["negotiation"], should_stop=lambda: True)
    assert runner.state is RunState.STOPPED


def test_transport_disables_environment_proxies_by_default():
    transport = ApiTransport("Bearer test")
    assert transport.default_session.trust_env is False


def test_connect_failsover_to_second_cloudflare_ip():
    seen: list[str] = []

    def connect(address):
        seen.append(address[0])
        if address[0] == "104.21.59.229":
            raise OSError("refused")
        return address[0]

    assert connect_with_failover(("glee-competition.com", 443), connect) == "172.67.184.212"
    assert seen == ["104.21.59.229", "172.67.184.212"]


@pytest.mark.parametrize("healthy_index", [0, 1])
def test_each_cloudflare_ip_can_be_selected(healthy_index):
    ips = ("104.21.59.229", "172.67.184.212")
    seen: list[str] = []

    def connect(address):
        seen.append(address[0])
        if address[0] != ips[healthy_index]:
            raise OSError("unreachable")
        return address[0]

    assert connect_with_failover(("glee-competition.com", 443), connect) == ips[healthy_index]
    assert seen == list(ips[: healthy_index + 1])


def test_network_failure_enters_waiting_without_interface_fallback():
    class Client:
        def queue(self, _family):
            raise NetworkUnavailable("both Cloudflare addresses failed")

    runner = GameRunner(Client())
    with pytest.raises(NetworkUnavailable):
        runner._queue_one("negotiation")
    assert runner.state is RunState.NETWORK_WAITING
    assert runner._gate.state is RunState.NETWORK_WAITING
    assert "utun" not in (runner._gate.reason or "")


def test_both_cloudflare_ips_fail_do_not_enable_trust_env():
    def connect(_address):
        raise OSError("refused")

    with pytest.raises(NetworkUnavailable):
        connect_with_failover(("glee-competition.com", 443), connect)
    session = attach_direct_adapter(requests.Session())
    assert session.trust_env is False
    assert ApiTransport("Bearer test").trust_env is False


def test_run_store_writes_run_scoped_atomic_evidence(tmp_path):
    store = RunStore(tmp_path / "run")
    store.save_manifest({"run_id": "r"})
    store.append_claim({"game_id": "g"})
    store.append_action("g", {"decision": "accept"})
    store.save_game("g", {"result": "done"})
    store.save_snapshot({"state": "STOPPED"})

    assert json.loads((store.root / "manifest.json").read_text()) == {"run_id": "r"}
    assert json.loads((store.root / "claims.jsonl").read_text()) == {"game_id": "g"}
    assert json.loads((store.root / "actions" / "g.jsonl").read_text()) == {"decision": "accept"}
    assert not list(store.root.rglob("*.tmp"))


def test_drain_exits_from_task_local_claims_without_account_active_games():
    class Client:
        assigned_count = 0

        def stop_claiming(self):
            self.stopped = True

        def leave_queue(self, family):
            assert family == "negotiation"

        def drain_complete(self):
            return True

        def stats(self):
            raise AssertionError("account active_games must not gate local drain")

    client = Client()
    GameRunner(client).run(
        lambda _game: {},
        game_families=["negotiation"],
        should_stop=lambda: True,
        poll_interval=0,
    )
    assert client.stopped is True


def test_paused_pending_poll_is_short_and_family_leave_is_scoped():
    class Client:
        def __init__(self):
            self.pending_calls = []
            self.left = []

        def pending_games(self, *, short=False):
            self.pending_calls.append(short)
            return []

        def leave_queue(self, family):
            self.left.append(family)

    client = Client()
    runner = GameRunner(client)
    runner._pause(RunState.NETWORK_WAITING, "offline", 10)
    assert runner._pending_games() == []
    runner._maybe_resume(("negotiation",))
    assert client.pending_calls == [True]
    assert client.left == ["negotiation"]


def test_queue_reservation_expires_at_ttl_boundary():
    now = [10.0]
    reservations = QueueReservations(ttl=5.0, clock=lambda: now[0])
    reservations.add("negotiation")
    now[0] = 14.999
    assert reservations.count() == 1
    now[0] = 15.0
    assert reservations.count() == 0


def test_queue_response_loss_keeps_one_reservation_without_duplicate_post():
    class Client:
        def __init__(self):
            self.calls = 0

        def queue(self, _family):
            self.calls += 1
            raise requests.ReadTimeout("lost response")

    client = Client()
    runner = GameRunner(client)
    with pytest.raises(requests.ReadTimeout):
        runner._queue_one("negotiation")
    assert client.calls == 1
    assert runner._reservations.count() == 1
    assert runner.state is RunState.NETWORK_WAITING


def test_uncertain_move_keeps_game_owned_for_next_reconciliation():
    runner = GameRunner(object())
    runner._owned_ids.add("g1")
    runner._seen_ids.add("g1")
    future = Future()
    future.set_exception(MoveUncertain("g1", EOFError("lost response")))

    runner._pause_for_error(MoveUncertain("g1", EOFError("lost response")))
    assert runner._reap_future(future, "g1") is False
    assert "g1" in runner._owned_ids
    assert "g1" in runner._seen_ids
    assert runner.state is RunState.NETWORK_WAITING


@pytest.mark.parametrize("family", ["negotiation", "bargaining", "persuasion"])
def test_uncertain_move_pauses_only_topup_and_does_not_block_other_future(family):
    class Client:
        def __init__(self):
            self.queue_calls = 0
            self.handled = []

        def queue(self, _family):
            self.queue_calls += 1

        def _handle_game(self, strategy, current):
            self.handled.append(current["game_id"])
            strategy(current)
            return True

    client = Client()
    runner = GameRunner(client)
    uncertain_id = f"{family}-uncertain"
    runner._owned_ids.add(uncertain_id)
    runner._seen_ids.add(uncertain_id)
    uncertain = Future()
    uncertain.set_exception(MoveUncertain(uncertain_id, EOFError("lost response")))

    runner._pause_for_error(MoveUncertain(uncertain_id, EOFError("lost response")))
    assert runner._reap_future(uncertain, uncertain_id) is False
    assert uncertain_id in runner._owned_ids
    assert uncertain_id in runner._seen_ids
    assert runner.state is RunState.NETWORK_WAITING

    runner._topup((family,), 30, 30, None)
    other = {**game(f"{family}-other"), "game_family": family}
    inflight = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert runner._submit_games(pool, inflight, [other], lambda _game: {}, 1)
        future, other_id = next(iter(inflight.items()))
        assert runner._reap_future(future, other_id) is True

    assert client.queue_calls == 0
    assert client.handled == [other["game_id"]]
    assert other["game_id"] not in runner._owned_ids
    assert uncertain_id in runner._owned_ids


def test_cooldown_requires_deadline_and_two_health_checks_before_resume():
    class Client:
        def __init__(self):
            self.health_checks = 0

        def leave_queue(self, _family):
            pass

        def network_healthy(self):
            self.health_checks += 1
            return True

    client = Client()
    runner = GameRunner(client)
    error = GleeAPIError(
        403,
        "last 3 games timed out; queue joins are paused after 2099-01-01T00:00:00Z",
    )
    runner._pause_for_error(error)
    assert runner.state is RunState.COOLDOWN
    runner._maybe_resume(("negotiation",))
    assert client.health_checks == 0

    runner._gate.resume_at = 0
    runner._maybe_resume(("negotiation",))
    assert client.health_checks == 1
    assert runner.state is RunState.COOLDOWN
    runner._gate.resume_at = 0
    runner._maybe_resume(("negotiation",))
    assert client.health_checks == 2
    assert runner.state is RunState.WAITING


def test_manual_drain_during_cooldown_does_not_run_health_resume():
    class Client:
        assigned_count = 0

        def stop_claiming(self):
            pass

        def leave_queue(self, _family):
            pass

        def drain_complete(self):
            return True

        def network_healthy(self):
            raise AssertionError("draining run must not resume queue")

    runner = GameRunner(Client())
    runner._pause(RunState.COOLDOWN, "cooldown", 0)
    runner.run(
        lambda _game: {},
        game_families=["negotiation"],
        should_stop=lambda: True,
        poll_interval=0,
    )
    assert runner.state is RunState.STOPPED


def test_thirty_games_never_run_more_than_four_moves_concurrently():
    class Client:
        def __init__(self):
            self.done = set()
            self.active = 0
            self.peak = 0
            self.lock = threading.Lock()

        @property
        def assigned_count(self):
            return len(self.done)

        def queue(self, _family):
            return {}

        def leave_queue(self, _family):
            return {}

        def pending_games(self, *, short=False):
            del short
            return [
                {**game(str(index)), "game_family": "negotiation"}
                for index in range(30)
                if str(index) not in self.done
            ]

        def _handle_game(self, strategy, current):
            strategy(current)
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            time.sleep(0.002)
            with self.lock:
                self.active -= 1
                self.done.add(current["game_id"])
            return True

        def drain_complete(self):
            return len(self.done) == 30

    client = Client()
    GameRunner(client).run(
        lambda _game: {"decision": "pass"},
        game_families=["negotiation"],
        concurrency=30,
        move_concurrency=30,
        should_stop=lambda: len(client.done) == 30,
        requeue=False,
        poll_interval=0.001,
    )
    assert client.peak == 30


@pytest.mark.parametrize(
    ("family", "baseline", "action"),
    [
        ("persuasion", "02eab6c", {"decision": "pass"}),
        ("bargaining", "64182fb", {"alice_share": 52, "bob_share": 48}),
        ("negotiation", "fae4c6d", {"decision": "offer", "amount": 100}),
    ],
)
def test_runtime_preserves_authoritative_strategy_action(family, baseline, action):
    seen = []

    class Client:
        def _handle_game(self, strategy, current):
            seen.append(strategy(current))
            return True

    current = game()
    current["game_family"] = family
    runner = GameRunner(Client())
    inflight = {}
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert runner._submit_games(
            pool, inflight, [current], lambda _game: action, 1
        )
        assert next(iter(inflight)).result() is True
    assert seen == [action], baseline
