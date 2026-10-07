"""Parallel closed-loop CARLA evaluation.

Runs one Bench2Drive leaderboard process (which starts its own CARLA server) per
GPU slot, one route XML at a time, retries routes that failed for
infrastructure reasons, and skips routes whose result is already complete.

    python tools/run_distributed_eval.py --routes-dir benchmark/data/bench2drive \\
        --output-dir work_dirs/eval/bench2drive --agent team_code/agent_safedrive.py \\
        --agent-config <checkpoint> --gpu-ids 0,1,2,3,4,5,6,7

Outputs: ``<output-dir>/res/<route>_res.json`` (leaderboard records),
``viz/<route>/`` (per-tick ego state, optional images), ``out/`` and ``err/`` logs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH2DRIVE_ROOT = REPO_ROOT / "third_party" / "bench2drive"
INFRA_FAILURES = ("Simulation crashed", "Agent's sensors were invalid", "Agent couldn't be set up", "Agent crashed")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def result_is_complete(res_path: Path) -> bool:
    """The route finished (any driving outcome) without an infrastructure failure."""
    try:
        data = json.loads(res_path.read_text())
        progress = data["_checkpoint"]["progress"]
        records = data["_checkpoint"].get("records", [])
    except Exception:
        return False
    if len(progress) < 2 or progress[0] < progress[1] or not records:
        return False
    return not any(token in rec.get("status", "") for rec in records for token in INFRA_FAILURES)


def _carla_processes() -> List[tuple]:
    """(pid, rpc port) of the running CARLA servers."""
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,args="], text=True)
    except Exception:
        return []
    procs = []
    for line in out.splitlines():
        match = re.search(r"-carla-rpc-port=(\d+)", line)
        if "CarlaUE4" in line and match:
            procs.append((int(line.split(None, 1)[0]), int(match.group(1))))
    return procs


@dataclass
class Slot:
    gpu: int
    port: int
    tm_port: int
    proc: Optional[subprocess.Popen] = None
    route: Optional[Path] = None
    started: float = 0.0
    carla_missing_since: Optional[float] = None


class Runner:
    def __init__(self, args, routes: List[Path]):
        self.args = args
        self.queue = list(routes)
        self.retries = {}
        self.dirs = {name: args.output_dir / name for name in ("res", "out", "err", "viz")}
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        gpus = [int(g) for g in args.gpu_ids.split(",")]
        self.slots = [Slot(gpu=g, port=args.base_port + i * 100, tm_port=args.base_tm_port + i * 100) for i, g in enumerate(gpus)]
        self.env = dict(os.environ)
        carla_api = os.path.join(os.environ.get("CARLA_ROOT", ""), "PythonAPI", "carla")
        self.env["PYTHONPATH"] = os.pathsep.join(filter(None, [
            carla_api, str(REPO_ROOT), str(BENCH2DRIVE_ROOT / "leaderboard"),
            str(BENCH2DRIVE_ROOT / "scenario_runner"), os.environ.get("PYTHONPATH", ""),
        ]))
        self.env["WORK_DIR"] = str(BENCH2DRIVE_ROOT)  # weather presets of the leaderboard
        # The leaderboard discovers the scenario classes under this root;
        # without it every scenario of a route is silently skipped.
        self.env["SCENARIO_RUNNER_ROOT"] = str(BENCH2DRIVE_ROOT / "scenario_runner")

    def launch(self, slot: Slot, route: Path) -> None:
        res = self.dirs["res"] / f"{route.stem}_res.json"
        res.unlink(missing_ok=True)
        viz = self.dirs["viz"] / route.stem
        viz.mkdir(parents=True, exist_ok=True)
        env = dict(self.env, CUDA_VISIBLE_DEVICES=str(slot.gpu), SAVE_PATH=str(viz), ROUTES=str(route))
        cmd = [
            sys.executable, "-u", str(BENCH2DRIVE_ROOT / "leaderboard" / "leaderboard" / "leaderboard_evaluator.py"),
            f"--routes={route}", "--repetitions=1", "--track=SENSORS", f"--checkpoint={res}",
            f"--debug-checkpoint={self.dirs['out'] / f'{route.stem}_live.txt'}",
            f"--timeout={self.args.timeout}", f"--agent={self.args.agent}", f"--agent-config={self.args.agent_config}",
            f"--traffic-manager-seed={self.args.seed}", f"--port={slot.port}",
            f"--traffic-manager-port={slot.tm_port}", f"--gpu-rank={slot.gpu}",
        ]
        log(f"START  gpu={slot.gpu} port={slot.port} {route.stem}")
        with open(self.dirs["out"] / f"{route.stem}_out.log", "w") as out, open(self.dirs["err"] / f"{route.stem}_err.log", "w") as err:
            slot.proc = subprocess.Popen(cmd, stdout=out, stderr=err, env=env, cwd=REPO_ROOT, start_new_session=True)
        slot.route, slot.started, slot.carla_missing_since = route, time.monotonic(), None

    def _kill(self, slot: Slot, reason: str) -> None:
        log(f"KILL   gpu={slot.gpu} {slot.route.stem}: {reason}")
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(slot.proc.pid, sig)
            except ProcessLookupError:
                break
            try:
                slot.proc.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                continue

    @staticmethod
    def _reap_carla(slot: Slot) -> None:
        # The leaderboard starts CARLA in its own session; reap it by port.
        for pid, port in _carla_processes():
            if slot.port <= port < slot.port + 100:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def release(self, slot: Slot) -> None:
        self._reap_carla(slot)
        route = slot.route
        if not result_is_complete(self.dirs["res"] / f"{route.stem}_res.json"):
            attempt = self.retries.get(route, 0)
            if attempt < self.args.max_retries:
                self.retries[route] = attempt + 1
                self.queue.append(route)
                log(f"RETRY  {route.stem} ({attempt + 1}/{self.args.max_retries})")
            else:
                log(f"FAILED {route.stem}")
        else:
            log(f"DONE   {route.stem}")
        slot.proc, slot.route = None, None

    def poll(self) -> None:
        """Release finished slots; kill slots past the route walltime or whose
        CARLA server disappeared."""
        ports = {port for _, port in _carla_processes()}
        now = time.monotonic()
        for slot in self.slots:
            if slot.proc is None:
                continue
            running = slot.proc.poll() is None
            if running:
                elapsed = now - slot.started
                if elapsed > self.args.route_walltime:
                    self._kill(slot, "route walltime exceeded")
                    running = False
                elif elapsed > self.args.carla_grace and slot.port not in ports:
                    slot.carla_missing_since = slot.carla_missing_since or now
                    if now - slot.carla_missing_since > self.args.carla_grace:
                        self._kill(slot, "CARLA server is gone")
                        running = False
                else:
                    slot.carla_missing_since = None
            if not running:
                self.release(slot)

    def run(self) -> None:
        while self.queue or any(s.proc is not None for s in self.slots):
            self.poll()
            for slot in self.slots:
                if slot.proc is None and self.queue:
                    self.launch(slot, self.queue.pop(0))
            time.sleep(2)

    def kill_all(self) -> None:
        for slot in self.slots:
            if slot.proc is not None:
                self._kill(slot, "interrupted")
            self._reap_carla(slot)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--routes-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--agent", type=Path, required=True)
    p.add_argument("--agent-config", required=True, help="checkpoint passed to the agent")
    p.add_argument("--gpu-ids", default="0")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--base-port", type=int, default=2000)
    p.add_argument("--base-tm-port", type=int, default=2500)
    p.add_argument("--timeout", type=int, default=1800, help="leaderboard agent/simulator timeout (s)")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--route-walltime", type=float, default=3600, help="kill a route after this many seconds")
    p.add_argument("--carla-grace", type=float, default=600, help="seconds before a missing CARLA server counts as a crash")
    args = p.parse_args()

    routes = sorted(args.routes_dir.glob("*.xml"))
    pending = [r for r in routes if not result_is_complete(args.output_dir / "res" / f"{r.stem}_res.json")]
    log(f"{len(routes)} routes, {len(pending)} pending, {len(routes) - len(pending)} already complete")
    runner = Runner(args, pending)

    def on_signal(signum, frame):
        runner.kill_all()
        sys.exit(130)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    runner.run()
    log("all routes finished")


if __name__ == "__main__":
    main()
