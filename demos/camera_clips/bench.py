"""Security camera clips through an EmbeddingGemma 2 typed-joint checkpoint: flide against the release's own runner.

    python demos/camera_clips/bench.py --checkpoint /path/to/model/joint --clips /path/to/clips --out bench.json

Both run in this process on the same GPU, one clip at a time and in turn (the order swaps every round), so whatever
else the machine is doing falls on both. Per clip: the wall time of a whole decision (decode the video, sample and
preprocess its frames, encode, score), and the part of it spent before the model. Then all clips as fast as each side
takes them: the release one after another (it has no other way), flide one after another and through its serving
batcher, which decodes on `--planners` threads while the GPU works. `--cpu-clips N` also times the release on the CPU
with `--cpu-threads` threads, the configuration it runs in when no GPU is given to it.

The release's code is read from gemma_decision/ beside the checkpoint (its repository layout).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import sys
import time
from pathlib import Path

STATE = "Security camera video of a residential property."


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def probs(a) -> dict:
    return a["probabilities"] if isinstance(a, dict) else a.probabilities


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="the typed-joint directory (config.json, head.safetensors, backbone/)")
    p.add_argument("--clips", required=True, nargs="+", help="video files, or directories searched for *.mp4")
    p.add_argument("--questions", default=str(Path(__file__).with_name("questions.json")))
    p.add_argument("--limit", type=int, default=24, help="clips to use")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--planners", type=int, default=4)
    p.add_argument("--threads", type=int, default=8, help="torch CPU threads, for both sides")
    p.add_argument("--cpu-clips", type=int, default=0)
    p.add_argument("--cpu-threads", type=int, default=8)
    p.add_argument("--exclude", default="_4k", help="skip files whose name contains this")
    p.add_argument("--out")
    args = p.parse_args(argv)

    import torch

    from flide import Engine
    from flide.api.schema import SystemOneRequest
    from flide.runtime.online import Batcher
    from flide_embeddinggemma import backbone_dir

    torch.set_num_threads(args.threads)
    root = Path(args.checkpoint).resolve()
    sys.path.insert(0, str(root.parent))
    from gemma_decision.typed import load_typed_model

    clips = []
    for c in args.clips:
        c = Path(c)
        clips += sorted(x for x in c.rglob("*.mp4")) if c.is_dir() else [c]
    clips = [str(c) for c in clips if not (args.exclude and args.exclude in Path(c).name)][: args.limit]
    if not clips:
        raise SystemExit("no clips")
    questions = json.loads(Path(args.questions).read_text())
    records = [{"state": STATE, "videos": [c], "questions": questions} for c in clips]
    requests = [SystemOneRequest.model_validate(r) for r in records]
    sync = torch.cuda.synchronize

    def load_native(device):
        m = load_typed_model(str(root), device=device, local_only=True, media_route="typed")
        m.backbone.max_frames = args.max_frames
        spent = {"pre": 0.0}
        inner = m.backbone.media_inputs

        def media_inputs(*a, **k):
            t = time.perf_counter()
            out = inner(*a, **k)
            spent["pre"] += time.perf_counter() - t
            return out

        m.backbone.media_inputs = media_inputs
        return m, spent

    native, spent = load_native("cuda")
    engine = Engine(backbone_dir(str(root)), max_frames=args.max_frames, backbone_options={"cuda_graphs": True})

    def run_native(i):
        spent["pre"] = 0.0
        sync()
        t = time.perf_counter()
        out = native.decide(records[i])
        sync()
        return out["answers"], time.perf_counter() - t, spent["pre"]

    def run_flide(i):
        sync()
        t = time.perf_counter()
        resp, meta = engine.decide(requests[i])
        sync()
        return resp.answers, time.perf_counter() - t, meta["timing"]["plan_s"]

    for i in range(min(3, len(clips))):
        run_native(i)
        run_flide(i)
    lat = {"native": [], "flide": []}
    pre = {"native": [], "flide": []}
    answers = {"native": {}, "flide": {}}
    for r in range(args.rounds):
        for i in range(len(clips)):
            order = (("native", run_native), ("flide", run_flide))
            for name, fn in (order if (r + i) % 2 == 0 else order[::-1]):
                a, dt, dp = fn(i)
                lat[name].append(dt)
                pre[name].append(dp)
                answers[name][i] = a
    same, worst, tokens = 0, 0.0, 0
    for i in range(len(clips)):
        for q in questions:
            n, f = answers["native"][i][q], answers["flide"][i][q]
            if questions[q]["type"] == "choice":
                same += n["choice"] == f.choice
                worst = max(worst, max(abs(probs(n)[k] - probs(f)[k]) for k in probs(n)))
            else:
                same += 1
    n_answers = len(clips) * len(questions)

    def through_native():
        sync()
        t = time.perf_counter()
        for r in records:
            native.decide(r)
        sync()
        return len(records) / (time.perf_counter() - t)

    def through_flide():
        sync()
        t = time.perf_counter()
        for r in requests:
            engine.decide(r)
        sync()
        return len(requests) / (time.perf_counter() - t)

    batcher = Batcher(engine, planners=args.planners)

    def one(req):
        plan = batcher.plan(req).result()
        branches, _ = batcher.forward(plan).result()
        return engine.format.finish(req, plan, branches)[0]

    def through_batcher():
        sync()
        t = time.perf_counter()
        with cf.ThreadPoolExecutor(args.planners + 2) as pool:
            out = list(pool.map(one, requests))
        sync()
        assert len(out) == len(requests)
        return len(requests) / (time.perf_counter() - t)

    through_batcher()
    rate = {"native": [], "flide": [], "flide_batcher": []}
    for _ in range(max(2, args.rounds)):
        rate["native"].append(through_native())
        rate["flide"].append(through_flide())
        rate["flide_batcher"].append(through_batcher())

    cpu = None
    if args.cpu_clips:
        torch.set_num_threads(args.cpu_threads)
        native_cpu, _ = load_native("cpu")
        native_cpu.decide(records[0])
        ts = []
        for r in records[: args.cpu_clips]:
            t = time.perf_counter()
            native_cpu.decide(r)
            ts.append(time.perf_counter() - t)
        cpu = {"clips": len(ts), "threads": args.cpu_threads, "median_ms": med(ts) * 1e3}

    util = None
    try:
        import subprocess

        util = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    ms = lambda xs: round(med(xs) * 1e3, 1)
    result = {
        "gpu": torch.cuda.get_device_name(0), "gpu_at_end": util, "torch": torch.__version__, "clips": len(clips),
        "rounds": args.rounds, "max_frames": args.max_frames, "threads": args.threads,
        "tokens_per_clip": engine.format.plan(requests[0]).tokens,
        "latency_ms": {k: {"median": ms(lat[k]), "before_model": ms(pre[k]),
                           "model": round(ms(lat[k]) - ms(pre[k]), 1),
                           "p10": round(sorted(lat[k])[len(lat[k]) // 10] * 1e3, 1),
                           "p90": round(sorted(lat[k])[len(lat[k]) * 9 // 10] * 1e3, 1)} for k in lat},
        "clips_per_s": {k: round(max(v), 2) for k, v in rate.items()},
        "clips_per_s_runs": {k: [round(x, 2) for x in v] for k, v in rate.items()},
        "planners": args.planners,
        "agreement": {"same_choice": same, "answers": n_answers, "largest_probability_difference": round(worst, 4)},
        "native_cpu": cpu,
    }
    print(json.dumps(result, indent=2))
    L = result["latency_ms"]
    print(f"\none clip: release {L['native']['median']} ms, flide {L['flide']['median']} ms "
          f"({L['native']['median'] / L['flide']['median']:.2f}x); model part {L['native']['model']} ms against "
          f"{L['flide']['model']} ms ({L['native']['model'] / L['flide']['model']:.2f}x)")
    R = result["clips_per_s"]
    print(f"all clips: release {R['native']} clips/s, flide {R['flide']} one at a time, {R['flide_batcher']} through "
          f"the batcher ({R['flide_batcher'] / R['native']:.2f}x)")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
