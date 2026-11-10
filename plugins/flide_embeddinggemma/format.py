"""EmbeddingGemma 2 typed joint decisions (the "typed-joint" member of
Danny-Dasilva/embeddinggemma-2-multimodal-typed-decision): the release's gemma_decision/joint.py.

The checkpoint is a directory with config.json ("format": "typed-joint"), head.safetensors and backbone/, the
fine-tuned EmbeddingGemma 2 encoder. The engine loads the encoder, so it is given the backbone directory:

    flide serve --model /path/to/joint/backbone

A request is one sequence, tokenized as JointEncoder.packs tokenizes it: the task prefix and the rendered state (a
request's images and videos after it, through the base model's processor: 280 soft tokens an image, 140 a video
frame, frames sampled at `fps` up to `max_frames`), then every question and each of its options as segments of their
own, then <eos>. Questions that do not fit one sequence go into further sequences that repeat the state; a text state
longer than one sequence is read in overlapping windows, every question on every window, and a question's logits are
pooled over the windows ("windowed" long mode).

The encoder reads each sequence both ways and the head reads its final hidden states at every position (kept on the
device): span means of the state, each question and each option feed the typed head, and every option also
cross-attends over all positions (JointDecisionHead.forward). The answer is the softmax over a question's options.

Videos are decoded with PyAV as the release decodes them (the same frames), on the decoder's own threads.
"""
from __future__ import annotations

import io
import json
import logging
import math
import threading
from pathlib import Path
from typing import Any

from flide.api.answers import make_choice, make_noul, make_score
from flide.api.schema import ChoiceQuestion, NoulQuestion, SystemOneRequest, SystemOneResponse, Usage
from flide.formats import register
from flide.formats.base import Format, Plan, Read, media_bytes

log = logging.getLogger("flide.embeddinggemma")

READ = "hidden"
PREFIX = "task: classification | query: "
MARKERS = ("<|image|>", "<|video|>", "<|audio|>")
TYPES = {"choice": 0, "noul": 1, "score": 2}
NOUL = {"false": "No. The statement is false.", "true": "Yes. The statement is true."}
DEFAULT_NOUL = {"Yes", "No", "Yes. The statement is true.", "No. The statement is false."}
OVERLAP = 256


def checkpoint(model_dir: str) -> Path | None:
    """The typed-joint checkpoint a backbone directory belongs to (its parent), or None."""
    root = Path(model_dir).resolve().parent
    cfg = root / "config.json"
    if not cfg.is_file() or not (root / "head.safetensors").is_file():
        return None
    try:
        with cfg.open() as f:
            # The config lists its training hashes after these keys and runs to megabytes; the format is at its start.
            if '"typed-joint"' not in f.read(4096):
                return None
    except OSError:
        return None
    return root


def backbone_dir(path: str) -> str:
    """The directory to give the engine for a checkpoint named by its root, its joint member or its backbone."""
    p = Path(path)
    for d in (p, p / "backbone", p / "joint" / "backbone"):
        if (d / "model.safetensors").is_file() and checkpoint(str(d)) is not None:
            return str(d)
    raise ValueError(f"{path} holds no typed-joint EmbeddingGemma 2 checkpoint")


def text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


def render_state(state: Any, workflow: Any = "") -> str:
    """gemma_decision.typed.render_state: a JSON object state as one "path: value" line per leaf."""
    state = "" if state is None else state
    if isinstance(state, str) and state.strip()[:1] in ("{", "["):
        try:
            state = json.loads(state)
        except ValueError:
            pass
    if not isinstance(state, dict):
        return f"workflow: {workflow} | state: {state}"
    lines = []

    def walk(prefix, value):
        if isinstance(value, dict):
            for key, item in value.items():
                walk(f"{prefix} · {key}" if prefix else key, item)
        else:
            lines.append(f"{prefix.replace('_', ' ')}: {value}")

    walk("", state)
    return f"workflow: {workflow}\n" + "\n".join(lines)


def options(q) -> list[tuple[str, str]]:
    """Each option's id and description, in the order the head scores them (schema.options over
    render_external_questions)."""
    if isinstance(q, NoulQuestion):
        c = dict(NOUL)
        if q.criteria is not None:
            given = q.criteria.model_dump(exclude_unset=True)
            if given:
                c = {k: text(given[k]) if given.get(k) is not None else ("Yes" if k == "true" else "No")
                     for k in ("false", "true")}
        return [("false", c["false"]), ("true", c["true"])]
    if isinstance(q, ChoiceQuestion):
        return [(str(k), text(k if v is None else v)) for k, v in q.criteria.items()]
    return [(str(i), text(v)) for i, v in enumerate(q.criteria)]


def segments(name: str, q, criteria: bool) -> tuple[str, list[str]]:
    """The question's text and one text per option (joint.segments, or segments_criteria with `criteria`: a noul
    option also shows a description the request gave)."""
    instruction = q.instructions
    instruction = name if instruction is None or instruction == "" else text(instruction)
    opts = options(q)
    out = []
    for key, desc in opts:
        if q.type == "noul":
            word = "no" if key == "false" else "yes"
            out.append(f"\nanswer: {word}: {desc}" if criteria and desc not in DEFAULT_NOUL else f"\nanswer: {word}")
        elif q.type == "score":
            out.append(f"\nanswer: level {key} of {len(opts) - 1}: {desc}")
        else:
            out.append(f"\nanswer: {desc}" if key == desc else f"\nanswer: {key}: {desc}")
    return f"\n{q.type} question: {instruction}", out


def promote_question_context(request: SystemOneRequest) -> SystemOneRequest:
    """typed.promote_question_context: one choice asked of an empty state, its instructions a header, ":\\n" and a
    body, is read with the body as the state."""
    state = request.state
    empty = state is None or (isinstance(state, str) and not state.strip()) or (isinstance(state, dict) and not state)
    if not empty or len(request.questions) != 1 or _media(request, "images") or _media(request, "videos"):
        return request
    name, q = next(iter(request.questions.items()))
    if q.type != "choice" or not isinstance(q.instructions, str):
        return request
    header, delimiter, body = q.instructions.partition(":\n")
    if not delimiter or not header.strip() or not body.strip():
        return request
    return request.model_copy(update={"state": {"question_context": body},
                                      "questions": {name: q.model_copy(update={"instructions": header + ":"})}})


def _media(request, kind: str) -> list:
    refs = getattr(request, kind, None) or []
    if not isinstance(refs, list):
        raise ValueError(f"{kind} must be a list")
    return refs


def media_source(ref: Any, what: str, root: str | None = None, allow_paths: bool = True):
    """A path or the bytes (as a file object) of an image or video given as a local path, a data URL or base64. Paths
    are read when `allow_paths`; with a media `root`, relative to it and only inside it."""
    if isinstance(ref, Path):
        ref = str(ref)
    if not isinstance(ref, str):
        raise ValueError(f"{what}: unsupported reference of type {type(ref).__name__}")
    if not ref.startswith("data:") and len(ref) < 4096:
        p = Path(ref)
        if root is not None:
            base = Path(root).resolve()
            p = (p if p.is_absolute() else base / p).resolve()
            if base not in p.parents:
                p = None
        if p is not None and p.is_file():
            if not allow_paths:
                raise ValueError(f"{what}: local paths are not accepted by this server; send base64 data")
            return str(p)
    return io.BytesIO(media_bytes(ref, what))


def read_image(ref: Any, root: str | None = None, allow_paths: bool = True):
    from PIL import Image

    if hasattr(ref, "convert"):
        return ref.convert("RGB")
    try:
        with Image.open(media_source(ref, "image", root, allow_paths)) as im:
            return im.convert("RGB")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"not a readable image ({type(exc).__name__})") from None


def read_video(ref: Any, sample_indices_fn, root: str | None = None, allow_paths: bool = True):
    """(frames [n, height, width, 3] in RGB, metadata) of a video given as a path, a data URL or base64: the frames
    transformers' read_video_pyav returns, decoded on the decoder's threads."""
    import av
    import numpy as np
    from transformers.video_utils import VideoMetadata

    source = media_source(ref, "video", root, allow_paths)
    try:
        container = av.open(source)
    except Exception as exc:
        raise ValueError(f"not a readable video ({type(exc).__name__})") from None
    with container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        total, fps = stream.frames, stream.average_rate
        meta = VideoMetadata(total_num_frames=int(total), fps=float(fps), duration=float(total / fps if fps else 0),
                             video_backend="pyav", height=stream.height, width=stream.width)
        indices = sample_indices_fn(metadata=meta)
        want, end = set(int(i) for i in indices), int(indices[-1])
        frames = []
        for i, frame in enumerate(container.decode(stream)):
            if i in want:
                frames.append(frame.to_ndarray(format="rgb24"))
            if i >= end:
                break
    if not frames:
        raise ValueError("the video holds no frames")
    meta.frames_indices = indices
    return np.stack(frames), meta


def _head_module(hidden: int):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class TypedDecisionHead(nn.Module):
        def __init__(self, dimension=768, hidden=256):
            super().__init__()
            self.state_proj = nn.Linear(dimension, hidden)
            self.question_proj = nn.Linear(dimension, hidden)
            self.option_proj = nn.Linear(dimension, hidden)
            self.type_emb = nn.Embedding(3, hidden)
            self.field = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Identity(), nn.Linear(hidden, hidden))
            self.field_norm = nn.LayerNorm(hidden)
            self.option_norm = nn.LayerNorm(hidden)
            self.residual = nn.Sequential(nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Identity(), nn.Linear(hidden, 1))
            self.log_prior_scale = nn.Parameter(torch.tensor(math.log(20.)))
            self.log_joint_scale = nn.Parameter(torch.tensor(math.log(10.)))
            self.gate = nn.Parameter(torch.tensor(-2.))
            self.register_buffer("temperatures", torch.ones(3))

        def forward(self, state, question, options, types, mask):
            s, q, o = self.state_proj(state), self.question_proj(question), self.option_proj(options)
            field = self.field_norm(self.field(torch.cat([s, q, s * q], -1)) + self.type_emb(types))
            option = self.option_norm(o)
            field = field[:, None].expand_as(option)
            joint = self.log_joint_scale.clamp(max=math.log(100.)).exp() * F.cosine_similarity(field, option, dim=-1)
            residual = self.residual(torch.cat([field, option, field * option, (field - option).abs()], -1)).squeeze(-1)
            anchor = F.normalize(state + question, dim=-1)
            prior = self.log_prior_scale.clamp(max=math.log(100.)).exp() * torch.einsum(
                "bkd,bd->bk", F.normalize(options, dim=-1), anchor)
            logits = prior + self.gate.sigmoid() * (joint + residual)
            return (logits / self.temperatures[types, None]).masked_fill(~mask, -1e4)

    class CrossAttentionRefiner(nn.Module):
        def __init__(self, dimension=768, width=256, heads=4, layers=2):
            super().__init__()
            self.query = nn.Linear(2 * dimension, width)
            self.key = nn.Linear(dimension, width)
            self.attn = nn.ModuleList(nn.MultiheadAttention(width, heads, batch_first=True) for _ in range(layers))
            self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(2 * layers))
            self.ffn = nn.ModuleList(nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
                                     for _ in range(layers))
            self.out = nn.Linear(width, 1)

        def forward(self, option_queries, tokens):
            q = self.query(option_queries)[None]
            kv = self.key(tokens)[None]
            for attn, n1, n2, ffn in zip(self.attn, self.norms[0::2], self.norms[1::2], self.ffn):
                q = n1(q + attn(q, kv, kv, need_weights=False)[0])
                q = n2(q + ffn(q))
            return self.out(q)[0, :, 0]

    class JointDecisionHead(nn.Module):
        def __init__(self, hidden=256):
            super().__init__()
            self.base = TypedDecisionHead(hidden=hidden)
            self.refiner = CrossAttentionRefiner()

    return JointDecisionHead(hidden)


class _Hidden:
    """The final hidden states as they are, left on the device for `finish`."""

    on_device = True

    def __call__(self, h):
        return h


@register("embeddinggemma_joint")
class EmbeddingGemmaJoint(Format):
    backbones = ("embedding_gemma2",)
    attention = "blocks"
    max_choice_options = 256
    max_score_levels = 256

    def __init__(self, model_dir: str, tokenizer=None, max_frames: int | None = None, fps: float | None = None,
                 context_from_question: bool = False, media_root: str | None = None, allow_paths: bool = True):
        import transformers

        root = checkpoint(model_dir)
        if root is None:
            raise ValueError(f"{model_dir} is not the backbone of a typed-joint checkpoint (config.json and "
                             "head.safetensors beside it)")
        self.model_dir, self.root = model_dir, root
        self.config = json.loads((root / "config.json").read_text())
        b, h = self.config["backbone"], self.config["head"]
        self.tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.max_tokens = int(b.get("max_tokens", 8192))
        self.fps = float(b["fps"] if fps is None else fps)
        self.max_frames = int(b["max_frames"] if max_frames is None else max_frames)
        self.image_tokens = int(b.get("image_soft_tokens", 280))
        self.video_tokens = int(b.get("video_soft_tokens", 140))
        self.criteria = h.get("noul_render", "plain") == "criteria"
        self.windowed = h.get("long_mode") == "windowed"
        self.window_pool = h.get("window_pool", "max")
        self.context_from_question = bool(context_from_question)
        self.media_root, self.allow_paths = media_root, bool(allow_paths)
        self.head = None
        self.prefix = self.tokenizer(PREFIX)["input_ids"][:-1]
        self.eos = self.tokenizer.eos_token_id
        self._shared = {"lock": threading.Lock()}

    @classmethod
    def detect(cls, model_dir: str, config: Any, decision_config: dict | None) -> bool:
        return getattr(config, "model_type", None) == "embedding_gemma2" and checkpoint(model_dir) is not None

    def for_thread(self) -> "EmbeddingGemmaJoint":
        twin = super().for_thread()
        twin._processor = None
        return twin

    def _load_processor(self):
        """The checkpoint's processor, or the base model's at the revision the checkpoint was trained from (a
        fine-tuned checkpoint keeps the base vision tower and ships no processor). Loaded once for every thread's
        twin; videos go through `read_video`."""
        with self._shared["lock"]:
            p = self._shared.get("processor")
            if p is not None:
                return p
            import transformers

            d = Path(self.model_dir)
            try:
                if any((d / f).exists() for f in ("processor_config.json", "preprocessor_config.json")):
                    p = transformers.AutoProcessor.from_pretrained(self.model_dir, local_files_only=True)
                else:
                    b, base = self.config["backbone"], self.config.get("base_model", {})
                    p = transformers.AutoProcessor.from_pretrained(
                        b.get("model") or base.get("model", "google/embeddinggemma-2"),
                        revision=b.get("revision") or base.get("revision"))
            except ImportError as e:
                if "orchvision" not in str(e):
                    raise
                raise ImportError("images and videos need torchvision: pip install torchvision") from e
            root, allow = self.media_root, self.allow_paths

            def fetch(vp, videos, sample_indices_fn=None):
                if isinstance(videos, (list, tuple)):
                    return list(zip(*[fetch(vp, v, sample_indices_fn) for v in videos]))
                return read_video(videos, sample_indices_fn, root, allow)

            if hasattr(p, "video_processor"):
                from types import MethodType

                p.video_processor.fetch_videos = MethodType(fetch, p.video_processor)
            self._shared["processor"] = p
            return p

    @property
    def processor(self):
        """This thread's processor: the shared image and video processors with this thread's tokenizer (a tokenizer
        fails when two threads encode at once; the checkpoint's is the base model's, which the processor ships)."""
        p = getattr(self, "_processor", None)
        if p is None:
            import copy

            p = copy.copy(self._load_processor())
            p.tokenizer = self.tokenizer
            self._processor = p
        return p

    def load_heads(self, backbone) -> dict[str, Any]:
        from safetensors.torch import load_file

        head = _head_module(int(self.config["head"]["hidden"]))
        head.load_state_dict(load_file(str(self.root / "head.safetensors")), strict=True)
        self.head = head.to(backbone.device).eval().requires_grad_(False)
        if getattr(backbone, "vision", None) is not None:
            try:
                # now, not inside the first request that carries an image or a video
                self._load_processor()
            except Exception as exc:
                log.warning("the processor did not load (%s); requests with images or videos will fail", exc)
        return {READ: _Hidden()}

    def _tokens(self, s: str) -> list[int]:
        return self.tokenizer(s, add_special_tokens=False)["input_ids"]

    def _state(self, request) -> tuple[list[int], dict | None, int]:
        """(tokens of the task prefix and the state without <eos>, vision inputs or None, where the state span
        starts)."""
        state = render_state(request.state, getattr(request, "workflow", None) or "")
        images, videos = _media(request, "images"), _media(request, "videos")
        if not images and not videos:
            return self.tokenizer(PREFIX + state)["input_ids"][:-1], None, 1
        for marker in MARKERS:
            state = state.replace(marker, marker.replace("<", "<​"))
        s = PREFIX + state + " <|image|>" * len(images) + " <|video|>" * len(videos)
        kw: dict[str, Any] = {"text": [s], "return_tensors": "pt", "padding": True, "truncation": False}
        if images:
            kw["images"] = [[read_image(r, self.media_root, self.allow_paths) for r in images]]
            kw["images_kwargs"] = {"max_soft_tokens": self.image_tokens}
        if videos:
            kw["videos"] = [list(videos)]
            kw["videos_kwargs"] = {"fps": self.fps, "max_frames": self.max_frames,
                                   "max_soft_tokens": self.video_tokens, "overflow_strategy": "uniform"}
        out = self.processor(**kw)
        vision = {k: v for k, v in out.items() if k not in ("input_ids", "attention_mask")}
        return out["input_ids"][0, :-1].tolist(), vision, 0

    def plan(self, request: SystemOneRequest, **_) -> Plan:
        if self.context_from_question:
            request = promote_question_context(request)
        state, vision, s0 = self._state(request)
        limit = self.max_tokens - 1
        windowed = self.windowed and vision is None
        if len(state) >= limit and not windowed:
            raise ValueError(f"the state and its media are {len(state)} tokens; the maximum is {limit - 1}")
        built = []
        for name, q in request.questions.items():
            qtext, otexts = segments(name, q, self.criteria)
            pieces = self.tokenizer([qtext] + otexts, add_special_tokens=False)["input_ids"]
            size = sum(map(len, pieces))
            if len(state) + size > limit and not windowed:
                raise ValueError(f"question {name!r} does not fit with its state ({len(state)} + {size} tokens; the "
                                 f"maximum is {limit})")
            built.append((name, TYPES[q.type], pieces[0], pieces[1:], size))
        if windowed and len(state) + max(b[4] for b in built) > limit:
            seqs = self._windows(state, built, limit)
        else:
            groups, cur, used = [], [], len(state)
            for item in built:
                if cur and used + item[4] > limit:
                    groups.append(cur)
                    cur, used = [], len(state)
                cur.append(item)
                used += item[4]
            groups.append(cur)
            seqs = [self._sequence(state, s0, g) for g in groups]
        data = {"spans": [spans for _, spans in seqs], "names": list(request.questions)}
        reads = [(Read(READ, positions=tuple(range(len(ids)))),) for ids, _ in seqs]
        return Plan([], [ids for ids, _ in seqs], reads, data, vision=vision, attention="blocks")

    def _sequence(self, state: list[int], s0: int, group) -> tuple[list[int], dict]:
        ids = list(state)
        spans = {"state": (s0, len(state)), "questions": []}
        for name, kind, qids, oids, _ in group:
            a = len(ids)
            ids += qids
            qspan, ospans = (a, len(ids)), []
            for o in oids:
                a = len(ids)
                ids += o
                ospans.append((a, len(ids)))
            spans["questions"].append((name, kind, qspan, ospans))
        ids.append(self.eos)
        return ids, spans

    def _windows(self, state: list[int], built, limit: int) -> list[tuple[list[int], dict]]:
        """JointEncoder.window_packs: the state in overlapping windows, the task prefix at the start of each, every
        group of questions on every window."""
        n = 0
        while n < len(self.prefix) and n < len(state) and state[n] == self.prefix[n]:
            n += 1
        head, body = state[:n], state[n:]
        budget = max(limit // 4, max(b[4] for b in built))
        if n + budget + 2 * OVERLAP > limit:
            raise ValueError("the questions leave no room for a window of the state")
        groups, cur, used = [], [], 0
        for item in built:
            if cur and used + item[4] > budget:
                groups.append(cur)
                cur, used = [], 0
            cur.append(item)
            used += item[4]
        groups.append(cur)
        out = []
        for g in groups:
            width = limit - n - sum(b[4] for b in g)
            start = 0
            while True:
                end = min(len(body), start + width)
                out.append(self._sequence(head + body[start:end], 1, g))
                if end >= len(body):
                    break
                start = end - OVERLAP
        return out

    def logits(self, hidden, spans: dict) -> dict:
        """{question: its option logits} of one sequence (float32, on the device): JointDecisionHead.forward."""
        import torch
        import torch.nn.functional as F

        head, dev = self.head, hidden.device
        hidden = hidden.float()
        questions = spans["questions"]
        counts = [len(q[3]) for q in questions]
        pairs = [spans["state"]] + [q[2] for q in questions] + [o for q in questions for o in q[3]]
        idx = torch.tensor(pairs, device=dev)
        csum = torch.cat([hidden.new_zeros(1, hidden.shape[-1]), hidden.cumsum(0)])
        means = F.normalize((csum[idx[:, 1]] - csum[idx[:, 0]]) / (idx[:, 1] - idx[:, 0]).unsqueeze(-1), dim=-1)
        Q = len(questions)
        state, qvec, ovec = means[0], means[1: 1 + Q], means[1 + Q:]
        owner = torch.repeat_interleave(torch.arange(Q, device=dev), torch.tensor(counts, device=dev))
        slot = torch.tensor([j for c in counts for j in range(c)], device=dev)
        opts = hidden.new_zeros(Q, max(counts), hidden.shape[-1])
        opts[owner, slot] = ovec
        mask = torch.zeros(Q, max(counts), dtype=torch.bool, device=dev)
        mask[owner, slot] = True
        types = torch.tensor([q[1] for q in questions], device=dev)
        base = head.base(state.expand(Q, -1), qvec, opts, types, mask)
        refine = head.refiner(torch.cat([ovec, qvec[owner]], -1), hidden) / head.base.temperatures[types[owner]]
        z = base[owner, slot] + refine
        return {q[0]: v for q, v in zip(questions, torch.split(z, counts))}

    def finish(self, request: SystemOneRequest, plan: Plan, outputs, **_):
        import torch

        if self.context_from_question:
            request = promote_question_context(request)
        with torch.inference_mode():
            parts: dict[str, list] = {}
            for spans, branch in zip(plan.data["spans"], outputs):
                for name, z in self.logits(branch[0], spans).items():
                    parts.setdefault(name, []).append(z)
            names = plan.data["names"]
            zs = []
            for name in names:
                xs = parts[name]
                if len(xs) > 1:
                    xs = torch.stack(xs)
                    zs.append(xs.mean(0) if self.window_pool == "mean" else xs.max(0).values)
                else:
                    zs.append(xs[0])
            probs = [p.tolist() for p in torch.split(torch.cat([z.softmax(-1) for z in zs]).cpu(),
                                                     [len(z) for z in zs])]
        answers = {}
        for name, p in zip(names, probs):
            q = request.questions[name]
            if q.type == "noul":
                answers[name] = make_noul(p[1])
            elif q.type == "choice":
                answers[name] = make_choice([str(k) for k in q.criteria], p)
            else:
                answers[name] = make_score(list(q.criteria), p)
        usage = Usage(input_tokens=plan.tokens, output_tokens=len(answers))
        return SystemOneResponse(model="", answers=answers, usage=usage), {"format": self.name, "tokens": plan.tokens}
