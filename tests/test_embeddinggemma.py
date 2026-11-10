"""The EmbeddingGemma 2 family (plugins/flide_embeddinggemma).

Without a checkpoint: the packed encoder against transformers' EmbeddingGemma2TextModel on a small random model in
float32 (sequences of several lengths in one stream, a sliding window shorter than the sequences), and the rendering
of states and questions; the fused vision tower against transformers' on images of several sizes. Needs a GPU for
the encoder and vision parts.

With a typed-joint checkpoint (FLIDE_EMBEDDINGGEMMA=/path/to/joint, its release code in gemma_decision/ beside it):
the plans against the release's JointEncoder.packs token for token and span for span (text, several passes, a
windowed long state, a video with FLIDE_EMBEDDINGGEMMA_VIDEO=/path/to/clip.mp4), and the answers against the release's
JointDecisionModel.decide in float32 (the release on the CPU, which takes a minute or two): the same probabilities to
1e-3 in float32, and within 0.05 of them in bfloat16.

    TRITON_F32_DEFAULT=ieee python tests/test_embeddinggemma.py
"""
from __future__ import annotations

import os

os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "plugins"))

from flide.api.schema import SystemOneRequest
from flide_embeddinggemma import format as fmt

QUESTIONS = {
    "action": {"type": "choice", "instructions": "What should support do next?",
               "criteria": {"refund": "refund the order", "replace": "send a replacement", "escalate": "escalate"}},
    "urgent": {"type": "noul", "instructions": "Does this need a reply within the hour?"},
    "angry": {"type": "noul", "instructions": "Is the customer angry?",
              "criteria": {"true": "The customer uses hostile language.", "false": "The customer is calm."}},
    "severity": {"type": "score", "instructions": "How severe is the damage?",
                 "criteria": ["none", "cosmetic", "partly unusable", "unusable"]},
}
STATE = "Order #1182 arrived with a cracked screen. The customer asks for a refund and attached two photos."


def test_rendering():
    assert fmt.render_state("abc") == "workflow:  | state: abc"
    assert fmt.render_state(None) == "workflow:  | state: "
    assert fmt.render_state({"order_id": 7, "customer": {"tier": "gold"}}, "support") == \
        "workflow: support\norder id: 7\ncustomer · tier: gold"
    assert fmt.render_state('{"a": 1}') == "workflow: \na: 1"
    req = SystemOneRequest.model_validate({"state": STATE, "questions": QUESTIONS})
    q = req.questions
    assert fmt.segments("action", q["action"], True) == (
        "\nchoice question: What should support do next?",
        ["\nanswer: refund: refund the order", "\nanswer: replace: send a replacement", "\nanswer: escalate"])
    assert fmt.segments("urgent", q["urgent"], True)[1] == ["\nanswer: no", "\nanswer: yes"]
    assert fmt.segments("angry", q["angry"], True)[1] == ["\nanswer: no: The customer is calm.",
                                                         "\nanswer: yes: The customer uses hostile language."]
    assert fmt.segments("angry", q["angry"], False)[1] == ["\nanswer: no", "\nanswer: yes"]
    assert fmt.segments("severity", q["severity"], True)[1][3] == "\nanswer: level 3 of 3: unusable"
    one = SystemOneRequest.model_validate({"state": "", "questions": {"topic": {
        "type": "choice", "instructions": "Classify the text:\nthe body", "criteria": {"a": "A", "b": "B"}}}})
    moved = fmt.promote_question_context(one)
    assert moved.state == {"question_context": "the body"} and moved.questions["topic"].instructions == "Classify the text:"
    assert fmt.promote_question_context(req) is req
    print("rendering ok")


def test_encoder():
    """Packed forward == EmbeddingGemma2TextModel, sequence by sequence."""
    import transformers

    from flide.formats.base import Plan, Read
    from flide.runtime.pack import pack
    from flide.runtime.schedule import schedule
    from flide_embeddinggemma.backbone import EmbeddingGemma2

    torch.manual_seed(0)
    text = dict(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=6, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, global_head_dim=32, num_global_key_value_heads=1,
                hidden_size_per_layer_input=32, embedding_dim=48, sliding_window=7, sliding_window_pattern=3,
                max_position_embeddings=512)
    config = transformers.EmbeddingGemma2Config(text_config=text, vision_config=None, audio_config=None)
    model = transformers.EmbeddingGemma2Model(config).to("cuda", torch.float32).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    for layer in model.language_model.layers:
        layer.layer_scalar.fill_(0.9)
    types = model.language_model.config.layer_types
    assert "sliding_attention" in types and "full_attention" in types
    ref_model = transformers.EmbeddingGemma2Model(config).to("cuda", torch.float32).eval()
    ref_model.load_state_dict(model.state_dict())
    backbone = EmbeddingGemma2(model)
    g = torch.Generator().manual_seed(1)
    seqs = [torch.randint(3, 300, (n,), generator=g).tolist() for n in (5, 40, 2, 133, 64, 9)]
    plans = [Plan([], seqs[:4], [(Read(None, positions=tuple(range(len(s)))),) for s in seqs[:4]], attention="blocks"),
             Plan([], seqs[4:], [(Read(None, positions=(0, -1)),) for s in seqs[4:]], attention="blocks")]
    worst = 0.0
    for budget in (10 ** 6, 100):
        for mb in schedule(plans, budget):
            b = pack([p.job for p in mb.pieces], "cuda")
            out = backbone.forward(b)[None]
            for grp in b.groups:
                at = grp.start
                for (br, _, n), piece in zip(grp.owners, mb.pieces):
                    s = piece.job.prefix
                    with torch.inference_mode():
                        ref = ref_model(input_ids=torch.tensor([s], device="cuda")).last_hidden_state[0]
                    # the scheduler moved reads from the end one back, past its copy of the last token
                    rows = [r + 1 if r < 0 else r for r in piece.job.reads[0][0].positions]
                    got = out[at: at + n]
                    err = (got - ref[rows]).abs().max().item() / ref.abs().max().item()
                    worst = max(worst, err)
                    at += n
    assert worst < 2e-4, worst
    print(f"encoder ok (max relative error {worst:.1e})")


def test_vision():
    """The fused vision tower without padding == transformers' tower with it, for images of two sizes."""
    import transformers

    from flide_embeddinggemma.backbone import EmbeddingGemma2

    torch.manual_seed(0)
    text = dict(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, hidden_size_per_layer_input=32, embedding_dim=48)
    vision = dict(hidden_size=48, intermediate_size=96, num_hidden_layers=3, num_attention_heads=3,
                  num_key_value_heads=3, head_dim=16, patch_size=4, pooling_kernel_size=2, position_embedding_size=64)
    config = transformers.EmbeddingGemma2Config(text_config=text, vision_config=vision, audio_config=None)
    model = transformers.EmbeddingGemma2Model(config).to("cuda", torch.float32).eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    backbone = EmbeddingGemma2(model)
    assert backbone.vision is not None

    def grid(w, h, patches):
        xy = torch.stack(torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy"), -1).reshape(-1, 2)
        return torch.cat([xy, torch.full((patches - w * h, 2), -1)])

    patches = 96
    ps = torch.stack([grid(8, 12, patches), grid(6, 4, patches), grid(8, 12, patches), grid(10, 8, patches)])
    px = torch.rand(4, patches, 3 * 4 * 4)
    px[(ps == -1).all(-1)] = 0
    with torch.inference_mode():
        ref = backbone._tower(px, ps)
        got = backbone._features(px, ps)
        backbone.cuda_graphs = True
        graphed = backbone._features(px, ps)
        again = backbone._features(px, ps)
    assert ref.shape == got.shape == (sum(n // 4 for n in (96, 24, 96, 80)), 64), (ref.shape, got.shape)
    worst = max((x - ref).abs().max().item() for x in (got, graphed, again)) / ref.abs().max().item()
    assert worst < 2e-4, worst
    print(f"vision ok (max relative error {worst:.1e})")


def _release(root: Path, device: str):
    sys.path.insert(0, str(root.parent))
    from gemma_decision.typed import load_typed_model

    return load_typed_model(str(root), device=device, local_only=True, media_route="typed")


def _records(video: str | None) -> list[dict]:
    long_state = " ".join(f"Line {i}: the customer wrote again about order {1000 + i}." for i in range(1500))
    many = {f"q{i}": {"type": "choice", "instructions": f"Which of these fits case {i} best? " + "Consider it. " * 40,
                      "criteria": {f"o{j}": f"option {j} " + "of the case " * 30 for j in range(12)}} for i in range(24)}
    out = [{"state": STATE, "questions": QUESTIONS},
           {"state": {"order": {"id": 1182, "status": "delivered"}, "note": STATE}, "workflow": "support",
            "questions": QUESTIONS},
           {"state": STATE, "questions": many},
           {"state": long_state, "questions": QUESTIONS}]
    if video:
        out.append({"state": "Security camera video of a residential property.", "videos": [video],
                    "questions": {k: QUESTIONS[k] for k in ("action", "urgent")}})
    return out


def _probs(answers) -> dict:
    out = {}
    for name, a in answers.items():
        if isinstance(a, dict):
            out[name] = a["probabilities"]
        else:
            out[name] = {"false": 1 - a.noul, "true": a.noul} if a.type == "noul" else a.probabilities
    return out


def test_release(root: Path, video: str | None):
    from flide import Engine

    native = _release(root, "cpu")
    engines = {"float32": Engine(fmt.backbone_dir(str(root)), dtype="float32"),
               "bfloat16": Engine(fmt.backbone_dir(str(root)))}
    family = engines["float32"].format
    for i, record in enumerate(_records(video)):
        packs = native.encoder.packs(record)
        req = SystemOneRequest.model_validate(record)
        plan = family.plan(req)
        assert len(plan.suffixes) == len(packs), (i, len(plan.suffixes), len(packs))
        for ids, spans, pack in zip(plan.suffixes, plan.data["spans"], packs):
            assert ids == pack["ids"], f"record {i}: tokens differ"
            assert tuple(spans["state"]) == tuple(pack["spans"]["state"])
            assert [(a, b, tuple(c), [tuple(o) for o in d]) for a, b, c, d in spans["questions"]] == \
                [(a, b, tuple(c), [tuple(o) for o in d]) for a, b, c, d in pack["spans"]["questions"]]
        if plan.vision is not None:
            for k, v in pack["extra"].items():
                assert torch.equal(v.cpu(), plan.vision[k]), f"record {i}: {k} differs from the release's processor"
        want = _probs(native.decide(record)["answers"])
        worst = {}
        for dtype, engine in engines.items():
            got = _probs(engine.decide(req)[0].answers)
            worst[dtype] = max(abs(got[name][k] - want[name][k]) for name in want for k in want[name])
        assert worst["float32"] < 1e-3 and worst["bfloat16"] < 0.05, (i, worst)
        print(f"release record {i}: {len(packs)} sequences, {plan.tokens} tokens match; largest probability "
              f"difference {worst['float32']:.5f} in float32, {worst['bfloat16']:.4f} in bfloat16")
    print("release ok")


if __name__ == "__main__":
    test_rendering()
    if not torch.cuda.is_available():
        print("no GPU: encoder and release tests skipped")
        sys.exit(0)
    test_encoder()
    test_vision()
    root = os.environ.get("FLIDE_EMBEDDINGGEMMA")
    if root:
        test_release(Path(root), os.environ.get("FLIDE_EMBEDDINGGEMMA_VIDEO"))
    else:
        print("FLIDE_EMBEDDINGGEMMA not set: release tests skipped")
