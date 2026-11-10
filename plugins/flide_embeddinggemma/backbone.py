"""Packed forward for the EmbeddingGemma 2 encoder (model_type embedding_gemma2).

EmbeddingGemma 2 is Gemma's layer stack read both ways: every layer attends over the whole sequence, the sliding
layers to keys at most `sliding_window` positions away on either side. Nothing of one sequence is shared with
another, so a plan is sequences of their own under block attention (formats.base.Plan with an empty prefix), and a
micro-batch is those sequences end to end in one stream with no padding.

The layers run as in flide's Gemma 4 decoder (backbones/gemma4/fused.py): the checkpoint's transformers modules hold
the weights, projections that read the same input are one matrix product, and the kernels of kernels/pointwise.py do
the norms, the rotary embedding and the gated products. Attention is flide's Triton kernel with exact key ranges: a
query's range is its own sequence (or its window inside it), which is the whole mask. The per-layer inputs come from
the embeddings alone (projection-only), and the final hidden states go through the final norm and the embedding
projection, as EmbeddingGemma2TextModel returns them.

Images and video frames: placeholder tokens embed as padding and the vision tower's features replace them
(EmbeddingGemma2Model.forward). The tower is Gemma 4's and runs the same way (FusedVision), each image without its
padding patches, from a CUDA graph per input shape when the engine has CUDA graphs (a camera's clips all have one
shape).
"""
from __future__ import annotations

import logging
from collections import OrderedDict

import torch

from flide.backbones.base import Backbone
from flide.formats.base import FINAL
from flide.kernels import attention as attn_kernel
from flide.kernels import pointwise as pw
from flide.kernels.fp8 import mm

log = logging.getLogger("flide.embeddinggemma")

SKIP = ("audio_tower.", "embed_audio.")


def _merge(linears):
    """One weight for several bias-free nn.Linear that read the same input; each module's weight becomes a row slice
    of it."""
    for m in linears:
        if m.bias is not None:
            raise ValueError("fused projections assume bias-free linear layers")
    w = torch.cat([m.weight.data for m in linears], 0)
    at = 0
    for m in linears:
        n = m.weight.shape[0]
        m.weight.data = w[at: at + n]
        at += n
    return w


class FusedEncoder:
    def __init__(self, lm):
        self.lm = lm
        self.cfg = lm.config
        self.dtype = lm.layers[0].mlp.gate_proj.weight.dtype
        self.window = int(self.cfg.sliding_window or 0)
        self.w = []
        for layer in lm.layers:
            a, mlp, ple = layer.self_attn, layer.mlp, layer.ple_block
            hd = a.head_dim
            self.w.append({"qkv": _merge([a.q_proj, a.k_proj, a.v_proj]), "o": a.o_proj.weight,
                           "gate_up": _merge([mlp.gate_proj, mlp.up_proj]), "down": mlp.down_proj.weight,
                           "inter": mlp.gate_proj.weight.shape[0], "hd": hd, "nh": a.q_proj.weight.shape[0] // hd,
                           "nkv": a.k_proj.weight.shape[0] // hd, "sliding": a.sliding_window is not None,
                           "type": self.cfg.layer_types[a.layer_idx], "ple_gate": ple.per_layer_input_gate.weight,
                           "ple_proj": ple.per_layer_projection.weight})

    def tables(self, pos: torch.Tensor) -> dict:
        """Float32 rotary tables [T, hd] per layer type."""
        x = torch.empty(0, device=pos.device, dtype=torch.float32)
        return {t: tuple(c[0] for c in self.lm.rotary_emb(x, pos[None], t)) for t in self.lm.unique_layer_types}

    def per_layer_inputs(self, emb: torch.Tensor) -> torch.Tensor:
        """[T, layers, n], EmbeddingGemma2TextPLE: the projection of the embeddings, normed per layer."""
        ple = self.lm.ple
        T, n = emb.shape[0], ple.hidden_size_per_layer_input
        p = mm(emb, ple.per_layer_model_projection.weight) * ple.per_layer_model_projection_scale
        norm = ple.per_layer_projection_norm
        return pw.rms_norm(p.view(-1, n), norm.weight, norm.eps, offset=False).view(T, -1, n)

    def __call__(self, emb: torch.Tensor, pos: torch.Tensor, ranges: torch.Tensor, groups: int) -> torch.Tensor:
        """Final hidden states [T, embedding_dim] of the stream emb [T, hidden]: `pos` each token's position in its
        sequence, `ranges` int32 [T, 4] with its sequence's [start, end) in the first two columns (the last two
        empty), `groups` the number of sequences."""
        lm, layers, eps = self.lm, self.lm.layers, self.cfg.rms_norm_eps
        T = emb.shape[0]
        tabs = self.tables(pos)
        pli = self.per_layer_inputs(emb)
        full = ranges.T.contiguous()
        idx = torch.arange(T, device=emb.device, dtype=torch.int32)
        near = full.clone()
        if self.window:
            near[0] = torch.maximum(full[0], idx - self.window)
            near[1] = torch.minimum(full[1], idx + self.window + 1)
        tok = (idx, idx, idx)
        tables: dict = {}
        h = emb
        x = pw.rms_norm(h, layers[0].input_layernorm.weight, eps, offset=False)
        for i, (layer, w) in enumerate(zip(layers, self.w)):
            a = layer.self_attn
            nh, nkv, hd = w["nh"], w["nkv"], w["hd"]
            cos, sin = tabs[w["type"]]
            qkv = mm(x, w["qkv"])
            q = pw.gemma_norm_rope(qkv[:, : nh * hd].view(T, nh, hd), cos, sin, a.q_norm.weight, eps)
            k = pw.gemma_norm_rope(qkv[:, nh * hd: (nh + nkv) * hd].view(T, nkv, hd), cos, sin, a.k_norm.weight, eps)
            v = pw.gemma_norm_rope(qkv[:, (nh + nkv) * hd:].view(T, nkv, hd), None, None, None, eps)
            ac = attn_kernel.config(q.device, q.dtype, hd)
            qr = near if w["sliding"] and self.window else full
            key = (w["sliding"], ac["BLOCK_M"], ac["BLOCK_N"])
            R = tables.get(key)
            if R is None:
                R = tables[key] = attn_kernel.block_ranges(qr[0], qr[1], qr[2], qr[3], ac["BLOCK_M"], ac["BLOCK_N"],
                                                           exact=True, groups=groups,
                                                           tiles_from=(full[0], full[1], full[2], full[3]))
            o = attn_kernel.attention(q, k, v, k, v, tok, tok, tok, R, 1.0, 0, ac, qranges=qr)
            h, x = pw.sandwich(mm(o.reshape(T, nh * hd), w["o"]), h, layer.post_attention_layernorm.weight, eps,
                               w2=layer.pre_feedforward_layernorm.weight)
            n = w["inter"]
            gu = mm(x, w["gate_up"])
            h, _ = pw.sandwich(mm(pw.geglu(gu[:, :n], gu[:, n:]), w["down"]), h,
                               layer.post_feedforward_layernorm.weight, eps)
            g = pw.geglu(mm(h, w["ple_gate"]), pli[:, i])
            nxt = layers[i + 1].input_layernorm.weight if i + 1 < len(layers) else None
            h, x = pw.sandwich(mm(g, w["ple_proj"]), h, layer.ple_block.post_per_layer_input_norm.weight, eps,
                               scale=layer.layer_scalar, w2=nxt)
        return mm(pw.rms_norm(h, lm.norm.weight, lm.norm.eps, offset=False), lm.embedding_projection.weight)


class FusedVision:
    """The Gemma 4 vision tower on the same kernels: images or frames of one size, without padding, as one stream in
    which each image attends over itself. The patch embedding, the pooler and the projection into the encoder are
    transformers' modules (once per call); the layers between are merged projections, flide's norm and gated
    product kernels, the axial rotary embedding of kernels.py and the packed attention with an image's range as its
    mask."""

    def __init__(self, model):
        self.model = model
        vt = self.vt = model.vision_tower
        self.cfg = vt.config
        self.ok = not getattr(self.cfg, "use_clipped_linears", False) and not self.cfg.standardize
        self.w = []
        if not self.ok:
            return
        for layer in vt.encoder.layers:
            a, mlp = layer.self_attn, layer.mlp
            hd = a.head_dim
            q, k, v = a.q_proj.linear, a.k_proj.linear, a.v_proj.linear
            self.w.append({"qkv": _merge([q, k, v]), "o": a.o_proj.linear.weight,
                           "gate_up": _merge([mlp.gate_proj.linear, mlp.up_proj.linear]),
                           "down": mlp.down_proj.linear.weight, "inter": mlp.gate_proj.linear.weight.shape[0],
                           "hd": hd, "nh": q.weight.shape[0] // hd, "nkv": k.weight.shape[0] // hd})

    def __call__(self, px: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
        """Projected features [images * tokens, hidden] of px [images, patches, pixels] at positions ps, none of
        them padding."""
        from .kernels import norm_rope_parts

        vt, layers, eps = self.vt, self.vt.encoder.layers, self.cfg.rms_norm_eps
        F, n = ps.shape[:2]
        T = F * n
        pad = torch.zeros(F, n, dtype=torch.bool, device=ps.device)
        h = vt.patch_embedder(px, ps, pad).reshape(T, -1)
        cos, sin = vt.encoder.rotary_emb(torch.empty(0, device=ps.device, dtype=torch.float32), ps)
        cos, sin = cos.reshape(T, -1), sin.reshape(T, -1)
        idx = torch.arange(T, device=ps.device, dtype=torch.int32)
        start = idx // n * n
        zero = torch.zeros_like(idx)
        qr = torch.stack([start, start + n, zero, zero])
        tok = (idx, idx, idx)
        R = None
        x = pw.rms_norm(h, layers[0].input_layernorm.weight, eps, offset=False)
        for i, (layer, w) in enumerate(zip(layers, self.w)):
            a = layer.self_attn
            nh, nkv, hd = w["nh"], w["nkv"], w["hd"]
            qkv = mm(x, w["qkv"])
            q = norm_rope_parts(qkv[:, : nh * hd].view(T, nh, hd), cos, sin, a.q_norm.weight, eps, ps.shape[-1])
            k = norm_rope_parts(qkv[:, nh * hd: (nh + nkv) * hd].view(T, nkv, hd), cos, sin, a.k_norm.weight, eps,
                                ps.shape[-1])
            v = pw.gemma_norm_rope(qkv[:, (nh + nkv) * hd:].view(T, nkv, hd), None, None, None, eps)
            ac = attn_kernel.config(q.device, q.dtype, hd)
            if R is None:
                R = attn_kernel.block_ranges(qr[0], qr[1], qr[2], qr[3], ac["BLOCK_M"], ac["BLOCK_N"], exact=True,
                                             groups=F)
            o = attn_kernel.attention(q, k, v, k, v, tok, tok, tok, R, 1.0, 0, ac, qranges=qr)
            h, x = pw.sandwich(mm(o.reshape(T, nh * hd), w["o"]), h, layer.post_attention_layernorm.weight, eps,
                               w2=layer.pre_feedforward_layernorm.weight)
            m = w["inter"]
            gu = mm(x, w["gate_up"])
            nxt = layers[i + 1].input_layernorm.weight if i + 1 < len(layers) else None
            h, x = pw.sandwich(mm(pw.geglu(gu[:, :m], gu[:, m:]), w["down"]), h,
                               layer.post_feedforward_layernorm.weight, eps, w2=nxt)
        k = self.cfg.pooling_kernel_size
        # every pooled token is real (no padding went in), so none is dropped: nothing here waits for the device
        pooled, _ = vt.pooler(hidden_states=h.view(F, n, -1), pixel_position_ids=ps, padding_positions=pad,
                              output_length=n // (k * k))
        return self.model.embed_vision(inputs_embeds=pooled.to(h.dtype)).flatten(0, 1)


class VisionGraphs:
    """The vision tower and its projection from CUDA graphs, one per input shape; the removal of the padding tokens,
    whose count depends on the frames, runs after the replay."""

    def __init__(self, model, fused=None, max_graphs: int = 4):
        self.model = model
        self.fused = fused
        self.max_graphs = max_graphs
        self.pool = torch.cuda.graph_pool_handle()
        self.graphs: OrderedDict = OrderedDict()

    def _body(self, pixels, positions, whole: bool):
        if whole and self.fused is not None:
            return self.fused(pixels, positions), None
        vt = self.model.vision_tower
        k = vt.config.pooling_kernel_size
        padding = (positions == -1).all(dim=-1)
        x = vt.patch_embedder(pixels, positions, padding)
        out = vt.encoder(inputs_embeds=x, attention_mask=~padding, pixel_position_ids=positions)
        h, keep = vt.pooler(hidden_states=out.last_hidden_state, pixel_position_ids=positions, padding_positions=padding,
                            output_length=pixels.shape[-2] // (k * k))
        if vt.config.standardize:
            h = (h - vt.std_bias.float()) * vt.std_scale.float()
        return self.model.embed_vision(inputs_embeds=h.to(x.dtype)), keep

    def __call__(self, pixels: torch.Tensor, positions: torch.Tensor, whole: bool = False) -> torch.Tensor:
        """`whole`: no patch is padding (the fused tower takes those)."""
        key = (tuple(pixels.shape), pixels.dtype, tuple(positions.shape), whole)
        entry = self.graphs.get(key)
        if entry is None:
            pbuf, qbuf = pixels.clone(), positions.clone()
            for _ in range(2):
                self._body(pbuf, qbuf, whole)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                out = self._body(pbuf, qbuf, whole)
            torch.cuda.synchronize()
            entry = self.graphs[key] = (g, pbuf, qbuf, out)
            while len(self.graphs) > self.max_graphs:
                self.graphs.popitem(last=False)
        else:
            self.graphs.move_to_end(key)
        g, pbuf, qbuf, (feats, keep) = entry
        pbuf.copy_(pixels)
        qbuf.copy_(positions)
        g.replay()
        return feats if keep is None else feats[keep]


class EmbeddingGemma2(Backbone):
    model_types = ("embedding_gemma2",)
    block_attention = True

    @classmethod
    def load(cls, model_dir: str, device: str = "cuda:0", dtype: str = "bfloat16", loader: str = "direct",
             **options) -> "EmbeddingGemma2":
        """`loader` "direct" reads the safetensors straight onto the device (flide.backbones.load); "hf" uses
        transformers' from_pretrained; "none" reads no weights (zeros). The audio tower stays on disk."""
        import transformers

        config = transformers.AutoConfig.from_pretrained(model_dir, local_files_only=True)
        config.audio_config = None
        arch = transformers.EmbeddingGemma2Model
        model = None
        if loader in ("direct", "none"):
            from flide.backbones.load import blank, build, load_into

            try:
                model = build(arch, config, torch.device(device), getattr(torch, dtype))
                if loader == "none":
                    blank(model)
                else:
                    load_into(model, model_dir, device, skip=SKIP)
            except (ValueError, FileNotFoundError) as exc:
                logging.getLogger("flide.load").warning("direct load failed (%s); using transformers", exc)
                model = None
                torch.cuda.empty_cache()
        if model is None:
            model = arch.from_pretrained(model_dir, config=config, dtype=getattr(torch, dtype), local_files_only=True)
            model = model.to(device)
        return cls(model.eval().requires_grad_(False), **options)

    def __init__(self, model, cuda_graphs: bool = False, quantize: str | None = None, tp=None, **_):
        if quantize is not None:
            raise NotImplementedError("embedding_gemma2: FP8 is not supported")
        if tp is not None and tp[1] > 1:
            raise NotImplementedError("embedding_gemma2: tensor parallel is not supported")
        self.model = model
        self.lm = model.language_model
        self.config = self.lm.config
        self.num_layers = self.config.num_hidden_layers
        p0 = next(self.lm.parameters())
        self.device, self.dtype = p0.device, p0.dtype
        top = model.config
        self.placeholders = {i for i in (getattr(top, k, None) for k in ("image_token_id", "video_token_id",
                                                                          "audio_token_id")) if i is not None}
        self.pad_id = self.config.pad_token_id
        self.cuda_graphs = bool(cuda_graphs) and self.device.type == "cuda"
        self.encoder = FusedEncoder(self.lm)
        self.vision = FusedVision(model) if model.vision_tower is not None and self.device.type == "cuda" else None
        if self.vision is not None and not self.vision.ok:
            self.vision = None
        self._vision_graphs = None
        if self.device.type == "cuda":
            self.warmup()

    @torch.inference_mode()
    def warmup(self) -> None:
        """One pass over two made-up sequences, so the encoder's kernels are loaded before the first request (the
        engine's own warm-up plans share a prefix between branches, which an encoder does not read)."""
        lens = (300, 70)
        T = sum(lens)
        ids = torch.randint(0, self.config.vocab_size, (T,), device=self.device)
        pos = torch.cat([torch.arange(n, device=self.device) for n in lens])
        ranges = torch.zeros(T, 4, dtype=torch.int32, device=self.device)
        ranges[: lens[0], 1] = lens[0]
        ranges[lens[0]:, 0] = lens[0]
        ranges[lens[0]:, 1] = T
        self.encoder(self.lm.embed_tokens(ids), pos, ranges, len(lens))
        torch.cuda.synchronize(self.device)

    def output_head(self):
        raise NotImplementedError("embedding_gemma2 is an encoder: it has no output head")

    def vision_positions(self, ids, vision):
        """Every token sits at its index (no multimodal rotary positions)."""
        return None, 0

    def _tower(self, px: torch.Tensor, ps: torch.Tensor, whole: bool = False) -> torch.Tensor:
        """`whole`: no patch is padding, so the fused tower runs (FusedVision); else transformers' modules."""
        px, ps = px.to(self.device, self.dtype), ps.to(self.device)
        whole = whole and self.vision is not None
        if self.cuda_graphs:
            if self._vision_graphs is None:
                self._vision_graphs = VisionGraphs(self.model, self.vision)
            return self._vision_graphs(px, ps, whole)
        if whole:
            return self.vision(px, ps)
        out = self.model.vision_tower(pixel_values=px, pixel_position_ids=ps)
        return self.model.embed_vision(inputs_embeds=out.last_hidden_state)

    def _features(self, px: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
        """Features [tokens, hidden] of images or frames px [n, patches, pixels] at positions ps, in order. The
        processor pads every image to the same number of patches; the tower masks the padding out of attention and
        drops its tokens, so images run without it (those of one size together), which costs that much less."""
        k2 = self.model.config.vision_config.pooling_kernel_size ** 2
        real = (ps != -1).all(-1)
        counts = real.sum(-1)
        ahead = torch.arange(ps.shape[1], device=ps.device)[None] < counts[:, None]
        if not bool((real == ahead).all()) or bool((counts % k2).any()) or not bool(counts.all()):
            return self._tower(px, ps)
        sizes = counts.unique().tolist()
        if len(sizes) == 1:
            return self._tower(px[:, :sizes[0]], ps[:, :sizes[0]], True)
        parts: list = [None] * px.shape[0]
        for n in sizes:
            rows = (counts == n).nonzero().squeeze(1)
            f = self._tower(px[rows, :n], ps[rows, :n], True)
            for r, part in zip(rows.tolist(), f.split(n // k2)):
                parts[r] = part
        return torch.cat(parts)

    def vision_features(self, vision: dict) -> dict:
        """{token id: features [tokens, hidden]} of one request's images and video frames, in order; kept on the
        request's `vision`, so the sequences of one request (questions over several passes) encode its media once."""
        feats = vision.get("_features")
        if feats is not None:
            return feats
        if self.model.vision_tower is None:
            raise ValueError("this checkpoint has no vision tower")
        top = self.model.config
        feats = {}
        for pix, pos, token in (("pixel_values", "image_position_ids", top.image_token_id),
                                ("pixel_values_videos", "video_position_ids", top.video_token_id)):
            if vision.get(pix) is not None:
                feats[token] = self._features(vision[pix], vision[pos])
        vision["_features"] = feats
        return feats

    def embed(self, ids: torch.Tensor, bounds: list[int], jobs) -> torch.Tensor:
        """Embeddings of the stream, media placeholders embedded as padding and replaced by their features."""
        mm_mask = torch.isin(ids, torch.tensor(sorted(self.placeholders), device=ids.device))
        emb = self.lm.embed_tokens(torch.where(mm_mask, self.pad_id, ids))
        for j, job in enumerate(jobs):
            if job.vision is None:
                continue
            a, e = bounds[j], bounds[j + 1]
            for token, f in self.vision_features(job.vision).items():
                at = (ids[a:e] == token).nonzero().squeeze(1)
                if at.shape[0] != f.shape[0]:
                    raise ValueError(f"{at.shape[0]} placeholder tokens for {f.shape[0]} vision features")
                emb[a + at] = f.to(emb.dtype)
        return emb

    @torch.inference_mode()
    def forward(self, b, pad_to: int | None = None) -> dict:
        if any(not j.blocks or len(j.suffixes) != 1 or j.start or j.marks for j in b.jobs):
            raise NotImplementedError("embedding_gemma2 reads whole sequences both ways: a plan without a prefix under "
                                      "block attention")
        if any(k is not FINAL for k in b.reads):
            raise NotImplementedError("embedding_gemma2 is read after its last layer")
        # Each sequence is its job's prefix; the one branch token after it is the scheduler's copy of the last token.
        T = b.n_prefix_tokens
        ids = b.input_ids[:T]
        emb = self.embed(ids, b.cu_prefix_cpu.tolist(), b.jobs)
        h = self.encoder(emb, b.positions[:T], b.key_ranges[:T], b.n_jobs)
        return {FINAL: h[b.reads[FINAL]]} if FINAL in b.reads else {}
