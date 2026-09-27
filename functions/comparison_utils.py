"""Shared utilities for the "comparison" family of interpretability experiments
(e11_comparison_largemodels, e12_compare_objects, and first_numbers_then_compare).

Every function takes model/tokenizer/device explicitly instead of relying on notebook
globals, so any notebook can do:

    sys.path.append('../')
    from functions.comparison_utils import get_attentions, causal_trace_sweep, ...

These were consolidated from duplicated definitions in where_compare.ipynb,
where_compare_truefalse.ipynb, which_components.ipynb, isolate_postion_value.ipynb,
multi_compare.ipynb, and compare_objects.ipynb.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Patch


# ── Token-position labeling ──────────────────────────────────────────────────

def num_char_positions(prompt, num_a, num_b, anchor):
    """Char spans of num_a and num_b in the question clause introduced by `anchor`
    (e.g. 'The maximum of '). Uses the LAST occurrence of `anchor` in `prompt`, so a
    few-shot example that repeats the same phrase earlier is skipped."""
    q_start = prompt.rfind(anchor) + len(anchor)
    rest = prompt[q_start:]
    a_s = rest.find(str(num_a))
    b_s = rest.find(' and ') + 5
    return (q_start + a_s, q_start + a_s + len(str(num_a)),
            q_start + b_s, q_start + b_s + len(str(num_b)))


def tok_indices_for_chars(offsets_np, c_start, c_end):
    """Token indices whose char span overlaps [c_start, c_end)."""
    return [i for i, (s, e) in enumerate(offsets_np) if e > c_start and s < c_end]


def question_start_position(prompt, offsets_np, anchor):
    """First token index of the actual question, skipping any preamble/few-shot
    example that repeats `anchor` earlier in the prompt."""
    q_char = prompt.rfind(anchor)
    return min(tok_indices_for_chars(offsets_np, q_char, q_char + 1))


def template_token_label(prompt, num_a, num_b, offsets_np, tokens, pos, anchor):
    """Template-relative label for token position `pos`: 'a1'/'a2' for num_a's own
    digit token(s), 'b1'/'b2' for num_b's, or the literal template token otherwise."""
    ca_s, ca_e, cb_s, cb_e = num_char_positions(prompt, num_a, num_b, anchor)
    toks_a = tok_indices_for_chars(offsets_np, ca_s, ca_e)
    toks_b = tok_indices_for_chars(offsets_np, cb_s, cb_e)
    if pos in toks_a:
        return f'a{toks_a.index(pos) + 1}'
    if pos in toks_b:
        return f'b{toks_b.index(pos) + 1}'
    return tokens[pos].strip()


def discrete_position_label(prompt, num_a, num_b, offsets_np, tokens, pos, T, anchor):
    """Like template_token_label, but flags the final prompt position as '[last tok]'."""
    if pos == T - 1:
        return '[last tok]'
    return template_token_label(prompt, num_a, num_b, offsets_np, tokens, pos, anchor)


# ── Forward-pass readouts ────────────────────────────────────────────────────

def get_attentions(model, input_ids):
    """Single forward pass -> stacked attention (n_layers, n_heads, T, T) on CPU."""
    with torch.no_grad():
        out = model(input_ids, output_attentions=True)
    return torch.stack([a[0].cpu().float() for a in out.attentions])


def logits_of(model, ids):
    """Last-position logits of a single forward pass."""
    with torch.no_grad():
        return model(ids).logits[0, -1]


def ld_of(model, ids, tok_a, tok_b):
    """Logit difference logit[tok_a] - logit[tok_b] at the last position."""
    lg = logits_of(model, ids)
    return (lg[tok_a] - lg[tok_b]).item()


def prob_of(model, ids, tok):
    """Softmax probability of `tok` at the last position."""
    lg = logits_of(model, ids)
    return torch.softmax(lg, dim=-1)[tok].item()


def hidden_states_of(model, input_ids):
    """Single forward pass with output_hidden_states=True. Returns a tuple of
    (n_layers+1) tensors, each (1, T, d_model) on CPU: index 0 = embedding output,
    index l (l>=1) = residual stream after decoder layer l-1 -- i.e. the same value a
    forward hook on model.model.layers[l-1] would capture."""
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    return tuple(hs.detach().cpu() for hs in out.hidden_states)


# ── Activation caching / single-position patching (ROME-style causal trace) ─
# Meng, Bau, Andonian & Belinkov, 2022 ("Locating and Editing Factual Associations
# in GPT") — restore one hidden state h_i^(l) at a time from a clean run into a
# corrupted run and measure how much of the clean answer that one restoration recovers.

def cache_all_layers(model, input_ids, n_layers):
    """Run `input_ids` and cache the residual-stream hidden state leaving each decoder
    layer, at every position. Returns {layer_idx: (1, T, d_model) tensor on CPU}."""
    cache = {}

    def make_hook(ll):
        def h(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            cache[ll] = hs.detach().cpu()
        return h

    handles = [model.model.layers[l].register_forward_hook(make_hook(l)) for l in range(n_layers)]
    with torch.no_grad():
        _ = model(input_ids)
    for hh in handles:
        hh.remove()
    return cache


def patch_position(model, clean_hidden, layer, pos, device):
    """Register a forward hook on model.model.layers[layer] that overwrites position
    `pos`'s hidden state with its cached clean value (clean_hidden[layer]). Returns the
    hook handle -- caller must call .remove()."""
    h_l = clean_hidden[layer].to(device)

    def hook(mod, inp, out, _pos=pos, _h=h_l):
        hs = (out[0] if isinstance(out, tuple) else out).clone()
        hs[:, _pos, :] = _h[:, _pos, :]
        return (hs,) + out[1:] if isinstance(out, tuple) else hs

    return model.model.layers[layer].register_forward_hook(hook)


def generate_n_patched(model, input_ids, hook_handle, n_new, device):
    """Greedy-generate n_new tokens with `hook_handle`'s patch active (no KV-cache reuse,
    since the patch only covers the original prompt length). Removes the hook when done.
    Returns the n_new generated token ids as a plain list."""
    cur_ids = input_ids
    T = input_ids.shape[1]
    with torch.no_grad():
        for _ in range(n_new):
            step_logits = model(cur_ids).logits[0, -1]
            next_tok = step_logits.argmax().view(1, 1).to(device)
            cur_ids = torch.cat([cur_ids, next_tok], dim=1)
    hook_handle.remove()
    return [int(t) for t in cur_ids[0, T:]]


def causal_trace_sweep(model, clean_hidden, inp_corr, n_layers, pos_start, T,
                        tok_a, tok_b, ld_clean, ld_corr, device, show_progress=True):
    """Single-hidden-state restoration sweep over every (layer, position) pair.

    Returns (recovery, indirect_effect), each shaped (n_layers, T):
      recovery(l,i)        = (LD_patched - LD_corr) / (LD_clean - LD_corr)   [0=none, 1=full]
      indirect_effect(l,i) = P_patched[tok_a] - P_corr[tok_a]                [Meng et al. 2022, Sec. 3]
    Positions before `pos_start` are left at 0 (guaranteed no-op when clean/corrupted
    prompts are identical up to that point). Set `show_progress=False` when calling this
    once per pair in an outer loop that already has its own progress bar.
    """
    P_corr = prob_of(model, inp_corr, tok_a)
    recovery = np.zeros((n_layers, T))
    indirect_effect = np.zeros((n_layers, T))

    layer_iter = tqdm(range(n_layers), desc='Causal trace') if show_progress else range(n_layers)
    for l in layer_iter:
        for pos in range(pos_start, T):
            hnd = patch_position(model, clean_hidden, l, pos, device)
            with torch.no_grad():
                logits_p = model(inp_corr).logits[0, -1]
            hnd.remove()
            ld_p = (logits_p[tok_a] - logits_p[tok_b]).item()
            recovery[l, pos] = (ld_p - ld_corr) / (ld_clean - ld_corr)
            indirect_effect[l, pos] = torch.softmax(logits_p, dim=-1)[tok_a].item() - P_corr

    return recovery, indirect_effect


def make_compute_metrics(model, inp_clean, inp_corr, tok_A, tok_Bc, tok_nc):
    """Builds a `compute_metrics(logits)` closure plus its baseline stats, for
    `causal_trace_sweep3`. Domain-agnostic port of `reuse_multidigit.ipynb`'s own
    per-digit-group closure -- `tok_A`/`tok_Bc`/`tok_nc` need only be vocabulary token ids,
    so the same construction works whether the comparison is numeric, object-weight, or any
    other two-slot comparison:
      tok_A  = clean run's own correct answer (the value/object that should win)
      tok_Bc = corrupted run's own correct answer, same token as clean's *losing* slot
               (shared between clean and corrupted -- the thing that stays in place)
      tok_nc = corrupted run's own *first*-slot token (the new value that makes the
               corrupted run's answer flip away from tok_A)
    """
    ld_clean, ld_corr       = ld_of(model, inp_clean, tok_A, tok_Bc),  ld_of(model, inp_corr, tok_A, tok_Bc)
    posld_clean, posld_corr = ld_of(model, inp_clean, tok_nc, tok_Bc), ld_of(model, inp_corr, tok_nc, tok_Bc)
    P_clean, P_corr = prob_of(model, inp_clean, tok_A), prob_of(model, inp_corr, tok_A)

    def compute_metrics(logits):
        """recovery, indirect_effect, position_recovery for one patched-logits vector."""
        recovery          = ((logits[tok_A]  - logits[tok_Bc]).item() - ld_corr)    / (ld_clean - ld_corr)
        indirect_effect   = torch.softmax(logits, dim=-1)[tok_A].item() - P_corr
        position_recovery = ((logits[tok_nc] - logits[tok_Bc]).item() - posld_corr) / (posld_clean - posld_corr)
        return recovery, indirect_effect, position_recovery

    stats = dict(ld_clean=ld_clean, ld_corr=ld_corr, posld_clean=posld_clean, posld_corr=posld_corr,
                 P_clean=P_clean, P_corr=P_corr, TE=P_clean - P_corr)
    return compute_metrics, stats


def causal_trace_sweep3(model, clean_hidden, inp_corr, n_layers, pos_start, T, compute_metrics, device,
                         show_progress=True):
    """Like `causal_trace_sweep`, but also reports `position_recovery` (see
    `make_compute_metrics`) alongside `recovery`/`indirect_effect`. `compute_metrics` comes
    from `make_compute_metrics`. Returns (recovery, indirect_effect, position_recovery),
    each shaped (n_layers, T)."""
    recovery = np.zeros((n_layers, T))
    indirect_effect = np.zeros((n_layers, T))
    position_recovery = np.zeros((n_layers, T))

    layer_iter = tqdm(range(n_layers), desc='Causal trace') if show_progress else range(n_layers)
    for l in layer_iter:
        for pos in range(pos_start, T):
            hnd = patch_position(model, clean_hidden, l, pos, device)
            with torch.no_grad():
                logits_p = model(inp_corr).logits[0, -1]
            hnd.remove()
            recovery[l, pos], indirect_effect[l, pos], position_recovery[l, pos] = compute_metrics(logits_p)

    return recovery, indirect_effect, position_recovery


# ── Module-restricted patching (MLP-only / attention-only traces) ───────────
# Meng, Bau, Andonian & Belinkov, 2022, Sec. 3.3: rerunning the causal trace while restoring only
# one submodule's output separates *where* a state matters from *which component* wrote it. The
# residual-stream trace above restores everything a layer produced at a position; these restore
# only that layer's MLP output, or only its attention output (post-o_proj, i.e. all heads of the
# layer summed), leaving the other component running on corrupted input.
#
# Meng et al. restore a sliding window of ~10 layers because a single MLP's own contribution is
# often too small to move the answer on its own. `causal_trace_sweep_module` defaults to a single
# layer (`window=1`), which keeps the output the same shape as `causal_trace_sweep`'s and therefore
# directly comparable to it; pass `window=3`, `window=10`, ... for the paper's windowed trace,
# which is correspondingly brighter.

MODULE_KINDS = ('mlp', 'attn')


def _submodule(model, layer, kind):
    """model.model.layers[layer]'s MLP or self-attention block."""
    lyr = model.model.layers[layer]
    if kind == 'mlp':
        return lyr.mlp
    if kind == 'attn':
        return lyr.self_attn
    raise ValueError(f"kind must be one of {MODULE_KINDS}, got {kind!r}")


def cache_module_outputs(model, input_ids, n_layers, kind):
    """Run `input_ids` and cache each layer's `kind` submodule output at every position.
    Returns {layer_idx: (1, T, d_model) tensor on CPU}. Attention modules return a
    (output, weights) tuple and MLPs a bare tensor; both are handled."""
    cache = {}

    def make_hook(ll):
        def h(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            cache[ll] = hs.detach().cpu()
        return h

    handles = [_submodule(model, l, kind).register_forward_hook(make_hook(l)) for l in range(n_layers)]
    with torch.no_grad():
        _ = model(input_ids)
    for hh in handles:
        hh.remove()
    return cache


def patch_module_position(model, clean_cache, layer, pos, kind, device):
    """Overwrite position `pos` of layer `layer`'s `kind` submodule output with its cached clean
    value. Everything else in the layer -- the other submodule, and this one at every other
    position -- keeps the corrupted run's own value. Returns the hook handle."""
    h_l = clean_cache[layer].to(device)

    def hook(mod, inp, out, _pos=pos, _h=h_l):
        hs = (out[0] if isinstance(out, tuple) else out).clone()
        hs[:, _pos, :] = _h[:, _pos, :]
        return (hs,) + out[1:] if isinstance(out, tuple) else hs

    return _submodule(model, layer, kind).register_forward_hook(hook)


def patch_module_window(model, clean_cache, center, pos, kind, device, window, n_layers):
    """Like patch_module_position, but restores `window` consecutive layers' submodule outputs at
    the same position, centred on `center` and clipped to [0, n_layers). An odd `window` is
    symmetric; `window=1` is exactly patch_module_position. Returns a list of handles -- the
    caller must remove all of them."""
    half = window // 2
    lo, hi = max(0, center - half), min(n_layers, center + half + 1)
    return [patch_module_position(model, clean_cache, l, pos, kind, device) for l in range(lo, hi)]


def causal_trace_sweep_module(model, clean_cache, inp_corr, n_layers, pos_start, T, tok_a, tok_b,
                               ld_clean, ld_corr, device, kind, window=1, show_progress=True):
    """`causal_trace_sweep` restricted to one submodule per layer (see patch_module_position).
    `clean_cache` must come from `cache_module_outputs(..., kind)` with the same `kind`.

    `window` > 1 restores that many consecutive layers at once, centred on the row's own layer --
    Meng et al.'s sliding-window trace (they use ~10 for MLPs). Entry (l, i) then means "restore
    the window centred on layer l at position i", not "restore layer l"; rows near layer 0 and
    layer n_layers-1 are clipped and so patch fewer layers than the rows in the middle.

    Returns (recovery, indirect_effect), each shaped (n_layers, T)."""
    P_corr = prob_of(model, inp_corr, tok_a)
    recovery = np.zeros((n_layers, T))
    indirect_effect = np.zeros((n_layers, T))

    desc = f'Causal trace ({kind}' + (f', window {window})' if window > 1 else ')')
    layer_iter = tqdm(range(n_layers), desc=desc) if show_progress else range(n_layers)
    for l in layer_iter:
        for pos in range(pos_start, T):
            hnds = patch_module_window(model, clean_cache, l, pos, kind, device, window, n_layers)
            with torch.no_grad():
                logits_p = model(inp_corr).logits[0, -1]
            for hnd in hnds:
                hnd.remove()
            ld_p = (logits_p[tok_a] - logits_p[tok_b]).item()
            recovery[l, pos] = (ld_p - ld_corr) / (ld_clean - ld_corr)
            indirect_effect[l, pos] = torch.softmax(logits_p, dim=-1)[tok_a].item() - P_corr

    return recovery, indirect_effect


# ── Individual attention-head patching ──────────────────────────────────────
# Geva, Bastings, Filippova & Globerson, 2023 ("Dissecting Recall of Factual Associations
# in Auto-Regressive Language Models") use "attention knockout" at individual (layer, head,
# source position -> target position) sites to trace when information moves between token
# positions; Wang, Variengien, Conmy, Shlegeris & Steinhardt, 2022 ("Interpretability in the
# Wild") patch the same pre-o_proj per-head vector to isolate one head's causal contribution.
# `patch_position`/`causal_trace_sweep3` above restore a whole layer's residual stream at one
# position; these restore one (or a chosen group of) attention head's own contribution instead.

def cache_all_heads_context(model, input_ids, n_layers):
    """Run `input_ids` and cache each layer's self_attn.o_proj *input* -- the concatenated
    per-head context vectors before the output projection mixes them -- at every position.
    Returns {layer_idx: (1, T, d_model) tensor on CPU}."""
    cache = {}

    def make_hook(ll):
        def h(mod, inp):
            cache[ll] = inp[0].detach().cpu()
        return h

    handles = [model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(make_hook(l))
               for l in range(n_layers)]
    with torch.no_grad():
        _ = model(input_ids)
    for hh in handles:
        hh.remove()
    return cache


def patch_heads_position(model, clean_ctx, layer, heads, pos, head_dim, device):
    """Register a forward pre-hook on model.model.layers[layer].self_attn.o_proj that
    overwrites position `pos`'s pre-projection context vector, for only the given `heads`
    (a list of head indices), with its cached clean value (clean_ctx[layer]). Every other
    head/position keeps the corrupted run's own value. Returns the hook handle -- caller
    must call .remove()."""
    ctx_l = clean_ctx[layer].to(device)
    slices = [(h * head_dim, (h + 1) * head_dim) for h in heads]

    def hook(mod, inp, _pos=pos, _ctx=ctx_l, _slices=slices):
        hs = inp[0].clone()
        for s, e in _slices:
            hs[:, _pos, s:e] = _ctx[:, _pos, s:e]
        return (hs,) + inp[1:]

    return model.model.layers[layer].self_attn.o_proj.register_forward_pre_hook(hook)


def head_sweep_single(model, clean_ctx, inp_corr, layer, pos, n_heads, head_dim, compute_metrics, device,
                       show_progress=True):
    """position_recovery from patching one attention head at a time at (layer, pos).
    Returns an (n_heads,) array."""
    position_recovery = np.zeros(n_heads)
    head_iter = tqdm(range(n_heads), desc=f'Heads @ layer {layer}') if show_progress else range(n_heads)
    for h in head_iter:
        hnd = patch_heads_position(model, clean_ctx, layer, [h], pos, head_dim, device)
        with torch.no_grad():
            logits_p = model(inp_corr).logits[0, -1]
        hnd.remove()
        _, _, position_recovery[h] = compute_metrics(logits_p)
    return position_recovery


def head_sweep_cumulative(model, clean_ctx, inp_corr, layer, pos, head_order, head_dim, compute_metrics, device,
                           show_progress=True):
    """position_recovery from patching the top-k heads together (`head_order`, most
    important first), for k = 1..len(head_order). Returns a (len(head_order),) array."""
    position_recovery = np.zeros(len(head_order))
    ks = range(1, len(head_order) + 1)
    k_iter = tqdm(ks, desc=f'Cumulative heads @ layer {layer}') if show_progress else ks
    for k in k_iter:
        hnd = patch_heads_position(model, clean_ctx, layer, head_order[:k], pos, head_dim, device)
        with torch.no_grad():
            logits_p = model(inp_corr).logits[0, -1]
        hnd.remove()
        _, _, position_recovery[k - 1] = compute_metrics(logits_p)
    return position_recovery


# ── Linear probes (Alain & Bengio, 2016) ────────────────────────────────────

def fit_eval_logistic(X, y, idx_train, idx_test, max_iter=2000):
    """Standard linear-probe methodology: a regularized logistic-regression classifier
    trained on a frozen (layer, position) residual-stream feature to test whether that
    representation linearly encodes `y`. Returns (acc_train, acc_test)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(X[idx_train], y[idx_train])
    acc_train = accuracy_score(y[idx_train], clf.predict(X[idx_train]))
    acc_test = accuracy_score(y[idx_test], clf.predict(X[idx_test]))
    return acc_train, acc_test


# ── Site ranking (top positions/layers by an averaged effect matrix) ───────

def rank_top_sites(metric, n_positions, layers_per_position=1, position_agg='max'):
    """Pick the `n_positions` token positions with the highest own aggregate-over-layers
    score (position_agg='max' or 'mean' of `metric`, shape (n_layers, T), over the layer
    axis), then for each position its own top `layers_per_position` layers (by that
    position's own values, argsort descending). Returns a list of (position, layer)
    tuples ordered by increasing position, then descending layer rank within a position."""
    agg = metric.max(axis=0) if position_agg == 'max' else metric.mean(axis=0)
    top_positions = sorted(np.argsort(agg)[::-1][:n_positions].tolist())
    sites = []
    for pos in top_positions:
        top_layers = np.argsort(metric[:, pos])[::-1][:layers_per_position].tolist()
        sites.extend((pos, int(l)) for l in top_layers)
    return sites


def layer_color_map(sites, cmap_name='tab10'):
    """Consistent color per unique layer across a list of (position, layer) sites, so a
    layer appearing at multiple positions keeps the same color throughout a plot."""
    unique_layers = sorted({l for _, l in sites})
    cmap = plt.get_cmap(cmap_name)
    return {l: cmap(i % 10) for i, l in enumerate(unique_layers)}


# ── Patching-outcome classification (does patching flip the generated token?) ──

OUTCOME_ORDER = ['recovered', 'copied_corrupted_value_at_clean_max_pos',
                  'gave_corrupted_prompt_max', 'other']
OUTCOME_COLORS = {'recovered': '#2ca02c',
                   'copied_corrupted_value_at_clean_max_pos': '#d62728',
                   'gave_corrupted_prompt_max': '#1f77b4',
                   'other': '#7f7f7f'}


def classify_patched_outcome(gen_tok, clean_max_tok, corrupted_val_tok, corrupted_max_tok):
    """Which of the 3 candidate tokens (if any) the patched generation's first token
    matches: the clean run's correct max, the corrupted value sitting at the clean-max's
    position, or the corrupted prompt's own naturally-correct (smaller) max."""
    if gen_tok == clean_max_tok:
        return 'recovered'
    if gen_tok == corrupted_val_tok:
        return 'copied_corrupted_value_at_clean_max_pos'
    if gen_tok == corrupted_max_tok:
        return 'gave_corrupted_prompt_max'
    return 'other'


DETAIL_CATEGORIES = [
    'full clean-max match', 'first-token-only clean-max match',
    'full corrupted-value match', 'first-token-only corrupted-value match',
    'full corrupted-prompt-max (nb) match', 'first-token-only corrupted-prompt-max (nb) match',
    'other',
]
DETAIL_COLORS = {
    'full clean-max match': '#2ca02c',                                # = 'recovered'
    'first-token-only clean-max match': '#98df8a',                    #   (lighter green)
    'full corrupted-value match': '#d62728',                          # = 'copied_corrupted_value_at_clean_max_pos'
    'first-token-only corrupted-value match': '#ff9896',              #   (lighter red)
    'full corrupted-prompt-max (nb) match': '#1f77b4',                # = 'gave_corrupted_prompt_max'
    'first-token-only corrupted-prompt-max (nb) match': '#aec7e8',    #   (lighter blue)
    'other': '#7f7f7f',
}


def classify_patched_outcome_detailed(gen_tok1, gen_tok2, clean_max_tok1, clean_max_tok2,
                                       corrupted_val_tok1, corrupted_val_tok2,
                                       corrupted_max_tok1, corrupted_max_tok2):
    """Like classify_patched_outcome, but using the first TWO generated tokens to
    distinguish a full match (both digits) from a first-token-only match (right leading
    digit, wrong second digit) for whichever of the 3 candidates the first token matches."""
    if gen_tok1 == clean_max_tok1:
        return 'full clean-max match' if gen_tok2 == clean_max_tok2 else 'first-token-only clean-max match'
    if gen_tok1 == corrupted_val_tok1:
        return 'full corrupted-value match' if gen_tok2 == corrupted_val_tok2 else 'first-token-only corrupted-value match'
    if gen_tok1 == corrupted_max_tok1:
        return 'full corrupted-prompt-max (nb) match' if gen_tok2 == corrupted_max_tok2 else 'first-token-only corrupted-prompt-max (nb) match'
    return 'other'


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_position_layer_heatmap(ax, data, row_labels, cmap='viridis', vmin=None, vmax=None,
                                 title='', xlabel='Layer'):
    """imshow of a (n_layers, T) matrix with layer on x, token position on y.
    `data`/`row_labels` should already be cropped to the rows you want shown."""
    im = ax.imshow(data.T, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='upper')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Token position')
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title)
    return im


def plot_metric_boxplot(ax, data_stack, sites, layers_per_position, site_labels,
                         layer_colors, ylabel='', title=''):
    """One box per (position, layer) site, grouped by position (increasing order
    left-to-right), `layers_per_position` boxes side by side within each group, colored
    by layer via `layer_colors` (see layer_color_map). `data_stack` is (N_pairs, n_layers,
    T) per-pair metric values; `sites`/`site_labels` come from rank_top_sites (same order)."""
    n_groups = len(sites) // layers_per_position
    box_width = 0.8 / layers_per_position
    box_x = [gi - 0.4 + (li + 0.5) * box_width
             for gi in range(n_groups) for li in range(layers_per_position)]

    for x, (pos, layer) in zip(box_x, sites):
        bp = ax.boxplot(data_stack[:, layer, pos], positions=[x],
                         widths=box_width * 0.9, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(layer_colors[layer])

    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_xticks(box_x)
    ax.set_xticklabels(site_labels)
    ax.set_xlabel('Token position / layer')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    legend_elems = [Patch(facecolor=layer_colors[l], label=f'Layer {l}') for l in sorted(layer_colors)]
    ax.legend(handles=legend_elems, loc='upper left', fontsize=8)
    return ax


def plot_outcome_stacked_bar(ax, tally, labels, n_pairs, outcome_order=OUTCOME_ORDER,
                              outcome_colors=OUTCOME_COLORS, title=''):
    """Stacked bar of outcome fractions per label (e.g. per patched site). `tally` is
    {label: {outcome: count}} (see classify_patched_outcome)."""
    bottoms = np.zeros(len(labels))
    for outcome in outcome_order:
        counts = np.array([tally[label][outcome] for label in labels])
        fractions = counts / n_pairs
        ax.bar(labels, fractions, bottom=bottoms, label=outcome, color=outcome_colors[outcome])
        for x, (frac, count) in enumerate(zip(fractions, counts)):
            if frac > 0.03:
                ax.text(x, bottoms[x] + frac / 2, f'{count}', ha='center', va='center',
                        fontsize=9, color='white')
        bottoms += fractions
    ax.set_ylabel('Fraction of pairs')
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    return ax


def plot_outcome_detailed_stacked_bar(ax, tally_detail, labels, n_pairs,
                                       categories=DETAIL_CATEGORIES, colors=DETAIL_COLORS, title=''):
    """Stacked bar of detailed (full-match vs first-token-only-match) outcome fractions
    per label. `tally_detail` is {label: {category: count}} (see
    classify_patched_outcome_detailed). Each category's color is a lighter/darker shade of
    its plot_outcome_stacked_bar counterpart, so this reads as a finer-grained breakdown of
    that plot."""
    bottoms = np.zeros(len(labels))
    for cat in categories:
        counts = np.array([tally_detail[label][cat] for label in labels])
        fractions = counts / n_pairs
        ax.bar(labels, fractions, bottom=bottoms, label=cat, color=colors[cat])
        for x, (frac, count) in enumerate(zip(fractions, counts)):
            if frac > 0.03:
                ax.text(x, bottoms[x] + frac / 2, f'{count}', ha='center', va='center',
                        fontsize=8, color='white')
        bottoms += fractions
    ax.set_ylabel('Fraction of pairs')
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    return ax
