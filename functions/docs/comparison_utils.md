# comparison_utils.py — Documentation

## Overview

Consolidates function definitions that were copy-pasted, identically or near-identically,
across `where_compare.ipynb`, `where_compare_truefalse.ipynb`, `which_components.ipynb`,
`isolate_postion_value.ipynb`, `multi_compare.ipynb` (in `e11_comparison_largemodels/`), and
`compare_objects.ipynb` (in `e12_compare_objects/`). Existing notebooks were **not** modified to
import from this file — it exists so new notebooks (starting with
`first_numbers_then_compare.ipynb`) don't redefine them again.

Every function takes `model`/`tokenizer`/`device` as explicit arguments rather than reading them
off notebook globals, so behavior doesn't depend on what a notebook happened to name its
variables.

Import from a notebook two directories below `llminterp/` via:
```python
import sys
sys.path.append('../')
from functions.comparison_utils import ...
```

---

## Token-position labeling

These all originally hardcoded the anchor phrase `'The maximum of '`; here it's a parameter
(`anchor`) so a notebook with a different prompt template (e.g.
`first_numbers_then_compare.ipynb`'s `'Given the numbers '`) can reuse the same logic.

#### `num_char_positions(prompt, num_a, num_b, anchor)`
Character spans `(a_start, a_end, b_start, b_end)` of the two numbers in the question clause
introduced by `anchor`. Uses `rfind(anchor)` so a few-shot example repeating the same phrase
earlier in the prompt is skipped in favor of the real (last) occurrence.

#### `tok_indices_for_chars(offsets_np, c_start, c_end)`
Token indices whose char span overlaps `[c_start, c_end)` — maps a character range to the
(possibly multiple) tokenizer tokens covering it.

#### `question_start_position(prompt, offsets_np, anchor)`
First token index of the real question (the last occurrence of `anchor`), skipping any
instruction/few-shot preamble.

#### `template_token_label(prompt, num_a, num_b, offsets_np, tokens, pos, anchor)`
Display label for token position `pos`: `'a1'`/`'a2'` for `num_a`'s own token(s), `'b1'`/`'b2'`
for `num_b`'s, else the literal decoded token — for heatmap row labels.

#### `discrete_position_label(prompt, num_a, num_b, offsets_np, tokens, pos, T, anchor)`
Same as `template_token_label`, but the final prompt position is flagged `'[last tok]'` — for
plots that single out one position without showing the whole sequence.

---

## Forward-pass readouts

#### `get_attentions(model, input_ids)`
Single forward pass with `output_attentions=True`. Returns `(n_layers, n_heads, T, T)` stacked
attention weights on CPU (float32).

#### `logits_of(model, ids)` / `ld_of(model, ids, tok_a, tok_b)` / `prob_of(model, ids, tok)`
Last-position logits; logit difference `logit[tok_a] - logit[tok_b]`; softmax probability of one
token — the three readouts causal tracing and patching demos compare across clean/corrupted/
patched runs.

#### `hidden_states_of(model, input_ids)`
Single forward pass with `output_hidden_states=True`. Returns a tuple of `n_layers+1` tensors on
CPU: index 0 is the embedding output, index `l>=1` is the residual stream after decoder layer
`l-1` — the same value `cache_all_layers`'s forward hooks capture, but for every layer including
the embedding in one pass, and usable directly as `{l: hs[l+1] for l in range(n_layers)}` to feed
`patch_position`/`causal_trace_sweep`.

---

## Activation caching / single-position patching (ROME-style causal trace)

Implements Meng, Bau, Andonian & Belinkov 2022 ("Locating and Editing Factual Associations in
GPT", Sec. 3 / Fig. 2): restore one hidden state $h_i^{(l)}$ at a time from a clean run into a
corrupted run, and measure how much that one restoration recovers the clean answer.

#### `cache_all_layers(model, input_ids, n_layers)`
Runs `input_ids` and returns `{layer_idx: (1, T, d_model) tensor}` — the residual-stream hidden
state leaving each decoder layer, every position, on CPU.

#### `patch_position(model, clean_hidden, layer, pos, device)`
Registers a forward hook on `model.model.layers[layer]` that overwrites position `pos`'s hidden
state with `clean_hidden[layer]`'s value at that position. Returns the hook handle (caller must
call `.remove()`).

#### `generate_n_patched(model, input_ids, hook_handle, n_new, device)`
Greedy-generates `n_new` tokens with `hook_handle`'s patch active (no KV-cache reuse, since the
cached clean state only covers the original prompt length), removes the hook, and returns the
generated token ids as a plain list.

#### `causal_trace_sweep(model, clean_hidden, inp_corr, n_layers, pos_start, T, tok_a, tok_b, ld_clean, ld_corr, device, show_progress=True)`
Sweeps every `(layer, position)` pair from `pos_start` to `T-1`, patching each in turn into
`inp_corr` and reading out both metrics from the same restored forward pass:
- **Recovery** `(l,i) = (LD_patched - LD_corr) / (LD_clean - LD_corr)` — 0=none, 1=full.
- **Indirect Effect** `(l,i) = P_patched[tok_a] - P_corr[tok_a]` — raw probability difference,
  Meng et al.'s own metric (not normalized).

Positions before `pos_start` are left at 0 — valid only when the clean/corrupted prompts are
identical up to that point (true whenever only the numbers themselves differ). Pass
`show_progress=False` when calling this once per pair inside an outer loop that already has
its own progress bar (otherwise it spawns one nested tqdm bar per pair).

---

## Module-restricted patching (MLP-only / attention-only traces)

Meng et al., 2022, Sec. 3.3. The residual-stream trace above answers *which (layer, position)
states matter*; rerunning it while restoring only one submodule's output answers *which component
wrote the state that mattered*. `patch_position` restores everything a decoder layer produced at a
position; these restore only that layer's MLP output, or only its attention output (post-`o_proj`,
i.e. all heads of the layer summed), leaving the other component running on corrupted input.

**Single layer vs window.** Meng et al. restore a sliding window of ~10 consecutive layers,
because one MLP's own contribution is often too small to move the answer alone. The default here
is a **single** layer (`window=1`), which keeps the result the same `(n_layers, T)` shape as
`causal_trace_sweep` and therefore directly comparable to it; `window=3` (or 10) gives the paper's
windowed trace, which is correspondingly brighter. Either way, expect smaller effects than the
residual-stream trace.

#### `MODULE_KINDS` / `_submodule(model, layer, kind)`
`kind` is `'mlp'` or `'attn'`, resolving to `model.model.layers[layer].mlp` or `.self_attn`.

#### `cache_module_outputs(model, input_ids, n_layers, kind)`
Runs `input_ids` and caches each layer's `kind` submodule output at every position;
`{layer: (1, T, d_model)}` on CPU. Attention modules return a `(output, weights)` tuple and MLPs a
bare tensor — both are handled, so the same function serves either `kind`.

#### `patch_module_position(model, clean_cache, layer, pos, kind, device)`
Forward hook overwriting position `pos` of that submodule's output with its cached clean value.
The layer's *other* submodule, and this one at every other position, keep the corrupted run's own
values. Returns the handle — the caller must `.remove()` it.

#### `patch_module_window(model, clean_cache, center, pos, kind, device, window, n_layers)`
`patch_module_position` over `window` consecutive layers at the same position, centred on
`center` and clipped to `[0, n_layers)`. An odd `window` is symmetric; `window=1` is exactly
`patch_module_position`. Returns a **list** of handles — remove all of them.

#### `causal_trace_sweep_module(model, clean_cache, inp_corr, n_layers, pos_start, T, tok_a, tok_b, ld_clean, ld_corr, device, kind, window=1, show_progress=True)`
`causal_trace_sweep` restricted to one submodule per layer. Same arguments, same two returned
matrices, plus `kind` and `window`; `clean_cache` must come from `cache_module_outputs` with the
*same* `kind`, and is reusable across windows.

With `window > 1`, entry `(l, i)` means **"restore the window centred on layer `l` at position
`i`"**, not "restore layer `l`" — read the heatmap accordingly. Rows near layer 0 and layer
`n_layers-1` are clipped and so patch fewer layers than rows in the middle, which makes the top
and bottom edges of a windowed heatmap slightly less comparable to its interior.

Verified on a randomly-initialised 3-layer `Qwen2ForCausalLM` (transformers 4.50.3): both hooks
fire, patching a clean run with its own cache is a no-op, patching a corrupted run moves the
logits, hooks are removed cleanly (including after a windowed sweep), rows before `pos_start` stay
0, `window=3` clips to the right widths at both ends, and `window=1` reproduces the single-layer
sweep exactly.

---

## Linear probes (Alain & Bengio, 2016, "Understanding Intermediate Layers Using Linear
Classifier Probes"; see also Belinkov, 2022 for the survey/critique)

#### `fit_eval_logistic(X, y, idx_train, idx_test, max_iter=2000)`
Trains an `sklearn.linear_model.LogisticRegression` on `X[idx_train]`/`y[idx_train]` (a frozen
residual-stream feature at one fixed (layer, position)) and returns `(acc_train, acc_test)` —
tests whether that representation linearly encodes `y`. Caller is responsible for building a
single stratified `idx_train`/`idx_test` split (e.g. via `sklearn.model_selection.train_test_split`)
and reusing it across every (layer, position) probe, so results are comparable.

---

## Site ranking (for box plots / patching-outcome bars over "top" positions)

#### `rank_top_sites(metric, n_positions, layers_per_position=1, position_agg='max')`
Two-step data-driven site selection used throughout `where_compare.ipynb`'s multi-pair analysis:
(1) rank token positions by their own aggregate over layers (`position_agg='max'` — best single
layer at that position — or `'mean'` — average across layers) and keep the top `n_positions`;
(2) for each, its own top `layers_per_position` layers (by that position's own values, not the
aggregate). Returns `[(position, layer), ...]`, ordered by increasing position.

#### `layer_color_map(sites, cmap_name='tab10')`
`{layer: color}` keyed by actual layer index (not rank), so a layer appearing at multiple
positions keeps one consistent color across a figure.

---

## Patching-outcome classification

#### `classify_patched_outcome(gen_tok, clean_max_tok, corrupted_val_tok, corrupted_max_tok)`
Buckets a patched generation's first token into one of `OUTCOME_ORDER` = `['recovered',
'copied_corrupted_value_at_clean_max_pos', 'gave_corrupted_prompt_max', 'other']` — did patching
recover the clean answer, leave the corrupted value in place, fall back to the corrupted prompt's
own (smaller) natural answer, or something else. `OUTCOME_COLORS` gives each a fixed color
(green/red/blue/gray) for stacked-bar plots.

#### `classify_patched_outcome_detailed(gen_tok1, gen_tok2, clean_max_tok1, clean_max_tok2, corrupted_val_tok1, corrupted_val_tok2, corrupted_max_tok1, corrupted_max_tok2)`
Same 3 candidates as `classify_patched_outcome`, but using the first TWO generated tokens to
split each into a full match (both digits) vs. a first-token-only match (right leading digit,
wrong second digit) — one of `DETAIL_CATEGORIES` (7 categories: full/first-token-only for each
of the 3 candidates, plus `'other'`). `DETAIL_COLORS` reuses each candidate's `OUTCOME_COLORS`
hue for its full-match category, with a lighter shade for the first-token-only counterpart, so
a detailed plot reads as a finer-grained breakdown of the simple one.

---

## Plotting

#### `plot_position_layer_heatmap(ax, data, row_labels, cmap, vmin, vmax, title, xlabel='Layer')`
`imshow` of a `(n_layers, T)` matrix with layer on x, token position on y (row_labels as
y-ticks). `data`/`row_labels` should already be cropped to the rows to display. Returns the
`imshow` handle for `plt.colorbar`. `xlabel` is overridable (e.g. `'Layer (0=embeddings)'` for a
probe grid that includes the embedding layer, unlike causal-trace layers which don't).

#### `plot_metric_boxplot(ax, data_stack, sites, layers_per_position, site_labels, layer_colors, ylabel, title)`
One box per `(position, layer)` site from `rank_top_sites`, grouped by position,
`layers_per_position` boxes side by side per group, colored by layer via `layer_color_map`.
`data_stack` is `(N_pairs, n_layers, T)` per-pair metric values (e.g. `np.stack(all_recovery)`).

#### `plot_outcome_stacked_bar(ax, tally, labels, n_pairs, outcome_order=OUTCOME_ORDER, outcome_colors=OUTCOME_COLORS, title)`
Stacked bar of outcome fractions per label (e.g. per patched site), from a
`{label: {outcome: count}}` tally built with `classify_patched_outcome`.

#### `plot_outcome_detailed_stacked_bar(ax, tally_detail, labels, n_pairs, categories=DETAIL_CATEGORIES, colors=DETAIL_COLORS, title)`
Same layout as `plot_outcome_stacked_bar`, but for a `{label: {category: count}}` tally built
with `classify_patched_outcome_detailed` — 7 stacked segments per label instead of 4.
