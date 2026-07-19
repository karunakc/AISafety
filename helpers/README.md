# helpers/

Standalone convenience scripts that sit on top of the main diffing methods
(`diffing/method1_cosine_similarity.py` .. `method4_bypass.py`,
`method5_steered_distribution.py`, `method6_cka.py`) or the `final_results/`
snapshot, but aren't methods or data in their own right. Nothing here is
wired into a Modal entrypoint (`modal/modal_app.py`) -- these are local-only.

## Comparison/overlay scripts (sit on top of a diffing method)

- `method2_compare.py` -- Method 2, overlaying multiple tested models against
  one base control on a single plot. Same computation as
  `diffing/method2_projection.py`, just with N tested curves instead of 1.
- `method2_all_layers_compare.py` -- same idea, for `diffing/method2_all_layers.py`.
- `combine_plots.py` -- generic overlay for methods 2/3/4's output JSON.
- `compare_cka_runs.py` -- overlay for `diffing/method6_cka.py`'s output JSON.

## Re-plotting / recompute scripts (sit on top of final_results/)

- `make_bypass_induce_kl_plots.py`, `make_method2_plots.py`,
  `make_method5_plots.py` -- re-plot `final_results/`'s cached JSON in a
  cleaner style. Read-only, no recomputation.
- `recompute_direction.py`, `recompute_method2.py` -- recompute from
  `final_results/`'s cached tensors under a new setting (e.g. a different
  `max_layer_frac`), writing to a new top-level `real_results/` directory.

## Why these moved here

`diffing/method2_all_layers.py` and `diffing/method3_all_layers.py` are also
comparison/sweep variants of a main method, but stayed in `diffing/` because
they're wired into live Modal entrypoints (`diffing_method2_all_layers`,
`diffing_method3_all_layers` in `modal/modal_app.py`) -- moving them would
require adding a matching Modal mount + module loader.
