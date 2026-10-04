# AI-generated code: what was wrong and what caught it

Every entry is a real event from building this repo with an AI coding assistant. Keep
appending as you run the GPU experiments; the useful ones for interviews are where a test or
diagnostic, not luck, caught the problem.

| # | Where | What was wrong | What caught it | Fix | Guard now |
|---|---|---|---|---|---|
| 1 | `models/sasrec.py`, `models/train.py` | Cold items were held out of every training *sequence*, but the full-softmax loss still scored them as negatives on every step. The model learned to push never-seen items down, so the content two-tower recommended **0%** cold items on the synthetic smoke run despite having their content. This is a protocol leak, not just a tuning issue: the cold set is supposed to be held out of training entirely. | Smoke-run diagnostics: `cold_slot_share = 0.0000` for a content-based model, which should be able to rank unseen items. | `exclude_items` masks cold items out of the CE softmax and out of the BCE negative pool. | `tests/test_models.py::test_cold_items_get_no_training_signal` (failed before the fix: no gradient may reach a cold item's embedding row, for both losses). |
| 2 | `utils.py` | Run-directory config logging called `yaml.safe_dump` on `torch.__version__`, which is a `TorchVersion` object, not a `str`; every training command crashed before training. | First smoke run (`scripts/smoke.sh`). | `str(torch.__version__)`. | The smoke run. |
| 3 | `models/sasrec.py` (caught in review, not by a test) | First draft combined a causal mask with a key-padding mask on left-padded sequences. The leading padded query positions then have every key masked, softmax over all `-inf` gives NaN, and NaN values leak into later layers through `0 * NaN`. | Reading the attention-mask logic before running. | Single combined mask in which a padded query may attend only to itself. | `test_sasrec_is_causal` exercises left padding; add a NaN assertion if you touch this code. |
| 4 | `cli.py` (TIGER evaluation) | The "invalid-ID rate without trie constraints" check reported unconstrained Recall@10 on the first 2,000 test users next to constrained recall on all 22,360. The first run showed 0.0% invalid IDs but recall 0.021 vs 0.051, which looked like constraints mattering a lot. | A contradiction: if no beam ever leaves the catalog, constrained and unconstrained beam search must return identical lists. Re-running both on the same 300 users gave identical items and scores. | Both decodings are now measured on the same users and reported side by side; added `eval-tiger` to re-evaluate a saved model without retraining. | Structural (one code path computes both on one user list). |
| 5 | `cli.py` (`table`) | After the fix in #4 added a scalar `test_beam` field to `metrics.json`, the table aggregator, which treated every key starting with `test` as a result block, crashed with `'int' object is not subscriptable`. | Regenerating the smoke tables after the refactor. | Aggregate only dict-valued `test*` blocks; the field is now `beam`. | `tests/test_cli_table.py` (mean ± std across seeds, scalar `test*` fields ignored). |
| 6 | `ids/encoders.py` | Image embedding assumed `model.get_image_features()` returns a tensor. In transformers 5.x it returns a `BaseModelOutputWithPooling` (embedding in `.pooler_output`), so `embed --variant image` crashed on Colab. Written against the older API and never run with real weights in the build sandbox. | First real run on Colab (A100). | `image_features_tensor()` accepts both return types. | `tests/test_encoders.py::test_image_features_tensor_handles_old_and_new_transformers`. |
| 7 | `notebooks/visid_colab.ipynb` | The copy-from-Drive step used `shutil.ignore_patterns('data', ...)`, which skips *every* folder named `data`, including the code package `visid/data/`; the first fix used `os.path.samefile`, which misbehaves on the Google Drive mount and skipped it again. | Colab test collection (`No module named 'visid.data'`), twice. | Compare normalized path strings; assert `visid/data/splits.py` exists after copying. | The assert in the notebook. |

## Diagnostics that fired during real runs (not code bugs, but worth knowing)
- **Codebook collapse, then recovery.** On the attribute-bag IDs (real Beauty interactions),
  level-0 utilization fell to 17% at epoch 50 (`collapse_warning_l0=True` in
  `results/beauty_s3rec/ids_attr/seed0/rqvae_log.jsonl`); periodic dead-code re-seeding brought it
  back to 95% by epoch 100 and 100% by epoch 300.
- **Content that cannot separate items.** The same run produced only 3,180 distinct 3-code
  prefixes for 12,101 items (87% of items share a prefix; largest group 160). The attribute
  file has 3,120 distinct brand/category bags, so the RQ-VAE was faithfully reproducing the
  resolution of its input; the 4th token carries the rest. Expect the same diagnostic to
  separate text from image IDs.
