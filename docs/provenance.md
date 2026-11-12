# Checkpoint source verification

Verified 2026-10-01. Source repository revision `31c5b904ca1bf2afb4c234a6675c683a4e5fc7cd` and root LICENSE fetched directly; GitHub license API identifies MIT. This establishes code licensing, not checkpoint licensing.

First action before extraction was to actually request the official README's iter3+ AS2M cpt1/cpt2 links with redirects enabled:

| Official link | Actual outcome |
| --- | --- |
| https://1drv.ms/u/s!AqeByhGUtINrgcpoZecQbiXeaUjN8A?e=DasbeC (cpt1) | Redirect to onedrive.live.com; HTTP 403, 873-byte HTML, “The request is blocked.” |
| https://1drv.ms/u/s!AqeByhGUtINrgcpj8ujXH1YUtxooEg?e=E9Ncea (cpt2) | Redirect to onedrive.live.com; HTTP 403, 873-byte HTML, “The request is blocked.” |
| https://onedrive.live.com/download?cid=6B83B49411CA81A7&resid=6B83B49411CA81A7!25955 | HTTP 403, same 873-byte blocked response |
| https://api.onedrive.com/v1.0/drives/6B83B49411CA81A7/items/6B83B49411CA81A7!25955/content | HTTP 401, 64-byte response |

No official checkpoint bytes were obtained. These are liveness failures from this environment, not proof that files were globally deleted. Original SHA comparison is therefore impossible. [unilm issue #1671](https://github.com/microsoft/unilm/issues/1671) reports related access trouble; our own requests supply the evidence above.

Selected cpt2: third-party `WeiChihChen/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2`, immutable HF revision `172e4071a2912392d9151c70750b9a12f55fc54e`; retrieved full 363,145,291 bytes, SHA256 `e5815275a04b6885e7b8af63d120b29bffae2cd2225cf4915e1ec6d819d3022c`. HF API had no license/card metadata. Its README did not supply a grant. The checkpoint config declares fine-tuning, predictor class count 527, embedding dimension 768 and patch size16. Selection follows the tested cpt2 mirror, not an unperformed cpt1 accuracy comparison.

**The official source is unavailable from this test environment. Weights come from this third-party mirror; original provenance cannot be verified and checkpoint licensing is unspecified.** No org mirror, license inference, or upstream issue submission is performed. The repository was made public on 2026-11-12 by the owner's decision with checkpoint licensing still unresolved; it rehosts no weights.
