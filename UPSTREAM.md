# Exact sources used by the matched-budget study

The matched-budget study that supersedes the v1.0.0 evaluation drives this
package from a separate experiment harness. Every component is pinned to a
commit rather than a branch or a tag, because a tag can be moved and these are
the revisions that actually ran.

| Component | Repository | Commit |
|---|---|---|
| StegTransX-CR package (this repository, method under test) | `Mustafa-1982/StegTransX-CR` | `44b4582f505c89f7a9cbde6ed12e4f91030120cd` (tag `v1.0.0`) |
| StegTransX-V1 (matched-budget baseline) | `QQ-Stars/StegTransX` | `8b403756439cb3e5dc9573f98f9abbdc27cb6b69` |
| Experiment harness | `yazanjer/stegtransx-cr-experiments` (private) | see the harness `bootstrap.sh` |

The code in this repository was **not modified** by that study. It was cloned at
the commit above and evaluated as released, which is why the two sets of numbers
can be compared directly.

## Protocol difference in one line

The v1.0.0 runs used early stopping on validation secret PSNR with 200 COCO test
pairs. The later study used a fixed budget of 30,000 optimisation steps with no
early stopping, reported the final checkpoint, and tested on 1,000 COCO pairs
and 100 DIV2K pairs. Both protocols used real encoders for the reported figures.
