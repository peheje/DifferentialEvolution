# Settlement Optimizer Experiment Notes

This branch explores whether the existing Kotlin Differential Evolution implementation can solve group expense settlement: given net balances, find payments that bring everyone to zero while using as few transactions as possible.

## Current Shape

- Net balances use this convention: positive means a person should receive money, negative means a person should pay money.
- The Kotlin optimizer still uses a `DoubleArray -> Double` score, but settlement scenarios now define people, balances, and generated payment pairs.
- Payment dimensions are restricted to debtor-to-creditor edges only. This removes self-payments, creditor-to-debtor payments, and pass-through/cycle paths that made the search space unnecessarily noisy.
- The harness contains small scenarios plus a 10-person scenario. It computes the exact minimum transaction count with a subset-DP oracle over zero-sum groups, so larger examples do not need manually trusted expected counts.

## Lessons Learned

- Allowing every directed pair creates misleading local basins. DE can find perfectly balanced settlements that use extra pass-through payments.
- Debtor-to-creditor-only edges are a useful representation improvement and do not remove meaningful optimal settlements for net-balance settlement problems.
- A deterministic greedy settlement pass can trivially create sparse settlements, but it bypasses DE and defeats the purpose of this experiment. Do not use that as the optimizer result if the goal is to test DE.
- The 10-person case is a useful red test. The exact oracle says the optimum is 6 transactions, while current DE still finds a much denser balanced solution.
- A first smooth sparsity attempt, `sum(sqrt(amount))`, helped only slightly in one run: the 10-person case moved from about 19 transactions to about 17, still far from 6.
- The core challenge is objective shape: balance error is smooth and heavily weighted, while transaction count is thresholded/discontinuous. Once DE reaches a dense balanced settlement, removing an edge requires coordinated changes across several dimensions.

## Fitness Experiments So Far

- Baseline blended score used balance error, hard transaction count, and total money moved. It solved the small cases but stayed dense on the 10-person case.
- Increasing phase-2 hard transaction penalties alone made results worse. It did not help if the search had not already found a sparse basin.
- A two-phase objective worked better: while unsettled, balance dominates but smooth sparsity still applies; once settled enough, exact transaction count dominates.
- Replacing `sqrt(amount)` with `amount / (amount + k)` gave a more useful smooth approximation of active transaction count.
- Current best settings use early smooth sparsity pressure with `softTransactionScale = 0.1` and `unsettledSoftTransactionPenalty = 100.0`.
- Current best observed result on the 10-person scenario is 7 transactions. The exact oracle says 6, so this is close but still red.
- Sharpening `softTransactionScale` to `0.01` still found 7 transactions, not 6.
- Raising `unsettledSoftTransactionPenalty` to `300.0` still found 7 transactions, not 6.
- Adding a debtor split penalty, intended to discourage one debtor paying multiple creditors, got worse in one run: 8 transactions.
- The remaining failure mode is usually one extra split that would require a coordinated reshuffle across multiple edges.

## Representation Shortcut

- The most effective change was switching `xs` from payment amounts to random keys / edge priorities.
- DE now evolves an ordering of debtor-to-creditor edges. A deterministic decoder walks edges in that order and fills in the maximum valid amount from remaining debtor/creditor balances.
- This avoids asking DE to discover exact money amounts. DE only influences which edges are considered earlier.
- With this representation, the 10-person scenario reached the exact oracle optimum of 6 transactions in repeated harness runs.
- This is not the same as the earlier greedy bypass, because the decoder depends on the DE-evolved edge order. However, the decoder is doing most of the arithmetic and feasibility work.

## Larger Scenarios and MILP Check

- A 20-person constructed case with ten independent debtor-creditor pairs was useful as a short-lived scale smoke test, but it was removed because it was too contrived and both DE and greedy should obviously solve it.
- A mixed 20-person case is now included instead. It is not small enough for the subset-DP oracle to be a comfortable default, so a Python/SciPy MILP side script was added under `tools/settlement_milp.py`.
- The MILP formulation minimizes binary edge-used variables while satisfying exact debtor and creditor flow constraints.
- The MILP proves the mixed 20-person case has an optimum of 13 transactions.
- The random-key DE representation found 13 transactions on that case, while the simple largest-balance greedy matcher found 14.
- This is the strongest current evidence that DE can add value over the simplest greedy matcher, although MILP remains the better tool when exact optimality is needed.

## Honest Takeaway

- For plain net-balance settlement, there are simpler deterministic algorithms than DE.
- If the goal is a practical settlement engine, a greedy debtor-creditor matcher gives a valid settlement quickly, and exact or MILP-style methods are better fits when optimality is required.
- DE is interesting here mainly as an experiment in representation and fitness shaping. Once we introduce a decoder that settles balances deterministically, DE mostly searches edge ordering rather than solving the settlement amounts.
- The exercise still taught useful lessons: representation dominates fitness, hard `count(nonzero)` objectives are awkward for continuous optimizers, and exact small oracles are valuable for preventing self-deception.
- Current shutdown status: DE has shown usefulness relative to the simple greedy baseline on at least two nontrivial cases, but the best exact solver/reference approach is MILP, not DE.

## Suggested Next Steps

- Keep the exact oracle in the harness and treat the 10-person scenario as the main red test.
- Run repeated trials and report distributions, not one-off results: best score, balance error, transaction count, and active payment list.
- Try one DE-aligned change at a time and preserve red/green evidence for each attempt.
- Explore smoother sparsity objectives, for example `sum(amount / (amount + k))`, stronger split penalties, or staged weights that prioritize sparsity after balance error is small.
- Consider a two-phase DE run: first optimize balance, then restart or continue from the best population with a much stronger sparsity term.
- Consider a more structured representation, such as limiting each debtor to a small number of creditor slots, but avoid rebuilding the answer deterministically after DE.
- If the goal becomes practical settlement rather than testing DE, compare against deterministic greedy and exact/reference solvers separately; do not conflate those with DE performance.
