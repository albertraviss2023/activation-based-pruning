# Method Math Notes

This document records the pruning score currently used by ReduCNN methods.
Higher score means higher importance and a higher chance of being kept.

## Bundled Methods

`l1_norm`:

```text
score_i = sum(abs(W_i))
```

For PyTorch, `W_i` is output filter `i`. For Keras, `W_i` is output channel
`i`.

`mean_abs_act`:

```text
score_i = mean(abs(A_i))
```

`apoz`:

```text
APoZ_i = mean(ReLU(A_i) == 0)
score_i = 1 - APoZ_i
```

## UI-Registered Research Methods

`chip`:

```text
score_i = nuclear_norm(A) - nuclear_norm(A without channel i)
```

The scores are normalized by the maximum channel score in the layer. This
implements CHIP-style channel independence: removing a more independent channel
causes a larger nuclear-norm change.

`custom_l2`:

```text
score_i = sqrt(sum(W_i^2))
```

`custom_entropy`:

```text
score_i = entropy(histogram(A_i))
```

`custom_class_entropy`:

```text
p(c | i) = class_mean_abs_activation(c, i) / sum_c class_mean_abs_activation(c, i)
score_i = 1 - normalized_entropy(p(c | i))
```

`custom_hrank`:

```text
score_i = mean_sample(matrix_rank(A_i(sample)))
```

`custom_spectral_energy`:

```text
score_i = mean(abs(fft2(A_i))^2)
```

`custom_nisp`:

```text
score_last = final_response_activation_importance
score_l = abs(W_{l+1})^T score_{l+1}
```

This follows the NISP backward propagation recurrence. The final response layer
is initialized from calibration activation energy when available, otherwise
from weight energy.

`custom_senpis`:

```text
IS_{c,i} = abs(loss_c(original) - loss_c(channel_i_zeroed))
score_i = mean_c(IS_{c,i})
```

Then similarity attenuation is applied: for highly similar channels, the channel
with lower importance is multiplied by an attenuation factor.

`custom_tis`:

```text
Taylor_{c,i} = mean(abs(A_i * dL_c/dA_i))
binary_{c,i} = 1 if Taylor_{c,i} >= class_threshold_c else 0
score_i = sum_c binary_{c,i}
```

`custom_reprune`:

Kernel similarity is clustered into redundant groups. The representative with
highest mean similarity inside each group receives coverage credit proportional
to the group size.

`custom_thinet`:

```text
score_i = mean((pooled_activation_i * next_layer_dependency_i)^2)
```

This is the marginal next-layer reconstruction damage score used to rank
channels under the ThiNet objective. Full ThiNet also includes greedy subset
selection and least-squares refinement, which should live in a dedicated
selection/surgery protocol rather than a scalar score function.

## Hybrid Meta-Pruner

The original smooth hybrid method blends three metrics across depth. By default
it uses:

```text
early layer: l1_norm
middle layer: mean_abs_act
late layer: apoz
```

The implementation rank-normalizes each metric before blending and downweights
flat metrics using confidence weighting:

```text
score = w1 * rank(metric1) + w2 * rank(metric2) + w3 * rank(metric3)
```

where weights are depth-dependent and adjusted by metric dispersion.

The recommended research mode is now adaptive:

```text
meta_mode = "adaptive"
```

In adaptive mode, ReduCNN does not blindly stack methods. For each layer it:

1. computes candidate method score vectors;
2. converts each vector into within-layer ranks;
3. computes pairwise rank correlations between methods;
4. selects a cheaper method when it is highly correlated with a more expensive
   method;
5. falls back to a cost-aware weighted blend when methods disagree.

For a simple method `a` and complex method `b`:

```text
if abs(spearman(rank(a), rank(b))) >= threshold
and prune_set_overlap(a, b) >= overlap_threshold:
    use cheaper method a as a proxy for b in that layer
```

The first condition follows the literature use of Spearman rank correlation for
similar filter rankings. The second condition checks that the actual bottom-k
filters selected for pruning are also similar. Exploratory experiments sweep
correlation thresholds such as `0.55`, `0.60`, `0.65`, ..., `0.90` instead of
assuming a single threshold.

Simple methods are not stacked only with other simple methods. They can proxy a
complex method, or one simple method can appear as the cheapest representative
inside a mixed stack. If the available methods are simple-only, the adaptive
hybrid selects a single simple method and records the decision as a simple
baseline, not a hybrid stack.

Otherwise:

```text
score = sum_m weight_m * rank(method_m)
weight_m proportional_to confidence(method_m) / cost(method_m)
```

Cost is preferably measured from full experiments:

```text
cost(method) = median(prune_time_sec + heal_time_sec)
```

The notebooks save these measured costs in the `_efficiency.json` files. When
`hybrid_efficiency_json_path` is provided, the adaptive hybrid uses those
measured values instead of only relying on the default information-cost table.

This makes the hybrid method a layer-wise method-selection policy: use simple
criteria when they make the same pruning decision as complex criteria, and use
weighted fusion only where methods provide complementary rankings.
