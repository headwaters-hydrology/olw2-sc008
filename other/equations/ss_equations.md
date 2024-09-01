## Load reduction factors

$$
\underline{\text{load reduction factor (LRF) for suspended sediment}}\\
LRF_{ss} = 1 - (V_{t50}/V_{50})^{1/d} \\
\text{where } d = -0.76 \text{ is the national estimate,}\\
V_{t50} \text{ is the target improvement in visual clarity (m),} \\
\text{and } V_{50} \text{ is the current visual clarity (m).} \\
\underline{\text{Improvement ratio (ImpRatio) for SS and V for routing}}\\
ImpRatio_{ss} = 1 - LRF_{ss} \\
\text{and fortunately for our routing...}\\
ImpRatio_{V} = V_{50}/V_{t50}\\
\text{which means...}\\
ImpRatio_{ss} = ImpRatio_{V}^{1/0.76}\\
\text{rearranged...}\\
ImpRatio_{V} = ImpRatio_{ss}^{0.76}
$$