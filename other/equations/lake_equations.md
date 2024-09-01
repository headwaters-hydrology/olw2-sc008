## Lake equations for changes in input loads
These are the simplified lake equations solved for the change ratio as defined as the new input load (from reductions to to mitigations) to the original input load. These are derived from Abell et al 2019 and Abell et al 2020.

### TP
#### Current conditions
$$
\underline{TP_{lake}}\\
\text{when} \hspace{2mm} z_{max} \gt 7.5: \\
r_{lake} = r_{in}^{1/b}\\
\text{where...}\\
b = 1 + 0.44 \tau^{0.13}\\
r_{*} \text{ is the load change ratio.}\\
z_{max} \text{ is the max depth of the lake in meters.}\\
\tau \text{ is the residence time of the lake in years.}\\
\text{else when} \hspace{2mm} z_{max} \lt= 7.5: \\
r_{lake} = r_{in}
$$

#### "Reference" conditions
$$
\text{when} \hspace{2mm} z_{max} \gt 7.5: \\
r_{lake} = r_{in}^{1/b_{ref}}\\
\text{where...}\\
b_{ref} = 1 + 0.27 \tau^{0.29}\\
\text{else when} \hspace{2mm} z_{max} \lt= 7.5: \\
r_{lake} = r_{in}
$$

### TN
#### Current conditions
$$
\underline{TN_{lake}}\\
r_{lake} = r_{in}^{0.54}
$$

#### "Reference" conditions
$$
\underline{TN_{lake}}\\
r_{lake} = r_{in}^{0.81}
$$

### Chla
#### Current conditions
$$
\underline{Chla_{lake}}\\
r_{lake} = r_{in}^{1.25}\\
\text{Assuming a fixed ratio between } TN_{lake}/TP_{lake}
$$

#### "Reference" conditions
$$
\underline{Chla_{lake}}\\
r_{lake} = r_{in}^{1.24}\\
\text{Assuming a fixed ratio between } TN_{lake}/TP_{lake}
$$

### Secchi
#### Current conditions
$$
\underline{Secchi_{lake}}\\
\text{when} \hspace{2mm} z_{max} \gt 20: \\
r_{lake} = r_{in}^{0.9}\\
\text{else when} \hspace{2mm} z_{max} \lt= 20: \\
r_{lake} = r_{in}^{0.38}
$$

#### "Reference" conditions
$$
\underline{Secchi_{lake}}\\
r_{lake} = r_{in}^{1.46}
$$










































