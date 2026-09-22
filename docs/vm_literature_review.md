# Circular basis learning: targeted literature review

## Material Passport

- Origin: academic-research-suite, literature discovery and synthesis.
- Search date: 2026-09-12; saved at the owner's request on 2026-09-21.
- Research context: completed VM methodology v1 and proposed locality follow-up.
- Status: targeted source search, not a systematic review or exhaustive novelty audit.
- Coverage: primary papers, author-hosted copies and conference/publisher records;
  some sources were screened from abstracts and accessible methods rather than
  read exhaustively. No human-read attestation is implied.

## Research question

**When does adapting circular basis locations and widths improve prediction
under equal label and computation budgets?**

The completed [v1 report](vm_methodology_results.md) found higher primary error
for polar optimization than Cartesian optimization (62.94%) and for learned VM
locality than the locked fixed-VM ridge comparator (65.09%). These are results
of the specified synthetic protocol, not universal impossibility statements.
Both failed continuation gates remain part of the evidence.

Three identities constrain interpretation: the sum/log-output VM layer is a
sine/cosine linear map in polar coordinates; its temporal bank plus projection
collapses to direct sine/cosine maps; and peak-normalized VM kernels are Gaussian
RBFs on the unit-circle embedding. A new name or equivalent implementation does
not establish a new function class.

## Search scope and source selection

Queries covered von Mises basis-function neural networks, circular regression,
polar/Cartesian optimization, magnitude/direction parameterization, variable
projection for RBF networks, Fourier features and products of directional
manifolds. Seminal sources were retained regardless of age. Sources addressing
these mathematical mechanisms were preferred over incidental mentions of
von Mises, unrelated probability theory, or generic neural forecasting.
Discovery used web search; no private research notes or experiment arrays were
uploaded. Search-engine publication/crawl dates were not used as paper dates.

## Closest prior work

### 1. Von Mises basis-function networks already exist

Jenison, R. L., & Fissell, K. (1996). *A spherical basis function neural network
for modeling auditory space*. Neural Computation, 8(1), 115–128.
DOI: 10.1162/neco.1996.8.1.115.
[Author-hosted paper](https://wavelet.psych.wisc.edu/NN_JF.pdf).

This paper presents a von Mises basis-function network and gradient-descent
learning rules for spherical approximation. It is a close architectural
predecessor, although its domain is a sphere rather than this project's scalar
circle. It substantially narrows any claim that a VM basis neuron itself is new.
Its comparison with planar Gaussian constructions does not negate our exact
equivalence to a Gaussian on the appropriate unit-circle embedding.

Related earlier study: Jenison, R. L., & Fissell, K. (1995). *A comparison of the
von Mises and Gaussian basis functions for approximating spherical acoustic
scatter*. IEEE Transactions on Neural Networks, 6(5), 1284–1287.
DOI: 10.1109/72.410375.
[Publication record and abstract](https://pubmed.ncbi.nlm.nih.gov/18263419/).

### 2. Fixed circular dictionaries are an established alternative

Johnson, T. D., Coleman, T. P., & Rangel, L. M. (2019). *A flexible likelihood
approach for predicting neural spiking activity from oscillatory phase*.
Journal of Neuroscience Methods, 311, 307–317.
DOI: 10.1016/j.jneumeth.2018.10.028.
[Author-hosted paper](https://coleman.ucsd.edu/wp-content/uploads/2019/03/JCR-JNSM-Jan2019.pdf).

The method uses an overcomplete circular basis, including von Mises functions,
and sparse selection for phase-dependent spiking. This supports treating a
fixed multiscale circular dictionary as a serious control. Its likelihood,
biological task and selection procedure differ from our squared-error study.

### 3. Variable projection separates geometry from the readout

Zheng, S., & Feng, R. (2023). *A variable projection method for the general radial
basis function neural network*. Applied Mathematics and Computation, 451, 128009.
DOI: 10.1016/j.amc.2023.128009.
[Publisher record](https://www.sciencedirect.com/science/article/pii/S0096300323001789).

The method eliminates linear output weights from the nonlinear optimization
problem. This is directly relevant to a circular bank whose centers and widths
are nonlinear parameters and whose readout is linear. The proposed application
here is a diagnostic: solve a regularized readout accurately at every geometry
step, then test whether adaptation helps. It is not a claim to invent variable
projection or a guarantee that it will improve our held-out errors. Access in
the search was limited to the publisher's indexed abstract/method descriptions.

## Optimization and frequency controls

Salimans, T., & Kingma, D. P. (2016). *Weight normalization: A simple
reparameterization to accelerate training of deep neural networks*. NeurIPS 29.
[Conference paper](https://proceedings.neurips.cc/paper/2016/hash/ed265bc903a5a097f61d3ec064d96d2e-Abstract.html).

Separating weight magnitude and direction can change optimization without
changing representational capacity. This is relevant to Stage A, but weight
normalization is not identical to a softplus-amplitude circular parameterization.

Wu, X., Dobriban, E., Ren, T., Wu, S., Li, Z., Gunasekar, S., Ward, R., & Liu, Q.
(2020). *Implicit regularization and convergence for weight normalization*.
NeurIPS 33. [Author preprint, later revised](https://arxiv.org/abs/1911.07956).

The analysis studies magnitude/direction reparameterization in overparameterized
least squares and its implicit regularization. It motivates explaining the
weak-signal Adam finding through optimization/regularization rather than calling
it evidence of a generally superior neuron. Its assumptions do not directly
prove a mechanism for our three-parameter experiment.

Tancik, M., et al. (2020). *Fourier features let networks learn high frequency
functions in low dimensional domains*. NeurIPS 33.
[Paper](https://arxiv.org/abs/2006.10739).

Fourier mappings alter frequency-dependent learning behavior. Our existing
Fourier ridge is a relevant control; a multi-harmonic Fourier-feature MLP would
test a different question from the existing first-harmonic MLP. Integer angular
harmonics are needed if exact 2*pi periodicity is required. Arbitrary learned or
random scalar frequencies do not automatically preserve that guarantee.

## Related directions beyond the current scalar-input study

Cao, Z., & Li, K. (2024). *Gaussian process on the product of directional
manifolds*. European Control Conference, 3509–3514.
[Conference paper](https://www.paperhost.org/proceedings/controls/ECC24/files/0037.pdf).

The paper constructs von Mises-based kernels for products of circular domains,
including correlated components. It is relevant prior work for multi-angle
interactions. It does not establish that the particular proposed relative-angle
neural layer is necessary or better than Cartesian product features.

Worrall, D. E., Garbin, S. J., Turmukhambetov, D., & Brostow, G. J. (2017).
*Harmonic networks: Deep translation and rotation equivariance*. CVPR, 5028–5037.
[Conference paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Worrall_Harmonic_Networks_Deep_CVPR_2017_paper.html).

Circular harmonics enforce structured responses to image rotation. This is
useful conceptual background for symmetry tests; image-rotation equivariance
is not the same property as scalar-input 2*pi periodicity.

Bishop, C. M., & Nabney, I. T. (1996). *Modeling conditional probability
distributions for periodic variables*. Neural Computation, 8(5), 1123–1133.
[Institutional paper](https://publications.aston.ac.uk/297/1/neco.1996.8.5.1123.pdf).

Neural outputs parameterize circular conditional densities, including adaptive
von Mises mixtures. This establishes substantial prior work for probabilistic
VM outputs. Density normalization, likelihood stability and calibration would
be essential in such a future study; our signed regression readout is not a
probability density.

Cohen, Y., Navarro, A. K. W., Frellsen, J., Turner, R. E., Riemer, R., & Pakman, A.
(2025). *Bayesian circular regression with von Mises quasi-processes*.
AISTATS, PMLR 258, 1693–1701.
[Conference paper and software](https://proceedings.mlr.press/v258/cohen25a.html).

This is recent related work on uncertain circle-valued outputs, evaluated on
wind direction and gait-cycle prediction. Its output domain and Bayesian
inference differ from learning a real-valued function of a circular input.
It would be a relevant comparator for a separately scoped uncertainty project.

## Synthesis and proposed follow-up

The evidence places this project within established circular RBF, Fourier and
reparameterization research. The useful open question is conditional performance
under controlled resources, not whether the VM kernel family is new.

The v1 comparison matched trainable parameter counts but used 16 learned kernels
versus 48 fixed features and joint Adam versus direct readout fitting. It also
used 256 validation labels at every training size. Thus it does not isolate the
causal value of adapting centers/widths or compare total-label efficiency.

The recommended next study uses identical initial 16-kernel banks with fixed,
center-only, width-only and jointly adapted geometry; regularized exact readouts;
an explicit joint-Adam control; equal total training/validation label budgets;
and measured computation allowances. Equal-parameter fixed controls should be
reported alongside equal-basis controls. Conditioning and prediction sensitivity
remain diagnostics, especially given v1's large unregularized coefficients.

Any follow-up uses a new protocol, fresh random streams and a new confirmation
lock. V1's failed gates are retained. Stage A geometry analysis, relative-angle
interactions, probabilistic outputs and larger architectures are separate
questions, not automatic scope extensions. No paper in this review establishes
that the proposed follow-up will succeed, and this search does not establish
that its exact experimental design is novel.
