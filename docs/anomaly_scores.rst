.. _anomaly_scores:

Anomaly scores
==============

The reconstruction errors of an autoencoder are per-sample *and* per-feature: for every input sample there is a
vector of residuals (prediction minus actual value). An :class:`~energy_fault_detector.core.anomaly_score.AnomalyScore`
reduces that vector to a **single scalar score per sample**. Samples with high scores deviate strongly from the learned
normal behaviour and are therefore candidate anomalies.

The chosen score is configured under ``train.anomaly_score`` in the YAML config. This page explains the available
scores, how they differ and when to use which, and recommends:
:class:`RMSEScore <energy_fault_detector.anomaly_scores.rmse_score.RMSEScore>` as the default.

.. note::
   An anomaly score only ranks samples; it does **not** classify them. The actual *anomaly decision*
   (normal vs. anomalous) is taken afterwards by a :class:`~energy_fault_detector.core.threshold_selector.ThresholdSelector`.

Available scores
----------------

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Class
     - Typical config names (``train.anomaly_score.name``)
     - Description
   * - :class:`RMSEScore <energy_fault_detector.anomaly_scores.rmse_score.RMSEScore>`
     - ``"rmse"``, ``"RMSE"``
     - Root-mean-square reconstruction error over all features. **Recommended default.**
   * - :class:`WeightedRMSEScore <energy_fault_detector.anomaly_scores.weighted_rmse_score.WeightedRMSEScore>`
     - ``"weighted_rmse"``, ``"WeightedRMSE"``
     - RMSE where selected features can be up- or down-weighted.
   * - :class:`MahalanobisScore <energy_fault_detector.anomaly_scores.mahalanobis_score.MahalanobisScore>`
     - ``"mahalanobis"``, ``"Mahalanobis"``
     - Correlation-aware Mahalanobis distance.

RMSE (recommended default)
--------------------------

The RMSE score standardises the reconstruction errors (using the mean and standard deviation learned on the training
data) and then computes the per-sample root mean square across all features:

.. math::

   s_i = \sqrt{\frac{1}{p} \sum_{j=1}^{p} z_{ij}^{\,2}},
   \qquad z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}

where :math:`x_{ij}` is the reconstruction error of sample :math:`i` on feature :math:`j`, and :math:`\mu_j`,
:math:`\sigma_j` the corresponding training mean and standard deviation.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Advantages
     - Limitations
   * - Simple, fast and numerically robust.
     - Treats all features as independent (does not account for correlation).
   * - Does not require estimating a covariance and is stable for **any** number of samples.
     - A feature shown to be more fault-relevant is not emphasised automatically.

.. code-block:: yaml

   train:
     anomaly_score:
       name: rmse           # RMSE is the recommended default
       params: {}

Use RMSE as a robust, well-understood default that works well out of the box for most use cases.

Weighted RMSE
-------------

:class:`WeightedRMSEScore <energy_fault_detector.anomaly_scores.weighted_rmse_score.WeightedRMSEScore>` extends RMSE by
allowing **per-feature weights**, so that deviations on sensor/feature values you care most about contribute more to the
final score. The reconstruction errors are first standardised (to remove model bias towards high-magnitude features) and
then multiplied by the feature weight before taking the per-sample root mean square.

Choosing a weight:

- ``weight > 1`` increases the influence of a feature (deviations on it are treated as more suspicious),
- ``0 <= weight < 1`` decreases its influence,
- a weight of ``0`` ignores the feature,
- features are *not* required to have an entry; omitted features keep weight ``1``.

.. code-block:: yaml

   train:
     anomaly_score:
       name: weighted_rmse
       params:
         feature_weights:
           rotor_speed: 2.0     # emphasise this sensor
           temperature_11: 0.5  # de-emphasise this sensor
           # any other feature keeps weight 1.0

.. note::
   ``WeightedRMSEScore`` requires the input to be a :class:`~pandas.DataFrame` so that the weights can be matched to
   the feature names. Negative weights are not allowed.

Use weighted RMSE when you have **domain knowledge** about which features are most relevant for the faults of interest
and want that prior to influence the anomaly score directly.

Mahalanobis
-----------

:class:`MahalanobisScore <energy_fault_detector.anomaly_scores.mahalanobis_score.MahalanobisScore>` computes the
**Mahalanobis distance**, which unlike RMSE accounts for the *correlation* between features. Instead of summing
independent standardised errors, it measures the distance in units of the covariance of the (mean-centred) errors:

.. math::

   s_i = \sqrt{(x_i - \mu)^{\top} \Sigma^{-1} (x_i - \mu)}

The covariance estimator is chosen via the ``covariance_method`` parameter:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - ``covariance_method``
     - Behaviour
   * - ``"auto"`` (default)
     - Use MinCovDet when it is stable, otherwise fall back to a shrinkage estimator (see below).
   * - ``"min_cov_det"``
     - Always use Minimum Covariance Determinant (robust, but requires enough samples).
   * - ``"shrinkage"``
     - Always use a shrinkage estimator (Ledoit-Wolf / OAS). Robust for high-dimensional data.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Advantages
     - Limitations
   * - Accounts for correlation between features; can separate faults that RMSE cannot.
     - More computationally expensive and more complex than RMSE.
   * - Robust (MinCovDet) or high-dimensional-friendly (shrinkage) variants available.
     - Sensitive to covariance estimation quality.

Use the Mahalanobis score when features are strongly correlated and you need a **correlation-aware** score, but reserve
it for cases where the added complexity is justified.

The derived stability threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MinCovDet needs enough samples per feature: as the sample-to-feature ratio ``n / p`` drops it quickly becomes unstable
(its covariance becomes rank-deficient and ill-conditioned). Concretely:

- for ``n / p >= min_cov_det_ratio`` (default ``2.0``) the covariance is well conditioned and MinCovDet is usable;
- below that, or when the covariance is singular / ill-conditioned, a **shrinkage** estimator (Ledoit-Wolf / OAS)
  should be used instead.

With ``covariance_method="auto"`` this threshold is applied automatically and the estimator falls back to shrinkage
(Ledoit-Wolf / OAS) whenever MinCovDet would be unstable.

.. code-block:: yaml

   train:
     anomaly_score:
       name: mahalanobis
       params:
         covariance_method: auto     # "auto" | "min_cov_det" | "shrinkage"
         min_cov_det_ratio: 2.0      # minimum n/p for MinCovDet when "auto"
         shrinkage_method: oas       # "oas" | "ledoit_wolf"

.. deprecated::
   The legacy ``pca=True`` path of :class:`MahalanobisScore` is deprecated and will be replaced by the clean Mahalanobis
   norm implementation (``pca=False``). Likewise the ``scale`` parameter is deprecated and no longer has any effect.
   See :doc:`advanced_usage` for details.

Which score should I use?
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Your situation
     - Recommended score
   * - You want a simple, robust default that works out of the box.
     - **RMSE** (``name: rmse``)
   * - You have domain knowledge about which features matter most for the fault.
     - Weighted RMSE (``name: weighted_rmse``)
   * - Features are strongly correlated and a correlation-aware score is important.
     - Mahalanobis (``name: mahalanobis``, ``covariance_method: auto``)
   * - High-dimensional input where a stable covariance is hard to estimate.
     - Mahalanobis with ``covariance_method: shrinkage`` (or RMSE)

For most projects the RMSE score is a safe, interpretable and efficient default: it needs no covariance estimation,
is stable regardless of the number of samples and features, and pairs well with a good threshold selector. Start with
RMSE and move to a more specialised score only when a concrete need arises.
