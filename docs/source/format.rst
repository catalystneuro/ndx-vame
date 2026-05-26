
.. _ndx-vame:

********
ndx-vame
********

Version |release| |today|

Data Structure Overview
=======================

This extension defines four main neurodata types:

1. **LatentSpaceSeries**: Extends ``TimeSeries`` to store VAME latent space vectors over time.

2. **MotifSeries**: Extends ``TimeSeries`` to store VAME motif IDs over time. Optionally links to the ``LatentSpaceSeries`` used to generate it.

3. **CommunitySeries**: Extends ``TimeSeries`` to store VAME community IDs over time. Optionally links to the ``MotifSeries`` used to generate it.

4. **VAMEProject**: A container for VAME data. Holds one optional ``LatentSpaceSeries``, zero or more ``MotifSeries``, and zero or more ``CommunitySeries``. Links to the original ``PoseEstimation`` data used as input.

Relationship Diagram
--------------------

.. code-block:: text

    VAMEProject
    ├── latent_space_series (LatentSpaceSeries, optional)
    ├── MotifSeriesHmm (MotifSeries)
    │   └── latent_space_series (link to LatentSpaceSeries, optional)
    ├── MotifSeriesKmeans (MotifSeries)
    │   └── latent_space_series (link to LatentSpaceSeries, optional)
    ├── CommunitySeriesHmm (CommunitySeries)
    │   └── motif_series (link to MotifSeries, optional)
    ├── CommunitySeriesKmeans (CommunitySeries)
    │   └── motif_series (link to MotifSeries, optional)
    └── pose_estimation (link to PoseEstimation, optional)

A ``VAMEProject`` can hold multiple ``MotifSeries`` and ``CommunitySeries`` (e.g. results from different algorithms). Each ``CommunitySeries`` optionally links back to the ``MotifSeries`` it was derived from.

Detailed Specification
======================

.. .. contents::

.. include:: _format_auto_docs/format_spec_main.inc
