
.. _ndx-vame:

********
ndx-vame
********

Version |release| |today|

Data Structure Overview
======================

This extension defines three main neurodata types:

1. **MotifSeries**: Extends ``TimeSeries`` to store VAME motif data. Each column in the data represents a different motif, and each row represents a time point.

2. **CommunitySeries**: Extends ``TimeSeries`` to store VAME community data. Communities are groups of similar motifs. Each column in the data represents a different community, and each row represents a time point.

3. **VAMEGroup**: A container for VAME data that includes both MotifSeries and CommunitySeries, and links to the original PoseEstimation data used to generate the VAME analysis.

Relationship Diagram
-------------------

.. code-block:: text

    VAMEGroup
    ├── motif_series (MotifSeries)
    ├── community_series (CommunitySeries)
    │   └── motif_series (link to MotifSeries)
    └── pose_estimation (link to PoseEstimation)

The VAMEGroup contains both the MotifSeries and CommunitySeries, and links to the PoseEstimation data that was used as input to VAME. The CommunitySeries also links to the MotifSeries to establish the relationship between motifs and communities.

Detailed Specification
=====================

.. .. contents::

.. include:: _format_auto_docs/format_spec_main.inc
