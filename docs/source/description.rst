Overview
========

The ``ndx-vame`` extension enables storing and accessing data from VAME (Variational Animal Motion Embedding) in NWB (Neurodata Without Borders) files.

What is VAME?
-------------

VAME (Variational Animal Motion Embedding) is a framework for unsupervised behavior analysis that uses deep learning to discover and classify behavioral motifs from pose estimation data. It is maintained by the EthoML team and is available at `https://github.com/EthoML/VAME <https://github.com/EthoML/VAME>`_.

VAME analyzes pose tracking data (like that from DeepLabCut, SLEAP, etc.) and:

1. Identifies recurring patterns of movement (motifs)
2. Groups similar motifs into communities
3. Provides tools for analyzing behavioral sequences

This extension allows these VAME outputs to be stored in a standardized way within NWB files, alongside the original pose estimation data.

Installation
------------

The extension can be installed via pip:

.. code-block:: bash

    pip install ndx-vame

Dependencies
------------

- pynwb (>=2.8.0)
- hdmf (>=3.14.1)
- ndx-pose (>=0.2.1)

Usage
-----

This extension provides three main classes:

1. ``MotifSeries``: Stores the motif data from VAME
2. ``CommunitySeries``: Stores the community data from VAME
3. ``VAMEGroup``: A container for VAME data that links to the original pose estimation data

Basic example:

.. code-block:: python

    from ndx_vame import VAMEGroup, MotifSeries, CommunitySeries
    from ndx_pose import PoseEstimation

    # Create MotifSeries and CommunitySeries
    motif_series = MotifSeries(
        name="MotifSeries",
        data=motifs_data,  # numpy array of shape (n_samples, n_motifs)
        rate=sampling_rate,
    )

    community_series = CommunitySeries(
        name="CommunitySeries",
        data=communities_data,  # numpy array of shape (n_samples, n_communities)
        rate=sampling_rate,
        motif_series=motif_series,
    )

    # Create VAMEGroup and link to pose estimation data
    vame_group = VAMEGroup(
        name="VAMEGroup",
        motif_series=motif_series,
        community_series=community_series,
        vame_settings="dict containing VAME configuration",
        pose_estimation=pose_estimation,  # PoseEstimation object
    )

    # Add to a processing module in your NWB file
    behavior_pm = nwbfile.create_processing_module(
        name="behavior",
        description="processed behavioral data",
    )
    behavior_pm.add(vame_group)

For a complete example, see the `examples folder <https://github.com/catalystneuro/ndx-vame/tree/main/examples>`_.
