import json
import datetime
import numpy as np
import pytest
import tempfile
import os

from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from ndx_pose import PoseEstimationSeries, PoseEstimation
from ndx_ethogram import EthogramBouts
from ndx_vame import VAMEProject, MotifSeries, CommunitySeries, LatentSpaceSeries


@pytest.fixture
def nwbfile():
    """Create a basic NWBFile for testing."""
    return NWBFile(
        session_description="Test session",
        identifier="TEST123",
        session_start_time=datetime.datetime.now(datetime.timezone.utc),
    )


@pytest.fixture
def pose_estimation(nwbfile):
    """Create a PoseEstimation object for testing."""
    # Create a device for the camera
    camera = nwbfile.create_device(
        name="camera",
        description="camera for recording behavior",
        manufacturer="manufacturer",
    )

    # Create PoseEstimationSeries for different body parts
    reference_frame = "(0,0,0) corresponds to the top-left corner of the video"

    # Front left paw
    data_flp = np.random.rand(100, 2)  # num_frames x (x, y)
    confidence_flp = np.random.rand(100)  # confidence value for every frame
    front_left_paw = PoseEstimationSeries(
        name="front_left_paw",
        description="Marker placed around fingers of front left paw.",
        data=data_flp,
        unit="pixels",
        rate=10.0,
        reference_frame=reference_frame,
        confidence=confidence_flp,
    )

    # Body
    data_body = np.random.rand(100, 2)
    confidence_body = np.random.rand(100)
    body = PoseEstimationSeries(
        name="body",
        description="Marker placed on center of body.",
        data=data_body,
        unit="pixels",
        rate=10.0,
        reference_frame=reference_frame,
        confidence=confidence_body,
    )

    # Front right paw
    data_frp = np.random.rand(100, 2)
    confidence_frp = np.random.rand(100)
    front_right_paw = PoseEstimationSeries(
        name="front_right_paw",
        description="Marker placed around fingers of front right paw.",
        data=data_frp,
        unit="pixels",
        rate=10.0,
        reference_frame=reference_frame,
        confidence=confidence_frp,
    )

    # Create PoseEstimation object
    pose_estimation = PoseEstimation(
        name="PoseEstimation",
        pose_estimation_series=[front_left_paw, body, front_right_paw],
        description="Estimated positions of front paws using DeepLabCut.",
        original_videos=["path/to/camera.mp4"],
        dimensions=np.array([[640, 480]], dtype="uint16"),  # pixel dimensions of the video
        devices=[camera],
        scorer="DLC_resnet50_openfieldOct30shuffle1_1600",
        source_software="DeepLabCut",
        source_software_version="2.3.8",
    )

    return pose_estimation


class TestVAMESingleAlgorithm:
    """Test the VAME extension classes."""

    def test_roundtrip(self, nwbfile, pose_estimation):
        """Test writing and reading VAME data to and from an NWB file."""
        # Add subject to the NWB file
        subject = Subject(subject_id="subject1", species="Mus musculus")
        nwbfile.subject = subject

        # Create a behavior processing module
        behavior_pm = nwbfile.create_processing_module(
            name="behavior",
            description="processed behavioral data",
        )
        behavior_pm.add(pose_estimation)

        # Create VAME data
        # Latent space series (n_samples, n_dims)
        n_samples = 100
        rate = 10.0
        latent_space_data = np.random.rand(n_samples, 30)
        latent_space_series = LatentSpaceSeries(
            name="LatentSpaceSeries",
            data=latent_space_data,
            rate=rate,
        )

        # Motif series (n_samples,)
        motif_data = np.random.randint(0, 15, size=n_samples)
        motif_series = MotifSeries(
            name="MotifSeries",
            data=motif_data,
            rate=rate,
            algorithm="hmm",
            latent_space_series=latent_space_series,
        )

        # Community series (n_samples,)
        community_data = np.random.randint(0, 3, size=n_samples)
        community_series = CommunitySeries(
            name="CommunitySeries",
            data=community_data,
            rate=rate,
            algorithm="hierarchical_clustering",
            motif_series=motif_series,
        )

        # Create VAMEProject with mock config
        config = {
            "vame_version": "0.11.0",
            "project_name": "my_vame_project",
            "creation_datetime": "2025-04-30T15:48:58+00:00",
        }
        vame_config = json.dumps(config)
        vame_project = VAMEProject(
            name="VAMEProject",
            latent_space_series=latent_space_series,
            motif_series=[motif_series],
            community_series=[community_series],
            vame_config=vame_config,
            time_window_samples=30,
            vame_version="0.11.0",
            pose_estimation=pose_estimation,
        )

        behavior_pm.add(vame_project)

        temp_file = os.path.join(tempfile.gettempdir(), "test_vame.nwb")
        try:
            # Write the NWB file
            with NWBHDF5IO(temp_file, "w") as io:
                io.write(nwbfile)

            # Read the NWB file
            with NWBHDF5IO(temp_file, "r") as io:
                nwbfile = io.read()

                # Verify the data was read correctly
                assert "behavior" in nwbfile.processing
                assert "VAMEProject" in nwbfile.processing["behavior"].data_interfaces

                read_vame_project = nwbfile.processing["behavior"].data_interfaces["VAMEProject"]
                assert read_vame_project.name == "VAMEProject"

                read_motif_series = read_vame_project.motif_series["MotifSeries"]
                assert read_motif_series.name == "MotifSeries"
                assert np.array_equal(read_motif_series.data[:], motif_data)
                assert read_motif_series.rate == 10.0

                read_community_series = read_vame_project.community_series["CommunitySeries"]
                assert read_community_series.name == "CommunitySeries"
                assert np.array_equal(read_community_series.data[:], community_data)
                assert read_community_series.rate == 10.0

                assert read_vame_project.vame_config == vame_config
                assert read_vame_project.time_window_samples == 30
                assert read_vame_project.vame_version == "0.11.0"
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    class TestVAMEMultipleAlgorithms:
        """Test that a VAMEProject can hold multiple MotifSeries and CommunitySeries."""

        def test_roundtrip(self, nwbfile, pose_estimation):
            behavior_pm = nwbfile.create_processing_module(
                name="behavior",
                description="processed behavioral data",
            )
            behavior_pm.add(pose_estimation)

            n_samples = 50
            rate = 10.0
            latent_space_series = LatentSpaceSeries(
                name="LatentSpaceSeries",
                data=np.random.rand(n_samples, 10),
                rate=rate,
            )

            motif_data_hmm = np.random.randint(0, 10, size=n_samples)
            motif_series_hmm = MotifSeries(
                name="MotifSeriesHmm",
                data=motif_data_hmm,
                rate=rate,
                algorithm="hmm",
            )

            motif_data_kmeans = np.random.randint(0, 10, size=n_samples)
            motif_series_kmeans = MotifSeries(
                name="MotifSeriesKmeans",
                data=motif_data_kmeans,
                rate=rate,
                algorithm="kmeans",
            )

            community_data = np.random.randint(0, 3, size=n_samples)
            community_series_hmm = CommunitySeries(
            name="CommunitySeriesHmm",
            data=community_data,
            rate=rate,
            algorithm="hierarchical_clustering",
            motif_series=motif_series_hmm,
            )

            community_data_kmeans = np.random.randint(0, 3, size=n_samples)
            community_series_kmeans = CommunitySeries(
                name="CommunitySeriesKmeans",
                data=community_data_kmeans,
                rate=rate,
                algorithm="graph_clustering",
                motif_series=motif_series_kmeans,
            )

            vame_config = json.dumps({"vame_version": "0.11.0", "project_name": "multi_series_project"})
            vame_project = VAMEProject(
                name="VAMEProject",
                latent_space_series=latent_space_series,
                motif_series=[motif_series_hmm, motif_series_kmeans],
                community_series=[community_series_hmm, community_series_kmeans],
                vame_config=vame_config,
                time_window_samples=30,
                vame_version="0.11.0",
            )

            behavior_pm.add(vame_project)

            # Verify in-memory before roundtrip
            assert len(vame_project.motif_series) == 2
            assert "MotifSeriesHmm" in vame_project.motif_series
            assert "MotifSeriesKmeans" in vame_project.motif_series
            assert len(vame_project.community_series) == 2
            assert "CommunitySeriesHmm" in vame_project.community_series
            assert "CommunitySeriesKmeans" in vame_project.community_series

            temp_file = os.path.join(tempfile.gettempdir(), "test_vame_multi.nwb")
            try:
                with NWBHDF5IO(temp_file, "w") as io:
                    io.write(nwbfile)

                with NWBHDF5IO(temp_file, "r") as io:
                    read_nwbfile = io.read()
                    read_vame_project = read_nwbfile.processing["behavior"].data_interfaces["VAMEProject"]

                    assert len(read_vame_project.motif_series) == 2
                    read_hmm = read_vame_project.motif_series["MotifSeriesHmm"]
                    assert np.array_equal(read_hmm.data[:], motif_data_hmm)
                    assert read_hmm.algorithm == "hmm"

                    read_kmeans = read_vame_project.motif_series["MotifSeriesKmeans"]
                    assert np.array_equal(read_kmeans.data[:], motif_data_kmeans)
                    assert read_kmeans.algorithm == "kmeans"

                    assert len(read_vame_project.community_series) == 2
                    read_hmm = read_vame_project.community_series["CommunitySeriesHmm"]
                    assert np.array_equal(read_hmm.data[:], community_data)
                    assert read_hmm.algorithm == "hierarchical_clustering"

                    read_kmeans = read_vame_project.community_series["CommunitySeriesKmeans"]
                    assert np.array_equal(read_kmeans.data[:], community_data_kmeans)
                    assert read_kmeans.algorithm == "graph_clustering"
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)


def _make_ethogram_bouts(
    name: str, labels: np.ndarray, rate: float, starting_time: float = 0.0
) -> EthogramBouts:
    """Run-length encode a label array into an EthogramBouts table."""
    labels = np.asarray(labels)
    boundaries = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], boundaries])
    stops = np.concatenate([boundaries, [len(labels)]])

    bouts = EthogramBouts(name=name, labeling_method="automated", source_software="VAME")
    for start, stop in zip(starts, stops):
        bouts.add_row(
            start_time=starting_time + start / rate,
            stop_time=starting_time + stop / rate,
            label=str(labels[start]),
        )
    return bouts


class TestEthogramLinks:
    """Test optional EthogramBouts links on MotifSeries and CommunitySeries."""

    def test_motif_series_ethogram_bouts_roundtrip(self, nwbfile):
        """MotifSeries.ethogram_bouts link survives HDF5 roundtrip; bout times honour starting_time."""
        behavior_pm = nwbfile.create_processing_module("behavior", "processed behavioral data")

        time_window_samples = 30
        rate = 10.0
        starting_time = (time_window_samples // 2) / rate  # 1.5 s
        motif_data = np.array([0] * 10 + [1] * 15 + [2] * 10 + [0] * 15, dtype=np.int32)

        ethogram_bouts = _make_ethogram_bouts(
            name="MotifBouts",
            labels=motif_data,
            rate=rate,
            starting_time=starting_time,
        )
        behavior_pm.add(ethogram_bouts)

        motif_series = MotifSeries(
            name="MotifSeries",
            data=motif_data,
            rate=rate,
            starting_time=starting_time,
            ethogram_bouts=ethogram_bouts,
        )
        vame_project = VAMEProject(
            name="VAMEProject",
            motif_series=[motif_series],
            vame_config=json.dumps({}),
            time_window_samples=time_window_samples,
            vame_version="0.11.0",
        )
        behavior_pm.add(vame_project)

        temp_file = os.path.join(tempfile.gettempdir(), "test_motif_ethogram.nwb")
        try:
            with NWBHDF5IO(temp_file, "w") as io:
                io.write(nwbfile)

            with NWBHDF5IO(temp_file, "r") as io:
                read_nwb = io.read()
                read_ms = read_nwb.processing["behavior"]["VAMEProject"].motif_series["MotifSeries"]
                assert read_ms.ethogram_bouts is not None
                assert read_ms.ethogram_bouts.name == "MotifBouts"
                assert len(read_ms.ethogram_bouts) == 4  # 4 contiguous runs
                assert read_ms.starting_time == starting_time
                assert read_ms.ethogram_bouts["start_time"][0] == starting_time
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_community_series_ethogram_bouts_roundtrip(self, nwbfile):
        """CommunitySeries.ethogram_bouts link survives HDF5 roundtrip; bout times honour starting_time."""
        behavior_pm = nwbfile.create_processing_module("behavior", "processed behavioral data")

        time_window_samples = 30
        rate = 10.0
        starting_time = (time_window_samples // 2) / rate  # 1.5 s
        motif_data = np.array([0] * 20 + [1] * 20, dtype=np.int32)
        community_data = np.array([0] * 20 + [1] * 20, dtype=np.int32)

        motif_series = MotifSeries(name="MotifSeries", data=motif_data, rate=rate, starting_time=starting_time)
        community_bouts = _make_ethogram_bouts("CommunityBouts", community_data, rate, starting_time)
        behavior_pm.add(community_bouts)

        community_series = CommunitySeries(
            name="CommunitySeries",
            data=community_data,
            rate=rate,
            starting_time=starting_time,
            motif_series=motif_series,
            ethogram_bouts=community_bouts,
        )
        vame_project = VAMEProject(
            name="VAMEProject",
            motif_series=[motif_series],
            community_series=[community_series],
            vame_config=json.dumps({}),
            time_window_samples=time_window_samples,
            vame_version="0.11.0",
        )
        behavior_pm.add(vame_project)

        temp_file = os.path.join(tempfile.gettempdir(), "test_community_ethogram.nwb")
        try:
            with NWBHDF5IO(temp_file, "w") as io:
                io.write(nwbfile)

            with NWBHDF5IO(temp_file, "r") as io:
                read_nwb = io.read()
                read_cs = read_nwb.processing["behavior"]["VAMEProject"].community_series["CommunitySeries"]
                assert read_cs.ethogram_bouts is not None
                assert read_cs.ethogram_bouts.name == "CommunityBouts"
                assert len(read_cs.ethogram_bouts) == 2  # 2 contiguous runs
                assert read_cs.starting_time == starting_time
                assert read_cs.ethogram_bouts["start_time"][0] == starting_time
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
