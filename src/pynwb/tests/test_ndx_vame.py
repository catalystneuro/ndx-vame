import datetime
import numpy as np
import pytest
import tempfile
import os

from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from ndx_pose import PoseEstimationSeries, PoseEstimation
from ndx_vame import VAMEGroup, MotifSeries, CommunitySeries


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


class TestVAME:
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
        motif_data = np.random.rand(100, 15)
        motif_series = MotifSeries(
            name="MotifSeries",
            data=motif_data,
            rate=10.0,
        )

        community_data = np.random.rand(100, 3)
        community_series = CommunitySeries(
            name="CommunitySeries",
            data=community_data,
            rate=10.0,
            motif_series=motif_series,
        )

        vame_group = VAMEGroup(
            name="VAMEGroup",
            motif_series=motif_series,
            community_series=community_series,
            vame_settings="dict containing config",
            pose_estimation=pose_estimation,
        )

        behavior_pm.add(vame_group)

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
                assert "VAMEGroup" in nwbfile.processing["behavior"].data_interfaces

                read_vame_group = nwbfile.processing["behavior"].data_interfaces["VAMEGroup"]
                assert read_vame_group.name == "VAMEGroup"

                read_motif_series = read_vame_group.motif_series
                assert read_motif_series.name == "MotifSeries"
                assert np.array_equal(read_motif_series.data[:], motif_data)
                assert read_motif_series.rate == 10.0

                read_community_series = read_vame_group.community_series
                assert read_community_series.name == "CommunitySeries"
                assert np.array_equal(read_community_series.data[:], community_data)
                assert read_community_series.rate == 10.0

                assert read_vame_group.vame_settings == "dict containing config"
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)
