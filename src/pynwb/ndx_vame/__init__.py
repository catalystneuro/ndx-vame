from importlib.resources import files
import os
from pynwb import load_namespaces, get_class, TimeSeries
from hdmf.utils import get_docval, docval

# NOTE: ndx-pose needs to be imported first because loading the ndx-vame namespace depends on
# having the ndx-pose namespace loaded into the global type map.
from ndx_pose import PoseEstimation


# Get path to the namespace.yaml file with the expected location when installed not in editable mode
__location_of_this_file = files(__name__)
__spec_path = __location_of_this_file / "spec" / "ndx-vame.namespace.yaml"

# If that path does not exist, we are likely running in editable mode. Use the local path instead
if not os.path.exists(__spec_path):
    __spec_path = __location_of_this_file.parent.parent.parent / "spec" / "ndx-vame.namespace.yaml"

# Load the namespace
load_namespaces(str(__spec_path))

# Define your classes here to make them accessible at the package level.
# Either have PyNWB generate a class from the spec using `get_class` as shown
# below or write a custom class and register it using the class decorator
AutoLatentSpaceSeries = get_class("LatentSpaceSeries", "ndx-vame")
AutoMotifSeries = get_class("MotifSeries", "ndx-vame")
AutoCommunitySeries = get_class("CommunitySeries", "ndx-vame")
VAMEProject = get_class("VAMEProject", "ndx-vame")


ls_docval = list(get_docval(AutoLatentSpaceSeries.__init__))
for arg in ls_docval.copy():
    if arg["name"] == "unit":
        ls_docval.remove(arg)
        break
class LatentSpaceSeries(AutoLatentSpaceSeries):
    @docval(*ls_docval)
    def __init__(self, **kwargs):
        # Set the unit to "n/a" in the constructor, as specified in the parent's spec
        kwargs["unit"] = "n/a"
        super().__init__(**kwargs)


ms_docval = list(get_docval(AutoMotifSeries.__init__))
for arg in ms_docval.copy():
    if arg["name"] == "unit":
        ms_docval.remove(arg)
        break
class MotifSeries(AutoMotifSeries):
    @docval(*ms_docval)
    def __init__(self, **kwargs):
        # Set the unit to "n/a" in the constructor, as specified in the parent's spec
        kwargs["unit"] = "n/a"
        super().__init__(**kwargs)


cs_docval = list(get_docval(AutoCommunitySeries.__init__))
for arg in cs_docval.copy():
    if arg["name"] == "unit":
        cs_docval.remove(arg)
        break
class CommunitySeries(AutoCommunitySeries):
    @docval(*cs_docval)
    def __init__(self, **kwargs):
        # Set the unit to "n/a" in the constructor, as specified in the parent's spec
        kwargs["unit"] = "n/a"
        super().__init__(**kwargs)


# Add all classes to __all__ to make them accessible at the package level
__all__ = [
    "LatentSpaceSeries",
    "MotifSeries",
    "CommunitySeries",
    "VAMEProject",
]

# Remove these functions/modules from the package
del load_namespaces, get_class, files, os, __location_of_this_file, __spec_path
