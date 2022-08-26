# NTIA/ITS SCOS Actions Plugin

![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/NTIA/scos-actions?display_name=tag&sort=semver)
![GitHub all releases](https://img.shields.io/github/downloads/NTIA/scos-actions/total)
![GitHub issues](https://img.shields.io/github/issues/NTIA/scos-actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains common actions and interfaces to be re-used by SCOS Sensor
plugins. See the [SCOS Sensor documentation](
https://github.com/NTIA/scos-sensor/blob/master/README.md)
for more information about SCOS Sensor, especially the [Architecture](
https://github.com/NTIA/scos-sensor/blob/master/README.md#architecture
) and the [Actions and Hardware Support](
https://github.com/NTIA/scos-sensor/blob/master/README.md#actions-and-hardware-support
) sections which explain how SCOS Actions is used in the SCOS plugin
architecture.

## Table of Contents

- [Overview of Repo Structure](#overview-of-repo-structure)
- [Running in SCOS Sensor](#running-in-scos-sensor)
- [Development](#development)
- [License](#license)
- [Contact](#contact)

## Overview of Repo Structure

- `scos_actions/actions`: This includes the base Action class, signals, and the following
  common action classes:
  - `acquire_single_freq_fft`: performs FFTs and calculates mean, median, min, max, and
    sample statistics at a single center frequency.
  - `acquire_single_freq_tdomain_iq`: acquires IQ data at a single center frequency.
  - `acquire_stepped_freq_tdomain_iq`: acquires IQ data at multiple center frequencies.
  - `calibrate_y_factor`: performs calibration using the Y-Factor method.
  - `sync_gps`: gets GPS location and syncs the host to GPS time
  - `monitor_sigan`: ensures a signal analyzer is available and is able to maintain a
    connection to the computer.
- `scos_actions/configs/actions`: This folder contains the YAML files with the parameters
  used to initialize the actions described above.
- `scos_actions/discover`: This includes the code to read YAML files and make actions
  available to scos-sensor.
- `scos_actions/hardware`: This includes the signal analyzer interface and GPS interface
  used by the actions and the mock signal analyzer. The signal analyzer interface is
  intended to represent universal functionality that is common across all signal
  analyzers. The specific implementations of the signal analyzer interface for
  particular signal analyzers are provided in separate repositories like
  [scos-usrp](https://github.com/NTIA/scos-usrp).
- `scos_actions/signal_processing`: This contains various common signal processing
routines which are used in actions.

## Running in SCOS Sensor

Refer to the [SCOS Sensor documentation](https://github.com/NTIA/scos-sensor#readme) for
detailed instructions. To run SCOS Actions in SCOS Sensor with a mock signal analyzer,
set `MOCK_SIGAN` and `MOCK_SIGAN_RANDOM` equal to 1 in `docker-compose.yml` before
starting SCOS Sensor:

```yaml
services:
  ...
  api:
    ...
    environment:
      ...
      - MOCK_SIGAN=1
      - MOCK_SIGAN_RANDOM=1
```

The following parameterized actions are offered for testing using a mock signal analyzer;
their parameters are defined in `scos_actions/configs/actions`.

- `test_multi_frequency_iq_action`
- `test_multi_frequency_y_factor_action`
- `test_single_frequency_iq_action`
- `test_single_frequency_m4s_action`
- `test_single_frequency_y_factor_action`

## Development

This repository is intended to be used by all SCOS Sensor plugins. Therefore, only
universal actions that apply to most RF measurement systems should be added to
SCOS Actions. Custom actions for specific hardware should be added to plugins in
repositories supporting that specific hardware. New functionality should only be
added to the [signal analyzer interface defined in this repository](scos_actions/hardware/sigan_iface.py)
if the new functionality can be supported by most signal analyzers.

### Requirements and Configuration

Set up a development environment using a tool like [Conda](https://docs.conda.io/en/latest/)
or [venv](https://docs.python.org/3/library/venv.html#module-venv), with `python>=3.8`. Then,
from the cloned directory, install the development dependencies by running:

```bash
pip install .[dev]
```

This will install the project itself, along with development dependencies for pre-commit
hooks, building distributions, and running tests. Set up pre-commit, which runs
auto-formatting and code-checking automatically when you make a commit, by running:

```bash
pre-commit install
```

The pre-commit tool will auto-format Python code using [Black](https://github.com/psf/black)
and [isort](https://github.com/pycqa/isort). Other pre-commit hooks are also enabled, and
can be found in [`.pre-commit-config.yaml`](.pre-commit-config.yaml).

### Building New Releases

This project uses [Hatchling](https://github.com/pypa/hatch/tree/master/backend) as a
backend. Hatchling makes versioning and building new releases easy. The package version can
be updated easily by using any of the following commands.

```bash
hatchling version major   # 1.0.0 -> 2.0.0
hatchling version minor   # 1.0.0 -> 1.1.0
hatchling version micro   # 1.0.0 -> 1.0.1
hatchling version "X.X.X" # 1.0.0 -> X.X.X
```

To build a new release (both wheel and sdist/tarball), run:

```bash
hatchling build
```

### Running Tests

Ideally, you should add a test to cover any new feature that you add. If you've done
that, then running the included test suite is the easiest way to check that everything
is working. In any case, all tests should be run after making any local modifications
to ensure that you haven't caused a regression.

The `scos_actions` package is tested using the [pytest](https://docs.pytest.org/en/stable/)
framework. Additionally, [tox](https://tox.readthedocs.io/en/latest/) is used to run all
available tests in a virtual environment against all supported versions of Python.
Running `pytest` directly is faster but running `tox` is a more thorough test.

The following commands can be used to run tests. Note, for tox to run with all Python
versions listed in the tox configuration (in [pyproject.toml](pyproject.toml)), all
those versions must be installed on your system. Any missing versions will be skipped.

```bash
pytest          # faster, but less thorough
tox             # tests code in clean virtual environments, with multiple versions of python
tox --recreate  # forces recreation of tox virtual environments
tox -e coverage # check where test coverage lacks
```

### Adding Actions

To expose a new action to the API, check out the available
[action classes](scos_actions/actions/__init__.py). An *action* is a parameterized
implementation of an action class. If an existing class covers your needs, you can
simply create YAML configs and use the `init` method in
[`scos_actions.discover`](scos_actions/discover/__init__.py) to make these actions available.

```python
from scos_actions.discover import init
from scos_usrp.hardware import gps, sigan

actions = {
  "monitor_usrp": MonitorSignalAnalyzer(sigan),
  "sync_gps": SyncGps(gps),
}

yaml_actions, yaml_test_actions = init(sigan=sigan, yaml_dir=ACTION_DEFINITIONS_DIR)

actions.update(yaml_actions)
```

Pass the implementation of the signal analyzer interface and the directory where the
YAML files are located to the `init` method.

If no existing action class meets your needs, see [Writing Custom Actions](
    #writing-custom-actions).

#### Creating a YAML config file for an action

Actions can be manually initialized in `discover/__init__.py`, but an easier method for
non-developers and configuration-management software is to place a YAML file in the
`configs/actions` directory which contains the action class name and parameter
definitions.

The file name can be anything. File extensions must be `.yml`.

The action initialization logic parses all YAML files in this directory and registers
the requested actions in the API.

Let's look at an example.

##### Example

Let's say we want to make an instance of the `SingleFrequencyFftAcquisition`.

First, create a new YAML file in the
`scos_actions/configs/actions` directory. In this example we're going to create
an acquisition for the LTE 700 C band downlink, so we'll call it `acquire_700c_dl.yml`.

Next, we want to find the appropriate string key for the
`SingleFrequencyFftAcquisition` class. Look in [actions/\_\_init\_\_.py](
    scos_actions/actions/__init__.py) at the `action_classes` dictionary. There, we
see:

```python
action_classes = {
    ...
    "single_frequency_fft": SingleFrequencyFftAcquisition,
    ...
}
```

That key tells the action loader which class to create an instance of. Put it as the
first non-comment line, followed by a colon:

```yaml
# File: acquire_700c_dl.yml

single_frequency_fft:
```

The next step is to see what parameters that class takes and specify the values. Open
up [actions/acquire_single_freq_fft.py](
    scos_actions/actions/acquire_single_freq_fft.py) and look at the documentation for
the class to see what parameters are available and what units to use, etc.

```python
class SingleFrequencyFftAcquisition(MeasurementAction):
    """Perform M4S detection over requested number of single-frequency FFTs.

    The action will set any matching attributes found in the signal
    analyzer object. The following parameters are required by the action:

        name: name of the action
        frequency: center frequency in Hz
        fft_size: number of points in FFT (some 2^n)
        nffts: number of consecutive FFTs to pass to detector

    For the parameters required by the signal analyzer, see the
    documentation from the Python package for the signal analyzer being
    used.

    :param parameters: The dictionary of parameters needed for the
        action and the signal analyzer.
    :param sigan: Instance of SignalAnalyzerInterface.
    """
```

Then look at the docstring for the signal analyzer class being used. This example will
use the [MockSignalAnalyzer](scos_actions/hardware/mocks/mock_sigan.py). That file
contains the following:

```python
class MockSignalAnalyzer(SignalAnalyzerInterface):
    """
    MockSignalAnalyzer is mock signal analyzer object for testing.

    The following parameters are required for measurements:
    sample_rate: requested sample rate in samples/second
    frequency: center frequency in Hz
    gain: requested gain in dB
    """
```

Lastly, simply modify the YAML file to define any required parameters from the action
and signal analyzer. Note that the `sigan` parameter is a special parameter that will get
passed in separately when the action is initialized from the YAML. Therefore, it does
not need to be defined in the YAML file.

```yaml
# File: acquire_700c_dl.yml

single_frequency_fft:
  name: acquire_700c_dl
  frequency: 751e6
  gain: 40
  sample_rate: 15.36e6
  fft_size: 1024
  nffts: 300
```

You're done.

#### Writing Custom Actions

"Actions" are one of the main concepts used by [SCOS Sensor](
https://github.com/NTIA/scos-sensor). At a high level, they are the things that the
sensor owner wants the sensor to be able to *do*. At a lower level, they are simply
Python classes with a special method `__call__`. Actions use [Django Signals](
https://docs.djangoproject.com/en/3.1/topics/signals/) to provide data and results to
scos-sensor.

Start by looking at the [`Action` base class](scos_actions/actions/interfaces/action.py).
It includes some logic to parse a description and summary out of the action class's
docstring, and a `__call__` method that must be overridden.

A new custom action can inherit from the existing action classes to reuse and build
upon existing functionality. A [`MeasurementAction` base class](scos_actions/actions/interfaces/measurement_action.py),
which inherits from the `Action` class, is also useful for building new actions.
For example, [`SingleFrequencyTimeDomainIqAcquisition`](scos_actions/actions/acquire_single_freq_tdomain_iq.py)
inherits from `MeasurementAction`, while [`SteppedFrequencyTimeDomainIqAcquisition`](scos_actions/actions/acquire_stepped_freq_tdomain_iq.py)
inherits from `SingleFrequencyTimeDomainIqAcquisition`.

Depending on the type of action, a signal should be sent upon action completion. This
enables SCOS Sensor to do something with the results of the action. This could range
from storing measurement data to recycling a Docker container or to fixing an unhealthy
connection to the signal analyzer. You can see the available signals in
[`scos_actions/actions/interfaces/signals.py`](scos_actions/actions/interfaces/signals.py).
The following signals are currently offered:

- `measurement_action_completed` - signal expects task_id, data, and metadata
- `location_action_completed` - signal expects latitude and longitude
- `monitor_action_completed` - signal expects boolean indicating if the signal analyzer
is healthy

New signals can be added. However, corresponding signal handlers must be added to
scos-sensor to receive the signals and process the results.

##### Adding custom action to SCOS Actions

A custom action meant to be re-used by other plugins can live in SCOS Actions. It can
be instantiated using a YAML file, or directly in the `actions` dictionary in the
`discover/__init__.py` module. This can be done in SCOS Actions with a mock signal
analyzer. Plugins supporting other hardware would need to import the action from
SCOS Actions. Then it can be instantiated in that plugin’s actions dictionary in its
discover module, or in a YAML file living in that plugin (as long as its discover
module includes the required code to parse the YAML files).

##### Adding system or hardware specific custom action

In the repository that provides the plugin to support the hardware being used, add the
action to the `actions` dictionary in the `discover/__init__.py` file. Optionally,
initialize the action using a YAML file by importing the YAML initialization code from
SCOS Actions. For an example of this, see the [Adding Actions subsection](#adding-actions)
above.

### Supporting a Different Signal Analyzer

[scos_usrp](https://github.com/NTIA/scos-usrp) adds support for the Ettus B2xx line of
signal analyzers to `scos-sensor`. Follow these instructions to add support for
another signal analyzer with a Python API.

- Create a new repository called `scos-[signal analyzer name]`.
- Create a new virtual environment and activate it:
  `python -m venv ./venv && source venv/bin/activate`.
  Upgrade pip: `python -m pip install --upgrade pip`.
- In the new repository, add this repository as a dependency and create a
  class that inherits from the [SignalAnalyzerInterface](scos_actions/hardware/sigan_iface.py)
  abstract class. Add properties or class variables for the parameters needed to
  configure the signal analyzer.
- Create YAML files with the parameters needed to run the actions imported from
  `scos_actions` using the new signal analyzer. Put them in the new repository in
  `configs/actions`. This should contain the parameters needed by the action as well as
  the signal analyzer settings based on which properties or class variables were
  implemented in the signal analyzer class in the previous step. The measurement actions
  in SCOS Actions are configured to check if any YAML parameters are available as
  attributes in the signal analyzer object, and to set them to the given YAML value if
  available. For example, if the new signal analyzer class has a bandwidth property,
  simply add a bandwidth parameter to the YAML file. Alternatively, you can create
  custom actions that are unique to the hardware. See [Adding Actions](#adding-actions)
  subsection above.
- In the new repository, add a `discover/__init__.py` file. This should contain a
  dictionary called `actions` with a key of action name and a value of action object.
  You can use the [init()](scos_actions/discover/__init__.py) and/or the
  [load_from_yaml()](scos_actions/discover/yaml.py) methods provided in this repository
  to look for YAML files and initialize actions. These methods allow you to pass your
  new signal analyzer object to the action's constructor. You can use the existing
  action classes [defined in this repository](scos_actions/actions/__init__.py) or
  [create custom actions](#writing-custom-actions). If the signal analyzer supports
  calibration, you should also add a `get_last_calibration_time()` method to
  `discover/__init__.py` to enable the status endpoint to report the last calibration
  time.

If your signal analyzer doesn't have a Python API, you'll need a Python wrapper that
calls out to your signal analyzer's available API and reads the samples back into
Python. Libraries such as [SWIG](http://www.swig.org/) can automatically generate
Python wrappers for programs written in C/C++.

The next step in supporting a different signal analyzer is to create a class that
inherits from the [GPSInterface](scos_actions/hardware/gps_iface.py) abstract class.
Then add the `sync_gps` and `monitor_sigan` actions to your `actions` dictionary,
passing the gps object to the `SyncGps` constructor, and the signal analyzer object to
the `MonitorSignalAnalyzer` constructor. See the example in the [Adding Actions
subsection](#adding-actions) above.

The final step would be to add a `setup.py` to allow for installation of the new
repository as a Python package. You can use the [setup.py](setup.py) in this repository
as a reference. You can find more information about Python packaging [here](
https://packaging.python.org/tutorials/packaging-projects/). Then add the new
repository as a dependency to [SCOS Sensor's requirements.txt](
https://github.com/NTIA/scos-sensor/blob/master/src/requirements.txt)
using the following format:
`<package_name> @ git+<link_to_github_repo>@<branch_name>`. If
specific drivers are required for your signal analyzer, you can attempt to link to them
within the package or create a docker image with the necessary files. You can host the
docker image as a [GitHub package](
https://docs.github.com/en/free-pro-team@latest/packages/using-github-packages-with-your-projects-ecosystem/configuring-docker-for-use-with-github-packages
). Then, when running scos-sensor, set the environment variable
`BASE_IMAGE=<image tag>`.

## License

See [LICENSE](LICENSE.md).

## Contact

For technical questions about SCOS Actions, contact Justin Haze, jhaze@ntia.gov
