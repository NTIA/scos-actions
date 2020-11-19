# 1. Title: NTIA/ITS SCOS Actions Plugin

This repository contains common actions and interfaces to be re-used by scos-sensor
plugins. See the [scos-sensor documentation](
    https://github.com/NTIA/scos-sensor/blob/SMBWTB475_refactor_radio_interface/README.md)
for more information about scos-sensor, especially the section about
[Actions and Hardware Support](
    https://github.com/NTIA/scos-sensor/blob/SMBWTB475_refactor_radio_interface/DEVELOPING.md#actions-and-hardware-support).

## 2. Table of Contents

- [Overview of Repo Structure](#3-overview-of-repo-structure)
- [Running in scos-sensor](#4-running-in-scos-sensor)
- [Development](#5-development)
- [License](#6-license)
- [Contact](#7-contact)

## 3. Overview of Repo Structure

- scos_actions/actions: This includes the base Action class, signals, and the following
  common action classes:
  - acquire_single_freq_fft: performs FFTs and calculates mean, median, min, max, and
    sample statistics at a single center frequency.
  - acquire_single_freq_tdomain_iq: acquires IQ data at a single center frequency.
  - acquire_stepped_freq_tdomain_iq: acquires IQ data at multiple center frequencies.
  - sync_gps: gets GPS location and syncs the host to GPS time
  - monitor_radio: ensures a signal analyzer is available and is able to maintain a
    connection to the computer.
- scos_actions/configs/actions: This folder contains the yaml files with the parameters
  used to initialize the actions described above.
- scos_actions/discover: This includes the code to read yaml files and make actions
  available to scos-sensor.
- scos_actions/hardware: This includes the radio interface and GPS interface used by
  the actions and the mock radio. The radio interface is intended to represent
  universal functionality that is common across all signal analyzers. The specific
  implementations of the radio interface for particular signal analyzers are provided
  in separate repositories like [scos-usrp](https://github.com/NTIA/scos-usrp).

## 4. Running in scos-sensor

Requires pip>=18.1 (upgrade using python3 -m pip install --upgrade pip) and
python>=3.6.

1. Clone scos-sensor: git clone <https://github.com/NTIA/scos-sensor.git>
1. Navigate to scos-sensor: `cd scos-sensor`
1. In scos-sensor/src/requirements.txt, comment out the following line:
   `scos_usrp @ git+ https://github.com/NTIA/scos-usrp@master#egg=scos_usrp`
1. Make sure scos_actions dependency is added to scos-sensor/src/requirements.txt. If
   you are using a different branch than master, change master in the following line to
   the branch you are using:
   `scos_actions @ git+https://github.com/NTIA/scos-actions@master#egg=scos_actions`
1. If it does not exist, create env file while in the root scos-sensor directory:
   `cp env.template ./env`
1. In env file, change `BASE_IMAGE=ubuntu:18.04` (at the bottom of the file)
1. Set `MOCK_RADIO` and `MOCK_RADIO_RANDOM` equal to 1 in docker-compose.yml
1. Get environment variables: `source ./env`
1. Build and start containers: `docker-compose up -d --build --force-recreate`

If scos-actions is installed to scos-sensor as a plugin, the following three
parameterized actions are offered for testing using a mock signal analyzer; their
parameters are defined in scos_actions/configs/actions.

- test_multi_frequency_iq_action
- test_single_frequency_iq_action
- test_single_frequency_m4s_action

## 5. Development

This repository is intended to be used by all scos-sensor plugins. Therefore, only
universal actions that apply to most RF measurement systems should be added to
scos-actions. Custom actions for specific hardware should be added to plugins in
repositories supporting that specific hardware. New functionality could be added to the
[radio interface defined in this repository](scos_actions/hardware/radio_iface.py) if
it is something that can be supported by most signal analyzers.

### Requirements and Configuration

Requires pip>=18.1 (upgrade using `python3 -m pip install --upgrade pip`) and
python>=3.6.

It is highly recommended that you first initialize a virtual development environment
using a tool such a conda or venv. The following commands create a virtual environment
using venv and install the required dependencies for development and testing.

```bash
python3 -m venv ./venv
source venv/bin/activate
python3 -m pip install --upgrade pip # upgrade to pip>=18.1
python3 -m pip install -r requirements-dev.txt
```

### Running Tests

Ideally, you should add a test that covers any new feature that you add. If you've done
that, then running the included test suite is the easiest way to check that everything
is working. In any case, all tests should be run after making any local modifications
to ensure that you haven't caused a regression.

scos-actions uses [pytest](https://docs.pytest.org/en/stable/) for testing.

[tox](https://tox.readthedocs.io/en/latest/) is a tool that can run all available tests
in a virtual environment against all supported versions of Python. Running `pytest`
directly is faster but running `tox` is a more thorough test.

The following commands can be used to run tests.

```bash
pytest          # faster, but less thorough
tox             # tests code in clean virtualenv
tox --recreate  # if you change `requirements.txt`
tox -e coverage # check where test coverage lacks
```

### Committing

Besides running the test suite and ensuring that all tests are passed, we also expect
all Python code that is checked in to have been run through an auto-formatter.

This project uses a Python auto-formatter called Black. Additionally, import statement
sorting is handled by isort.

There are several ways to autoformat your code before committing. First, IDE
integration with on-save hooks is very useful. Second, if you've already pip-installed
the dev requirements from the section above, you already have a utility called
pre-commit installed that will automate setting up this project's git pre-commit hooks.
Simply type the following *once*, and each time you make a commit, it will be
appropriately autoformatted.

```bash
pre-commit install
```

You can manually run the pre-commit hooks using the following command.

```bash
pre-commit run --all-files
```

In addition to Black and isort, various other pre-commit tools are enabled including
markdownlint. Markdownlint will show an error message if it detects any style issues in
markdown files. See .pre-commit-config.yaml for a list of pre-commit tools enabled for
this repository.

### Supporting a Different Signal Analyzer

[scos_usrp](https://github.com/NTIA/scos-usrp) adds support for the Ettus B2xx line of
software-defined radios to scos-sensor. Follow these instructions to add support for
another signal analyzer with a Python API.

- Create a new repository called scos-[signal analyzer name].
- Create a new virtual environment and activate it:
  `python3 -m venv ./venv && source venv/bin/activate`.
  Upgrade pip: `python3 -m pip install --upgrade pip`.
- In the new repository, add this scos-actions repository as a dependency and create a
  class that inherits from the [RadioInterface](scos_actions/hardware/radio_iface.py)
  abstract class. Add properties or class variables for the parameters needed to
  configure the radio.
- Create .yml files with the parameters needed to run the actions imported from
  scos-actions using the new signal analyzer. Put them in the new repository in
  `configs/actions`. This should contain the parameters needed by the action as well as
  the radio settings based on which properties or class variables were implemented in
  the radio class in the previous step. The measurement actions in scos-actions are
  configured to check if any yaml parameters are available as attributes in the radio
  object, and to set them to the given yaml value if available. For example, if the new
  radio class has a bandwidth property, simply add a bandwidth parameter to the yaml
  file. Alternatively, you can create custom actions that are unique to the hardware.
  See [Adding Actions](#adding-actions) subsection below.
- In the new repository, add a `discover/__init__.py` file. This should contain a
  dictionary called `actions` with a key of action name and a value of action object.
  You can use the [init()`](scos_actions/discover/__init__.py) and/or the
  [load_from_yaml()](scos_actions/discover/yaml.py) methods provided in this repository
  to look for yaml files and initialize actions. These methods allow you to pass your
  new radio object to the action's constructor. You can use the existing action classes
  [defined in this repository](scos_actions\actions\__init__.py) or
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
Then add the `sync_gps` and `monitor_radio` actions to your `actions` dictionary,
passing the gps object to the `SyncGps` constructor, and the radio object to the
`RadioMonitor` constructor. See the example in the [Adding Actions subsection](
    #adding-actions) below.

The final step would be to add a `setup.py` to allow for installation of the new
repository as a Python package. You can use the [setup.py](setup.py) in this repository
as a reference. You can find more information about Python packaging [here](
    https://packaging.python.org/tutorials/packaging-projects/). Then add the new
repository as a dependency to [scos-sensor requirements.txt](
    https://github.com/NTIA/scos-sensor/blob/SMBWTB475_refactor_radio_interface/src/requirements.txt)
using the following format:
`<package_name> @ git+<link_to_github_repo>@<branch_name>#egg=<package_name>`. If
specific drivers are required for your signal analyzer, you can attempt to link to them
within the package or create a docker image with the necessary files. You can host the
docker image as a GitHub package. Then, when running scos-sensor, set the environment
variable `BASE_IMAGE=<image tag>`.

### Adding Actions

To expose a new action to the API, check out the available [action classes](
    scos_actions/actions/__init__.py). An *action* is a parameterized implementation of
an action class. If an existing class covers your needs, you can simply create yaml
configs and use the `init` method in `scos_actions.discover` to make these actions
available.

```python
from scos_actions.discover import init
from scos_usrp.hardware import gps, radio
actions = {

"monitor_usrp": RadioMonitor(radio),
"sync_gps": SyncGps(gps),

}

yaml_actions, yaml_test_actions = init(radio=radio, yaml_dir=ACTION_DEFINITIONS_DIR)

actions.update(yaml_actions)
```

Pass the `init` method the implementation of the radio interface and the directory
where the yaml files are located.

If no existing action class meets your needs, see [Writing Custom Actions](
    #writing-custom-actions).

#### Creating a yaml config file for an action

Actions can be manually initialized in `discover/__init__.py`, but an easier method for
non-developers and configuration-management software is to place a yaml file in the
`configs/actions` directory which contains the action class name and parameter
definitions.

The file name can be anything. Files must end in .yml.

The action initialization logic parses all yaml files in this directory and registers
the requested actions in the API.

Let's look at an example.

##### Example

Let's say we want to make an instance of the `SingleFrequencyFftAcquisition`.

First, create a new yaml file in the
`scos_actions/configs/actions` directory. In this example we're going to create
an acquisition for the LTE 700 C band downlink, so we'll call it `acquire_700c_dl.yml`.

Next, we want to find the appropriate string key for the
`SingleFrequencyFftAcquisition` class. Look in [actions/\_\_init\_\_.py](
    scos_actions/actions\]/__init__.py) at the action_classes dictionary. There, we
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
class SingleFrequencyFftAcquisition(SingleFrequencyTimeDomainIqAcquisition):
    """Perform m4s detection over requested number of single-frequency FFTs.

    :param parameters: The dictionary of parameters needed for the action and the radio.

    The action will set any matching attributes found in the radio object. The following
    parameters are required by the action:

        name: name of the action
        frequency: center frequency in Hz
        fft_size: number of points in FFT (some 2^n)
        nffts: number of consecutive FFTs to pass to detector
```

Then look at the docstring for the radio class being used. This example will use the
[MockRadio](scos_actions/hardware/mocks/mock_radio.py). That file contains the
following:

```python
class MockRadio(RadioInterface):
    """
    MockRadio is mock radio object for testing.

    The following parameters are required for measurements:
    sample_rate: requested sample rate in samples/second
    frequency: center frequency in Hz
    gain: requested gain in dB
    """
```

Lastly, simply modify the yaml file to define any required parameters from the action
and radio. Note that the radio parameter is a special parameter that will get passed in
separately when the action is initialized from the yaml. Therefore, it does not need to
be defined in the yaml file.

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

"Actions" are one of the main concepts used by scos-sensor. At a high level, they are
the things that the sensor owner wants the sensor to be able to *do*. At a lower level,
they are simply Python classes with a special method `__call__`. Actions use [Django
Signals](https://docs.djangoproject.com/en/3.1/topics/signals/) to provide data and
results to scos-sensor.

Start by looking at the [Action base class](scos_actions/actions/interfaces/action.py).
It includes some logic to parse a description and summary out of the action class's
docstring, and a `__call__` method that must be overridden. If you pass
`admin_only=True` to this base class, the API will not make it or any data it created
available to non-admin users.

A new custom action can inherit from the existing action classes to reuse and build
upon existing functionality. For example, the `SingleFrequencyFftAcquisition` and
`SteppedFrequencyTimeDomainIqAcquisition` classes inherit from the
`SingleFrequencyTimeDomainIqAcquisition` class.

Depending on the type of action, a signal should be sent upon action completion. This
enables scos-sensor to do something with the results of the action. This could range
from storing measurement data to recycling a docker container or to fixing an unhealthy
connection to the signal analyzer. You can see the available signals in
[scos_actions/actions/interfaces/signals.py](
    scos_actions/actions/interfaces/signals.py). The following signals are currently
offered:

- `measurement_action_completed` - signal expects task_id, data, and metadata
- `location_action_completed` - signal expects latitude and longitude
- `monitor_action_completed` - signal expects boolean indicating if the radio is
  healthy

New signals can be added. However, corresponding signal handlers must be added to
scos-sensor to receive the signals and process the results.

##### Adding custom action to scos-actions

A custom action meant to be re-used by other plugins can live in scos-actions. It can
be instantiated using a yaml file, or directly in the `actions` dictionary in the
`discover/__init__.py` module. This can be done in scos-actions with a mock radio.
Plugins supporting other hardware would need to import the action from scos-actions.
Then it can be instantiated in that plugin’s actions dictionary in its discover module,
or in a yaml file living in that plugin (as long as its discover module includes the
required code to parse the yaml files).

##### Adding system or hardware specific custom action

In the repository that provides the plugin to support the hardware being used, add the
action to the `actions` dictionary in the `discover/__init__.py` file. Optionally,
initialize the action using a yaml file by importing the yaml initialization code from
scos-actions. For an example of this, see [Adding Actions subsection](#adding-actions)
above.

## 6. License

See [LICENSE](LICENSE.md).

## 7. Contact

For technical questions about scos-actions, contact Justin Haze, jhaze@ntia.gov
