# YAML-Defined Action Initialization

Actions can be manually initialized in
[`discover/__init__.py`](../discover/__init__.py), but an easier method for
non-developers and configuration-management software is to place a YAML file in the
[configs\actions directory](../configs/actions) which contains the action class name
and parameter definitions.

The file name can be anything. Files must end in `.yml`.

The action initialization logic parses all YAML files in this directory and registers
the requested actions in the API.

Let's look at an example.

## Example

Let's say we want to make an instance of the
[`SingleFrequencyFftAcquisition`](../actions/acquire_single_freq_fft.py).

First, create a new YAML file in this directory. In this example we're going to create
an acquisition for the LTE 700c band downlink, so we'll call it `acquire_700c_dl.yml`.

Next, we want to find the appropriate string key for the
`SingleFrequencyFftAcquisition` class. Look in
[`actions/__init__.py`](../actions/__init__.py) at the `action_classes` dictionary.
There, we see:

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
up [`actions/acquire_single_freq_fft.py`](../actions/acquire_single_freq_fft.py) and
look at the documentation for the class to see what parameters are available and what
units to use, etc.

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

    For the parameters required by the radio, see the documentation for the radio being used.

    :param radio: instance of RadioInterface
    """
    ...
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

Lastly, simply modify the YAML file to define any required parameters from the action
and radio. Note that the radio parameter is a special parameter that will get passed
in separately when the action is initialized from the YAML. Therefore, it does not need
to be defined in the YAML file.

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

You're done. You can define multiple actions in a single file.
