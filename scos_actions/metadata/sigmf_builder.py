import json
from typing import List, Union

import msgspec
from sigmf import SigMFFile

from scos_actions.metadata.structs.capture import CaptureSegment
from scos_actions.metadata.structs.ntia_algorithm import DFT, DigitalFilter, Graph
from scos_actions.metadata.structs.ntia_diagnostics import Diagnostics
from scos_actions.metadata.structs.ntia_scos import Action, ScheduleEntry
from scos_actions.metadata.structs.ntia_sensor import Sensor
from scos_actions.metadata.utils import msgspec_dec_dict, msgspec_enc

# Global info which is ALWAYS true for SCOS-generated recordings
GLOBAL_INFO = {
    "core:version": "v1.0.0",
    "core:extensions": [
        {
            "name": "ntia-algorithm",
            "version": "v2.0.0",
            "optional": False,
        },
        {
            "name": "ntia-core",
            "version": "v2.0.0",
            "optional": False,
        },
        {
            "name": "ntia-diagnostics",
            "version": "v1.0.0",
            "optional": True,
        },
        {
            "name": "ntia-environment",
            "version": "v1.0.0",
            "optional": True,
        },
        {
            "name": "ntia-scos",
            "version": "v1.0.0",
            "optional": True,
        },
        {
            "name": "ntia-sensor",
            "version": "v2.0.0",
            "optional": False,
        },
        {
            "name": "ntia-nasctn-sea",
            "version": "v0.4.0",
            "optional": True,
        },
    ],
    "core:recorder": "SCOS",
}


class SigMFBuilder:
    def __init__(self):
        self.sigmf_md = SigMFFile()
        self.sigmf_md.set_global_info(GLOBAL_INFO.copy())
        self.metadata_generators = {}

    def reset(self):
        self.sigmf_md = SigMFFile()
        self.sigmf_md.set_global_info(GLOBAL_INFO.copy())

    @property
    def metadata(self):
        self.build()
        return self.sigmf_md._metadata

    def set_data_type(
        self,
        is_complex: bool,
        sample_type: str = "floating-point",
        bit_width: int = 32,
        endianness: str = "little",
    ) -> None:
        """
        Set the global ``core:datatype`` field of a SigMF metadata file.

        Defaults to "cf32_le" for complex 32 bit little-endian floating point.

        :param is_complex: True if the data is complex, False if the data is real.
        :param sample_type: The sample type, defaults to "floating-point"
        :param bit_width: The bit-width of the data, defaults to 32
        :param endianness: Data endianness, defaults to "little". Provide an empty
            string if the data is saved as bytes.
        :raises ValueError: If ``bit_width`` is not an integer.
        :raises ValueError: If ``sample_type`` is not one of: "floating-point",
            "signed-integer", or "unsigned-integer".
        :raises ValueError: If the endianness is not one of: "big", "little", or
            "" (an empty string).
        """
        if not isinstance(bit_width, int):
            raise ValueError("Bit-width must be an integer.")

        dset_fmt = "c" if is_complex else "r"

        if sample_type == "floating-point":
            dset_fmt += "f"
        elif sample_type == "signed-integer":
            dset_fmt += "i"
        elif sample_type == "unsigned-integer":
            dset_fmt += "u"
        else:
            raise ValueError(
                'Sample type must be one of: "floating-point", "signed-integer", or "unsigned-integer"'
            )

        dset_fmt += str(bit_width)

        if endianness == "little":
            dset_fmt += "_le"
        elif endianness == "big":
            dset_fmt += "_be"
        elif endianness != "":
            raise ValueError(
                'Endianness must be either "big", "little", or "" (for saving bytes)'
            )

        self.sigmf_md.set_global_field("core:datatype", dset_fmt)

    def set_sample_rate(self, sample_rate: float) -> None:
        """
        Set the value of the Global "core:sample_rate" field.

        :param sample_rate: The sample rate of the signal in samples
            per second.
        """
        self.sigmf_md.set_global_field("core:sample_rate", sample_rate)

    # core:version omitted, set by GLOBAL_INFO on metadata init

    def set_num_channels(self, num_channels: int) -> None:
        """
        Set the value of the Global "core:num_channels" field.

        :param num_channels: Total number of interleaved channels in
            the Dataset file.
        """
        self.sigmf_md.set_global_field("core:num_channels", num_channels)

    def set_sha512(self, sha512: str) -> None:
        """
        Set the value of the Global "core:sha512" field.

        :param sha512: The SHA512 hash of the Dataset file associated
            with the SigMF file.
        """
        self.sigmf_md.set_global_field("core:sha512", sha512)

    def set_offset(self, offset: int) -> None:
        """
        Set the value of the Global "core:offset" field.

        :param offset: The index number of the first sample in the Dataset.
            Typically used when a Recording is split over multiple files.
            All sample indices in SigMF are absolute, and so all other indices
            referenced in metadata for this recording SHOULD be greater than or
            equal to this value.
        """
        self.sigmf_md.set_global_field("core:offset", offset)

    def set_description(self, description: str) -> None:
        """
        Set the value of the Global "core:description" field.

        :param description: A text description of the SigMF recording.
        """
        self.sigmf_md.set_global_field("core:description", description)

    def set_author(self, author: str) -> None:
        """
        Set the value of the Global "core:author" field.

        :param author: A text identifier for the author potentially including
            name, handle, email, and/or other ID like Amateur Call Sign. For
            example "Bruce Wayne bruce@waynetech.com" or "Bruce (K3X)".
        """
        self.sigmf_md.set_global_field("core:author", author)

    def set_meta_doi(self, meta_doi: str) -> None:
        """
        Set the value of the Global "core:meta_doi" field.

        :param author: The registered DOI (ISO 26324) for a Recording's
            Metadata file.
        """
        self.sigmf_md.set_global_field("core:meta_doi", meta_doi)

    def set_data_doi(self, data_doi: str) -> None:
        """
        Set the value of the Global "core:data_doi" field.

        :param data_doi: The registered DOI (ISO 26324) for a Recording's
            Dataset file.
        """
        self.sigmf_md.set_global_field("core:data_doi", data_doi)

    # core:recorder omitted, set by GLOBAL_INFO on metadata init

    def set_license(self, license: str) -> None:
        """
        Set the value of the Global "core:license" field.

        :param license: A URL for the license document under which the
            Recording is offered.
        """
        self.sigmf_md.set_global_field("core:license", license)

    def set_hw(self, hw: str) -> None:
        """
        Set the value of the Global "core:hw" field.

        :param hw: A text description of the hardware used to make the
            Recording.
        """
        self.sigmf_md.set_global_field("core:hw", hw)

    def set_dataset(self, dataset: str) -> None:
        """
        Set the value of the Global "core:dataset" field.

        :param dataset: The full filename of the Dataset file this Metadata
            file describes.
        """
        self.sigmf_md.set_global_field("core:dataset", dataset)

    def set_trailing_bytes(self, trailing_bytes: int) -> None:
        """
        Set the value of the Global "core:trailing_bytes" field.

        :param trailing_bytes: The number of bytes to ignore are the end of
            a Non-Conforming Dataset file.
        """
        self.sigmf_md.set_global_field("core:trailing_bytes", trailing_bytes)

    def set_metadata_only(self, metadata_only: bool) -> None:
        """
        Set the value of the Global "core:metadata_only" field.

        :param metadata_only: Indicates the Metadata file is intentionally
            distributed without the Dataset.
        """
        self.sigmf_md.set_global_field("core:metadata_only", metadata_only)

    def set_geolocation(self, geolocation: dict) -> None:
        """
        Set the value of the Global "core:geolocation" field.

        :param geolocation: A dictionary containing the GeoJSON Point representation
            of the sensor's location.
        """
        self.sigmf_md.set_global_field("core:geolocation", geolocation)

    # core:extensions omitted, set dynamically when metadata is built

    def set_collection(self, collection: str) -> None:
        """
        Set the value of the Global "core:collection" field.

        :param collection: The base filename of a `collection` with which
            this Recording is associated.
        """
        self.sigmf_md.set_global_field("core:collection", collection)

    ### ntia-algorithm v2.0.0 ###

    def set_data_products(self, data_products: List[Graph]) -> None:
        """
        Set the value of the Global "ntia-algorithm:data_products" field.

        :param data_products: List of data products produced for each capture.
        """
        self.sigmf_md.set_global_field("ntia-algorithm:data_products", data_products)

    def set_processing(self, processing: List[str]) -> None:
        """
        Set the value of the Global "ntia-algorithm:processing" field.

        :param processing: IDs associated with the additional metadata describing
            processing applied to ALL data.
        """
        self.sigmf_md.set_global_field("ntia-algorithm:processing", processing)

    def set_processing_info(
        self, processing_info: List[Union[DigitalFilter, DFT]]
    ) -> None:
        """
        Set the value of the Global "ntia-algorithm:processing_info" field.

        :param processing_info: List of objects that describe processing used to
            generate some of the data products listed in `ntia-algorithm:data_products`.
            Supported objects include `DigitalFilter` and `DFT`. The IDs of any processing
            performed on ALL data products should be listed in `ntia-algorithm:processing`.
        """
        self.sigmf_md.set_global_field(
            "ntia-algorithm:processing_info", processing_info
        )

    ### ntia-core v2.0.0 ###

    def set_classification(self, classification: str) -> None:
        """
        Set the value of the Global "ntia-core:classification" field.

        :param classification: The classification markings for the acquisition,
            e.g., `"UNCLASSIFIED"`, `"CONTROLLED//FEDCON"`, `"SECRET"`
        """
        self.sigmf_md.set_global_field("ntia-core:classification", classification)

    ### ntia-diagnostics v1.0.0 ###

    def set_diagnostics(self, diagnostics: Diagnostics) -> None:
        """
        Set the value of the Global "ntia-diagnostics:diagnostics" field.

        :param diagnostics: Metadata for capturing component diagnostics.
        """
        self.sigmf_md.set_global_field("ntia-diagnostics:diagnostics", diagnostics)

    ### ntia-nasctn-sea v0.4.0 ###

    def set_max_of_max_channel_powers(
        self, max_of_max_channel_powers: List[float]
    ) -> None:
        """
        Set the value of the Global "ntia-nasctn-sea:max_of_max_channel_powers" field.

        :param max_of_max_channel_powers: The maximum of the maximum power per channel, in dBm.
        """
        self.sigmf_md.set_global_field(
            "ntia-nasctn-sea:max_of_max_channel_powers", max_of_max_channel_powers
        )

    def set_median_of_mean_channel_powers(
        self, median_of_mean_channel_powers: List[float]
    ) -> None:
        """
        Set the value of the Global "ntia-nasctn-sea:median_of_mean_channel_powers" field.

        :param median_of_mean_channel_powers: The median of the mean power per channel, in dBm.
        """
        self.sigmf_md.set_global_field(
            "ntia-nasctn-sea:median_of_mean_channel_powers",
            median_of_mean_channel_powers,
        )

    ### ntia-scos v1.0.0 ###

    def set_schedule(self, schedule: ScheduleEntry) -> None:
        """
        Set the value of the Global "ntia-scos:schedule" field.

        :param schedule: Metadata that describes the schedule that caused an
            action to be performed.
        """
        self.sigmf_md.set_global_field("ntia-scos:schedule", schedule)

    def set_action(self, action: Action) -> None:
        """
        Set the value of the Global "ntia-scos:action" field.

        :param action: Metadata that describes the action that was performed.
        """
        self.sigmf_md.set_global_field("ntia-scos:action", action)

    def set_task(self, task: int) -> None:
        """
        Set the value of the Global "ntia-scos:task" field.

        :param task: Unique identifier that increments with each task performed
            as a result of a schedule entry.
        """
        self.sigmf_md.set_global_field("ntia-scos:task", task)

    def set_recording(self, recording: int) -> None:
        """
        Set the value of the Global "ntia-scos:recording" field.

        :param recording: Unique identifier that increments with each recording
            performed in a task. The recording SHOULD be indicated for tasks that
            perform multiple recordings.
        """
        self.sigmf_md.set_global_field("ntia-scos:recording", recording)

    ### ntia-sensor v2.0.0 ###

    def set_sensor(self, sensor: Sensor) -> None:
        """
        Set the value of the Global "ntia-sensor:sensor" field.

        :param sensor: Describes the sensor model components.
        """
        self.sigmf_md.set_global_field("ntia-sensor:sensor", sensor)

    def add_capture(self, capture: CaptureSegment) -> None:
        capture_dict = json.loads(msgspec_enc.encode(capture))
        sample_start = capture_dict.pop("core:sample_start")
        self.sigmf_md.add_capture(sample_start, metadata=capture_dict)

    def add_annotation(self, start_index, length, annotation_md):
        self.sigmf_md.add_annotation(
            start_index=start_index, length=length, metadata=annotation_md
        )

    def set_last_calibration_time(self, last_cal_time):
        self.sigmf_md.set_global_field(
            "ntia-sensor:calibration_datetime", last_cal_time
        )

    def add_to_global(self, key, value):
        self.sigmf_md.set_global_field(key, value)

    def add_metadata_generator(self, key, generator):
        self.metadata_generators[key] = generator

    def remove_metadata_generator(self, key):
        self.metadata_generators.pop(key, "")

    def build(self):
        # Convert msgspec Structs to dictionaries to support
        # serialization by standard json.dumps (e.g., in SCOS Sensor)
        # NOTE: This should be removed in the future, and msgspec should
        # be used natively throughout SCOS. This would require changing
        # the serialization implementations in SCOS Sensor, which makes sense
        # to do when a python library is created to support sigmf-ns-ntia
        for k, v in self.sigmf_md._metadata["global"].items():
            if issubclass(type(v), msgspec.Struct):
                # Recursion is not needed: encode/decode will convert nested objects
                self.sigmf_md._metadata["global"][k] = msgspec_dec_dict.decode(
                    msgspec_enc.encode(v)
                )
            if isinstance(v, list):
                for i, item in enumerate(v):
                    if issubclass(type(item), msgspec.Struct):
                        v[i] = msgspec_dec_dict.decode(msgspec_enc.encode(item))
                self.sigmf_md._metadata["global"][k] = v
        for i, capture in enumerate(self.sigmf_md._metadata["captures"]):
            for k, v in capture.items():
                if issubclass(type(v), msgspec.Struct):
                    self.sigmf_md._metadata["captures"][i][k] = msgspec_dec_dict.decode(
                        msgspec_enc.encode(v)
                    )
        for metadata_creator in self.metadata_generators.values():
            metadata_creator.create_metadata(self)
