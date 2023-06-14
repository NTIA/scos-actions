from typing import Optional

SIGMF_OBJECT_KWARGS = {
    "omit_defaults": True,
    "forbid_unknown_fields": True,
}


def prepend_core_namespace(name: str) -> Optional[str]:
    return f"core:{name}"


def construct_geojson_point(
    longitude: float, latitude: float, altitude: float = None
) -> dict:
    """
    Construct a dict containing the GeoJSON Point representation of a location.

    :param longitude: The longitude, in decimal degrees WGS84.
    :param latitude: The latitude, in decimal degrees WGS84.
    :param altitude: (Optional) The altitude, in meters above the WGS84 ellipsoid.
    :return: A dictionary containing the GeoJSON Point representation
        of the input.
    """
    geolocation = {"type": "Point"}
    geolocation["coordinates"] = [longitude, latitude]
    if altitude is not None:
        geolocation["coordinates"].append(altitude)
    return geolocation
