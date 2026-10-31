"""GPS coords to a place name people recognise.

Uses `reverse_geocoder`, ships GeoNames towns dataset inside the package.
"""

import contextlib
import io
from dataclasses import dataclass
from functools import cache, lru_cache

import pycountry
import reverse_geocoder as rg


@dataclass(frozen=True, slots=True)
class Place:
    town: str
    region: str
    country_code: str
    country: str
    abroad: bool

    @property
    def label(self) -> str:
        """Caption-ready name. Town at home, town plus country abroad."""
        return f"{self.town}, {self.country}" if self.abroad else self.town


@cache
def _geocoder() -> rg.RGeocoder:
    with contextlib.redirect_stdout(io.StringIO()):
        return rg.RGeocoder(mode=1, verbose=False)


def _country_name(code: str) -> str:
    country = pycountry.countries.get(alpha_2=code)
    if country is None:
        return code
    return getattr(country, "common_name", None) or country.name


@lru_cache(maxsize=10_000)
def _nearest_town(lat: float, lon: float) -> tuple[str, str, str]:
    hit = _geocoder().query([(lat, lon)])[0]
    return hit["name"], hit["admin1"], hit["cc"]


def place_name(lat: float | None, lon: float | None, home_country: str = "GB") -> Place | None:
    """Nearest town, or None without GPS. Rounded to ~100 m first so the cache works."""
    if lat is None or lon is None:
        return None
    town, region, code = _nearest_town(round(lat, 3), round(lon, 3))
    return Place(
        town=town,
        region=region,
        country_code=code,
        country=_country_name(code),
        abroad=code != home_country,
    )
