import os
from collections import namedtuple
from typing import Iterable, Union, Dict, Tuple
from gmplot import gmplot
import json
from functools import lru_cache
from hospitals.common import relative_path, get_csv_lines

Location = namedtuple('Location', 'lat, lng')


class Map(gmplot.GoogleMapPlotter):

    __locations = None

    def __init__(self, center_lat: Union[int, float] = 52.0, center_lng: Union[int, float] = 20.0,
                 zoom: int = 7, apikey: Union[str, None] = None):
        if apikey is None:
            apikey = os.environ.get('GOOGLE_API_KEY', '')

        super().__init__(center_lat, center_lng, zoom, apikey)

    def circles(self, locations: Iterable[Location], radius: Union[int, float] = 93.75*1000, **kwargs)-> None:
        kwargs.setdefault('color', 'red')
        kwargs.setdefault('face_alpha', 0.2)
        for loc in locations:
            self.circle(*loc, radius, **kwargs)

    @classmethod
    @lru_cache(maxsize=None)
    def get_cycle(cls, *args, **kwargs):
        return super().get_cycle(*args, **kwargs)

    @classmethod
    def locations(cls) -> Dict[str, Location]:
        if cls.__locations is None:
            cls.__locations = {row['city']: Location(float(row['lat']), float(row['lng']))
                               for row in get_csv_lines(relative_path(__file__, 'city-N-E.csv'), ';')}
        return cls.__locations

    @staticmethod
    @lru_cache()
    def country_border(country: str = 'Poland') -> Tuple[Location]:
        with open(relative_path(__file__, "poland_border.json")) as f:
            return tuple(Location(lat, lng) for lat, lng in json.load(f))


def main():
    m = Map()
    locations = (
        Location(52.20, 21),
        Location(54.32, 18.60),
    )

    m.circles(locations, 93.75*1000)

    lats, lngs = zip(*Map.country_border('Poland'))
    m.polygon(lats, lngs, 'green')

    m.draw('gmaps.html')


if __name__ == '__main__':
    main()
