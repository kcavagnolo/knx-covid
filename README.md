# Knoxville "Metro" COVID19 Cases

What's the story with COVID19 cases in Knoxville Metro?

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

Use the [Free API](https://covid19api.com/):

```bash
curl --request GET 'https://api.covid19api.com/dayone/country/us/status/confirmed' | jq | grep -B 2 -A 6 "Tennessee, Knox"
```

## Credits

- [COVID case data](https://github.com/pomber/covid19)
- [Tennessee county boundaries](https://tn-tnmap.opendata.arcgis.com/datasets/TWRA::tn-counties)
- [Hospital Referral Regions( HRR)](https://hub.arcgis.com/datasets/fedmaps::hospital-referral-regions)
- [Mapbox](https://www.mapbox.com/about/maps/)
- [OpenStreetMap](http://www.openstreetmap.org/about/)
- [Mapbox contributor plugins](https://docs.mapbox.com/mapbox-gl-js/plugins/)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](LICENSE).

![Creative Commons License](https://i.creativecommons.org/l/by-sa/4.0/88x31.png "license")
