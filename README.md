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

* COVID case data -- https://github.com/pomber/covid19
* Tennessee county boundaries -- https://tn-tnmap.opendata.arcgis.com/datasets/TWRA::tn-counties

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
