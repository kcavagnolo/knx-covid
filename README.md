# Knoxville "Metro" COVID19 Cases

What's the story with COVID19 cases in Knoxville Metro?

<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d826061.3845233425!2d-84.64513901086222!3d36.025121292986455!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x885c2357ec3c75fb%3A0x5bc79b51196338be!2sKnoxville%20Metropolitan%20Area%2C%20TN!5e0!3m2!1sen!2sus!4v1585492021049!5m2!1sen!2sus" width="600" height="450" frameborder="0" style="border:0;" allowfullscreen="" aria-hidden="false" tabindex="0"></iframe>

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

* https://github.com/pomber/covid19

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
