# Knoxville "Metro" COVID19 Cases

What's the story with COVID19 cases in Knoxville Metro?

**LAST UPDATED: Mon Mar 30 15:03:29 EDT 2020**

version: v0.2.0

## Defining Knoxville Metro

Before making an estimate, let us first define what data interests us. In this case, consider the Knoxville metro area as all locations within a 30 minute or 60 minute drive time, aka, the catchment area. [See details on this map.](https://www.kcavagnolo.com/knx-covid/) The isochrone you see on the map (green overlay) covers several counties (orange outlines; click for name) and the associated hospitals (pink points). The blue dashed line indicates the Hospital Referral Region defined by the [Center for Medicare & Medicaid Services (CMS)](https://www.cms.gov/) and is provided as a reference point to validate regional coverage for healthcare centered on Knoxville.

## Projections

To estimate the total confirmed COVID19 cases in the Knoxville metro area, I use the simplest population model: [the logistic function](https://en.wikipedia.org/wiki/Logistic_function#In_ecology:_modeling_population_growth). This model has one assumption: [growth of cases is capped by external forces](https://www.khanacademy.org/science/biology/ecology/population-growth-and-regulation/a/exponential-logistic-growth). There are no assumptions on what those forces are, just that they exert an immediate and irreversible influence on the growth rate.

Cumulative Knoxville Metro cases per county per day:
![Cumulative Knoxville Metro cases per county per day](/imgs/metro-county-cases.png)

Cumulative Knoxville Metro cases per day:
![Cumulative Knoxville Metro cases per day](/imgs/metro-all.png)

Best-fit logistics model for cumulative Knoxville Metro cases per day:
![Best-fit logistics model for cumulative Knoxville Metro cases per day](/imgs/metro-all-fit.png)

## Installation

Created using Python 3.8.0. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements:

```bash
pip install -r analysis/requirements
```

An interactive Jupyter notebook (deprecated) and command line script are provided in [/analysis](/analysis).

```bash
python3 analysis/knx_metro_covid.py
```

## Usage

Use the COVID free API to get aggregated data:

```bash
curl --request GET 'https://api.covid19api.com/dayone/country/us/status/confirmed' | jq | grep -B 2 -A 6 "Tennessee, Knox"
```

## Credits

- [COVID raw data](https://github.com/CSSEGISandData/COVID-19)
- [COVID aggregated data](https://github.com/pomber/covid19)
- [COVID aggregated data API](https://covid19api.com/)
- [Tennessee county boundaries](https://tn-tnmap.opendata.arcgis.com/datasets/TWRA::tn-counties)
- [Tennessee hospital locations](https://hub.arcgis.com/datasets/TDH::hospitals)
- [Hospital Referral Regions (HRR)](https://hub.arcgis.com/datasets/fedmaps::hospital-referral-regions)
- [Mapbox maps](https://www.mapbox.com/about/maps/)
- [OpenStreetMap road network data](http://www.openstreetmap.org/about/)
- [Mapbox contributor plugins](https://docs.mapbox.com/mapbox-gl-js/plugins/)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Please make sure to update tests as appropriate.

## License

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](LICENSE).

![Creative Commons License](https://i.creativecommons.org/l/by-sa/4.0/88x31.png "license")
