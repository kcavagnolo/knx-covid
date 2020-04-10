# Knoxville "Metro" COVID19 Cases

What's the story with COVID19 cases in Knoxville Metro?

**Updated on 2020-04-10T02:20:48.614888+00:00**

## Defining Knoxville Metro

Before making an estimate, let us first define what data interests us. In this case, consider the Knoxville metro area as all locations within a 30 minute or 60 minute drive time, aka, the catchment area. [See details on this map.](https://www.kcavagnolo.com/knx-covid/) The isochrone you see on the map (green overlay) covers several counties (orange outlines; click for name) and the associated hospitals (pink points). The blue dashed line indicates the Hospital Referral Region defined by the [Center for Medicare & Medicaid Services (CMS)](https://www.cms.gov/) and is provided as a reference point to validate regional coverage for healthcare centered on Knoxville.

## Projections

To estimate the total confirmed COVID19 cases in the Knoxville metro area, I use the simplest population model: [the logistic function](https://en.wikipedia.org/wiki/Logistic_function#In_ecology:_modeling_population_growth).

Best case scenario: new case growth is [regulated](https://www.khanacademy.org/science/biology/ecology/population-growth-and-regulation/a/exponential-logistic-growth).

![Knoxville Metro COVID19 Projected Cumulative Cases](/imgs/metro-cases-all-fit-best.png)

---

Worst case scenario: new case growth is [unregulated](https://facebook.github.io/prophet/docs/saturating_forecasts.html#forecasting-growth).

![Knoxville Metro COVID19 Projected Cumulative Cases](/imgs/metro-cases-all-fit-worst.png)

Based on data provided by [Models of Infectious Disease Agent Study (MIDAS)](https://midasnetwork.us/covid-19/), the rate of COVID cases requiring hospitalization is 10%-20%. The rate of hospitalized cases requiring admittance to an ICU is 5%. The Knoxville Metro area has an [ICU capacity of 295 beds](https://www.wbir.com/article/news/health/coronavirus/knox-co-has-173-icu-beds-all-others-in-our-area-combined-have-131-its-indicative-of-a-broader-problem/51-7727cc22-384c-4e67-8008-ec770b949b25). Assuming the MIDAS rates, and all beds being available, the critical case load which results in all ICU capacity being consumed is 29,500 cases. In the worst case scenario model (as of April 1), this occurs around April 15th.

![Knoxville Metro COVID19 Projected Cumulative Cases](/imgs/wc.gif)

---

![Knoxville Metro COVID19 Forecasted Daily New Cases](/imgs/metro-cases-all-daily-forecasted.png)

---

![Knoxville Metro COVID19 Cumulative Cases](/imgs/metro-cases-all.png)

---

![Knoxville Metro COVID19 Cumulative Cases by County](/imgs/metro-cases-county.png)

## Installation & Usage

Created using Python 3.8.0. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements. I recommend using [`pyenv-virtualenv`](https://github.com/pyenv/pyenv-virtualenv) so you don't mess up your Python env.

```bash
cd analysis
pip install -r requirements
```

A command line script is provided in [/analysis](/analysis):

```bash
python3 analysis/update.py -d data/ -i imgs/ -v
```

A deprecated interactive Jupyter notebook is also in the `analysis/` directory.

To create the animation:

```bash
ffmpeg -framerate 2 -i imgs/wc_%04d.png -r 60 -vcodec copy -acodec copy -vcodec libx264 -pix_fmt yuv420p -y imgs/wc.mp4
ffmpeg -i imgs/wc.mp4 -filter_complex "fps=2,scale=-1:640,setsar=1,palettegen" -y imgs/palette.png
ffmpeg -i imgs/wc.mp4 -i imgs/palette.png -filter_complex "[0]fps=2,scale=-1:640,setsar=1[x];[x][1:v]paletteuse" -y imgs/wc.gif
```

### Code Quality

1. Install and configure [Sonarqube](https://docs.sonarqube.org/latest/) or launch a maintained container:

   ```bash
   docker run -it --name sonarqube -p 9000:9000 sonarqube
   ```

2. Setup a `sonar-project.properties` file with configurations for scans.

3. Run a scan:

   ```bash
   sonar-scanner
   ```

## Credits

- [Data from The New York Times, based on reports from state and local health agencies](https://github.com/nytimes/covid-19-data)
- [Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE) raw COVID data](https://github.com/CSSEGISandData/COVID-19)
- [MIDAS 2019 Novel Coronavirus GitHub Repository](https://github.com/midas-network/COVID-19)
- [Tennessee county boundaries](https://tn-tnmap.opendata.arcgis.com/datasets/TWRA::tn-counties)
- [Tennessee hospital locations](https://hub.arcgis.com/datasets/TDH::hospitals)
- [CMS Hospital Referral Regions (HRR)](https://hub.arcgis.com/datasets/fedmaps::hospital-referral-regions)
- [Mapbox maps](https://www.mapbox.com/about/maps/)
- [Mapbox contributor plugins](https://docs.mapbox.com/mapbox-gl-js/plugins/)
- [OpenStreetMap road network data](http://www.openstreetmap.org/about/)
- [Facebook Prophet forecasting](https://github.com/facebook/prophet)
- [Seaborn data visualization](https://github.com/mwaskom/seaborn)

## References

- [COVID aggregated data](https://github.com/pomber/covid19)
- [COVID aggregated data API](https://covid19api.com/)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](LICENSE).

![Creative Commons License](https://i.creativecommons.org/l/by-sa/4.0/88x31.png "license")
