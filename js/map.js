// verify Mapbox GL JS support
var bodyElement = document.body;
var mapElement = document.getElementById('map');

// check for mapbox support
if (!mapboxgl.supported()) {
    alert('Mapbox GL is not supported by your browser');
    mapElement.innerHTML = mapboxgl.notSupportedReason({
        failIfMajorPerformanceCaveat: true
    });
    bodyElement.className = 'fill-red';
} else {
    // an access token
    mapboxgl.accessToken = 'pk.eyJ1Ijoia2NhdmFnbm9sbyIsImEiOiJjamtvOWExZHYydm1xM3BreG9pb3hza3J6In0.yKhLQH7NxC9rbsrCpilLQw';

    // initial values
    var urlBase = 'https://api.mapbox.com/isochrone/v1/mapbox/';
    var lon = -83.920592;
    var lat = 35.960750;
    var profile = 'driving';
    var minutes = 60;
    var colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e']
    var hoverId = null;
    var popup = new mapboxgl.Popup({
        closeButton: false,
        closeOnClick: false,
        offset: {
            "top": [0, 10],
            "bottom": [0, -10]
        }
    });
    var origin = {
        lon: lon,
        lat: lat
    };

    // load mapbox gl
    var map = new mapboxgl.Map({
        container: 'map',
        style: 'mapbox://styles/mapbox/dark-v10?optimize=true',
        center: [lon, lat],
        zoom: 8,
        attributionControl: false,
        failIfMajorPerformanceCaveat: true
    }).addControl(new mapboxgl.AttributionControl({
        compact: true,
        customAttribution: "<a href='https://github.com/kcavagnolo/knx-covid' target='_blank'>&copy; kcavagnolo</a>"
    }));

    // function to add isochrone
    function getIso() {
        var query = urlBase + profile + '/' + lon + ',' + lat + '?contours_minutes=' + minutes + '&polygons=true&access_token=' + mapboxgl.accessToken;
        $.ajax({
            method: 'GET',
            url: query
        }).done(function (data) {
            map.getSource('iso').setData(data);
        })
    };

    // function to add iso layer
    function addIso() {
        map.addSource('iso', {
            type: 'geojson',
            data: {
                "type": 'FeatureCollection',
                "features": []
            }
        });
        map.addLayer({
            'id': 'isoLayer',
            'type': 'fill',
            'source': 'iso',
            'layout': {},
            'paint': {
                'fill-color': colors[0],
                'fill-opacity': 0.3
            }
        }, "poi-label");
    }

    // function to add county boundaries
    function addCounties() {
        var url = 'https://raw.githubusercontent.com/kcavagnolo/knx-covid/master/data/geojson/tn-counties.geojson';
        fetch(url)
            .then(function (response) {
                if (!response.ok) {
                    throw Error(response.statusText);
                }
                return response.json();
            })
            .then(function (countyData) {
                //console.log(countyData);
                map.addSource('tn-county', {
                    'type': 'geojson',
                    'data': countyData,
                    'generateId': true
                });
                map.addLayer({
                    'id': 'tn-county-boundaries',
                    'type': 'line',
                    'source': 'tn-county',
                    'paint': {
                        'line-color': colors[1],
                        'line-width': 1
                    },
                    'filter': ['==', '$type', 'Polygon']
                });
                map.addLayer({
                    'id': 'tn-county-fills',
                    'type': 'fill',
                    'source': 'tn-county',
                    'paint': {
                        'fill-color': colors[1],
                        'fill-opacity': 0.0
                    },
                    'filter': ['==', '$type', 'Polygon']
                });
                map.on('click', 'tn-county-fills', function (e) {
                    popup.setLngLat(e.lngLat)
                        .setHTML(e.features[0].properties.NAME)
                        .addTo(map);
                });
                map.on('mouseenter', 'tn-county-fills', function () {
                    map.getCanvas().style.cursor = 'pointer';
                    popup.remove();
                });
                map.on('mouseleave', 'tn-county-fills', function () {
                    map.getCanvas().style.cursor = '';
                    popup.remove();
                });
            })
            .catch(function (error) {
                console.log('Looks like there was a problem: \n', error);
            });
    }

    // function to add county boundaries
    function addHrr() {
        var url = 'https://raw.githubusercontent.com/kcavagnolo/knx-covid/master/data/geojson/hrr.geojson';
        fetch(url)
            .then(function (response) {
                if (!response.ok) {
                    throw Error(response.statusText);
                }
                return response.json();
            })
            .then(function (hrrData) {
                //console.log(hrrData);
                map.addSource('hrr', {
                    'type': 'geojson',
                    'data': hrrData,
                    'generateId': true
                });
                map.addLayer({
                    'id': 'hrr-fills',
                    'type': 'line',
                    'source': 'hrr',
                    'paint': {
                        'line-color': colors[2],
                        'line-width': 1,
                        'line-dasharray': [5, 5]
                    },
                    'filter': ['==', '$type', 'Polygon']
                });
            })
            .catch(function (error) {
                console.log('Looks like there was a problem: \n', error);
            });
    }

    // function to add county boundaries
    function addHospitals() {
        var url = 'https://raw.githubusercontent.com/kcavagnolo/knx-covid/master/data/geojson/tn-hospitals.geojson';
        fetch(url)
            .then(function (response) {
                if (!response.ok) {
                    throw Error(response.statusText);
                }
                return response.json();
            })
            .then(function (hospitalData) {
                //console.log(hospitalData);
                map.addSource('hospitals', {
                    'type': 'geojson',
                    'data': hospitalData,
                    'generateId': true
                });
                map.addLayer({
                    'id': 'hospitals-points',
                    'type': 'circle',
                    'source': 'hospitals',
                    'layout': {
                        'visibility': 'visible',
                    },
                    'paint': {
                        'circle-radius': [
                            "interpolate",
                            ["linear"],
                            ["zoom"],
                            12, 4,
                            22, 10
                        ],
                        'circle-color': [
                            'case',
                            ['boolean', ['feature-state', 'hover'], false],
                            "#ffffff",
                            colors[3]
                        ],
                        'circle-opacity': [
                            'case',
                            ['boolean', ['feature-state', 'hover'], false],
                            0.8,
                            1.0
                        ],
                        'circle-stroke-color': '#ffffff',
                        'circle-stroke-width': 0.5
                    }
                });
                // hovering stops opens popup and changes symbol
                map.on('mouseenter', 'hospitals-points', function (e) {

                    // change cursor to indicator
                    map.getCanvas().style.cursor = 'pointer';

                    // set the html description
                    var coordinates = e.features[0].geometry.coordinates.slice();
                    var properties = e.features[0].properties;
                    var description = '<h3>' + e.features[0].properties.HospitalName + '</h3><p>' +
                        properties.Address1 + '<br>' +
                        properties.City + ',' + properties.State + ' ' + properties.PostalCode +
                        '<br>' +
                        'Ebola TTX? ' + properties.ebola_ttx +
                        '</p>'
                    while (Math.abs(e.lngLat.lng - coordinates[0]) > 180) {
                        coordinates[0] += e.lngLat.lng > coordinates[0] ? 360 : -360;
                    }

                    // load the hover popup
                    popup.setLngLat(coordinates)
                        .setHTML(description)
                        .addTo(map);

                    // change the symbol color and opacity
                    if (e.features.length > 0) {
                        if (hoverId) {
                            map.setFeatureState({
                                source: 'hospitals',
                                id: hoverId
                            }, {
                                hover: false
                            });
                        }
                        hoverId = e.features[0].id;
                        map.setFeatureState({
                            source: 'hospitals',
                            id: hoverId
                        }, {
                            hover: true
                        });
                    }
                });

                // no hover reverts state
                map.on('mouseleave', 'hospitals-points', function () {
                    if (hoverId) {
                        map.setFeatureState({
                            source: 'hospitals',
                            id: hoverId
                        }, {
                            hover: false
                        });
                    }
                    hoverId = null;

                    // change cursor back to pointer
                    map.getCanvas().style.cursor = '';

                    // remove the hover popup
                    popup.remove();
                });

            })
            .catch(function (error) {
                console.log('Looks like there was a problem: \n', error);
            });
    }

    // load map
    map.on('load', function () {

        // add iso
        addIso();

        // add counties
        addCounties();

        // add hospitals
        addHospitals();

        // add HRR
        addHrr();

        // add isochrone
        getIso();
    });

    // change drive time and call isochrone API
    var params = document.getElementById('params');
    params.addEventListener('change', function (e) {
        if (e.target.name === 'profile') {
            profile = e.target.value;
            getIso();
        } else if (e.target.name === 'duration') {
            minutes = e.target.value;
            getIso();
        }
    });
}