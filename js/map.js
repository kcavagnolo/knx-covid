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
    var volOrange = '#FF8200';
    var volWhite = '#FFFFFF';
    var volGray = '#58595B';
    var hoverId = null;
    var popup = new mapboxgl.Popup({
        closeButton: false,
        closeOnClick: false,
        offset: {
            "top": [0, 10],
            "bottom": [0, -10]
        }
    });
    var marker = new mapboxgl.Marker({
        'color': volOrange
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
                'fill-color': volOrange,
                'fill-opacity': 0.3
            }
        }, "poi-label");
    }

    // function to add county boundaries
    function addCounties() {
        var url = 'https://raw.githubusercontent.com/kcavagnolo/knx-covid/master/data/geojson/tn_county.geojson';
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
                        'line-color': volWhite,
                        'line-width': 1
                    },
                    'filter': ['==', '$type', 'Polygon']
                });
                map.addLayer({
                    'id': 'tn-county-fills',
                    'type': 'fill',
                    'source': 'tn-county',
                    'paint': {
                        'fill-color': volWhite,
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
                    'type': 'fill',
                    'source': 'hrr',
                    'paint': {
                        'fill-color': volGray,
                        'fill-opacity': 0.1
                    },
                    'filter': ['==', '$type', 'Polygon']
                });
            })
            .catch(function (error) {
                console.log('Looks like there was a problem: \n', error);
            });
    }

    // load map
    map.on('load', function () {

        // Initialize the marker at the query coordinates
        marker.setLngLat(origin).addTo(map);

        // add iso
        addIso();

        // add counties
        addCounties();

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