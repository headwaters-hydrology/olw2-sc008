window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        eco_catch_style_handle: function style(feature) {
            return {
                fillColor: 'grey',
                weight: 2,
                opacity: 1,
                color: 'black',
                fillOpacity: 0.1
            };
        },
        eco_base_reach_style_handle: function style3(feature) {
            return {
                weight: 2,
                opacity: 0.75,
                color: 'grey',
            };
        },
        eco_sites_points_handle: function rivers_sites_points_handle(feature, latlng, context) {
            const {
                classes,
                colorscale,
                circleOptions,
                colorProp
            } = context.props.hideout; // get props from hideout
            const value = feature.properties[colorProp]; // get value the determines the fillColor
            for (let i = 0; i < classes.length; ++i) {
                if (value == classes[i]) {
                    circleOptions.fillColor = colorscale[i]; // set the color according to the class
                }
            }

            return L.circleMarker(latlng, circleOptions);
        },
        eco_sites_marae_handle: function(feature, latlng) {
            const flag = L.icon({
                iconUrl: '/assets/nzta-marae.svg',
                iconSize: [20, 30]
            });
            return L.marker(latlng, {
                icon: flag
            });
        }
    }
});