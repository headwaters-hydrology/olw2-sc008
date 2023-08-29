window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        rivers_catch_style_handle_sites: function style(feature) {
            return {
                fillColor: 'grey',
                weight: 2,
                opacity: 1,
                color: 'black',
                fillOpacity: 0.1
            };
        },
        rivers_base_reach_style_handle_sites: function style3(feature) {
            return {
                weight: 2,
                opacity: 0.75,
                color: 'grey',
            };
        },
        rivers_sites_points_handle_sites: function rivers_sites_points_handle(feature, latlng, context) {
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
        }
    }
});