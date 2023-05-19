window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        gw_base_rc_style_handle: function style3(feature) {
            return {
                weight: 2,
                opacity: 0.75,
                color: 'grey',
            };
        },
        gw_rc_style_handle: function style(feature) {
            return {
                fillColor: 'grey',
                weight: 2,
                opacity: 1,
                color: 'black',
                fillOpacity: 0.1
            };
        },
        gw_points_style_handle: function gw_points_style_handle(feature, latlng, context) {
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