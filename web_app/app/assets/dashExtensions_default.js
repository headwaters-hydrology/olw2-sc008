window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        rivers_catch_style_handle: function style(feature) {
            return {
                fillColor: 'grey',
                weight: 2,
                opacity: 1,
                color: 'black',
                fillOpacity: 0.1
            };
        },
        rivers_base_reach_style_handle: function style3(feature) {
            return {
                weight: 2,
                opacity: 0.75,
                color: 'grey',
            };
        },
        rivers_reach_style_handle: function style2(feature, context) {
            const {
                classes,
                colorscale,
                style,
                colorProp
            } = context.props.hideout; // get props from hideout
            const value = feature.properties[colorProp]; // get value the determines the color
            for (let i = 0; i < classes.length; ++i) {
                if (value == classes[i]) {
                    style.color = colorscale[i]; // set the fill color according to the class
                }
            }
            return style;
        },
        sites_points_handle: function style_sites(feature, latlng, context) {
            const {
                circleOptions
            } = context.props.hideout;
            return L.circleMarker(latlng, circleOptions);
        },
        lakes_lake_style_handle: function style4(feature, context) {
            const {
                classes,
                colorscale,
                style,
                colorProp
            } = context.props.hideout; // get props from hideout
            const value = feature.properties[colorProp]; // get value the determines the color
            for (let i = 0; i < classes.length; ++i) {
                if (value == classes[i]) {
                    style.color = colorscale[i]; // set the color according to the class
                }
            }

            return style;
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