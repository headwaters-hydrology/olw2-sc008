window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
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
        rivers_catch_style_handle: function style(feature) {
            return {
                fillColor: 'grey',
                weight: 2,
                opacity: 1,
                color: 'black',
                fillOpacity: 0.1
            };
        }
    }
});