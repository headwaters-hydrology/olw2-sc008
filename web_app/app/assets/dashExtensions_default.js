window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
        function0: function style(feature) {
            return {
                fillColor: 'grey',
                weight: 2,
                opacity: 1,
                color: 'black',
                fillOpacity: 0.1
            };
        },
        function1: function style3(feature) {
            return {
                weight: 2,
                opacity: 0.75,
                color: 'grey',
            };
        },
        function2: function style2(feature, context) {
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
        function3: function style3(feature) {
            return {
                weight: 2,
                opacity: 0.75,
                color: 'grey',
            };
        },
        function4: function style2(feature, context) {
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
        }
    }
});