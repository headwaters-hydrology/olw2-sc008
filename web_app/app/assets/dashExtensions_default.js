window.dashExtensions = Object.assign({}, window.dashExtensions, {
    default: {
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
        lakes_catch_style_handle: function style(feature) {
            return {
                fillColor: 'grey',
                weight: 2,
                opacity: 1,
                color: 'black',
                fillOpacity: 0.1
            };
        },
        lakes_sites_points_handle: function lakes_sites_points_handle(feature, latlng, context) {
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
        lakes_marae_handle: function(feature, latlng) {
            const flag = L.icon({
                iconUrl: '/assets/nzta-marae.svg',
                iconSize: [20, 30]
            });
            return L.marker(latlng, {
                icon: flag
            });
        },
        lakes_lake_style_handle_sites: function style4(feature, context) {
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
        lakes_sites_points_handle_sites: function lakes_sites_points_handle(feature, latlng, context) {
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
        rivers_lc_style_handle: function style2(feature, context) {
            const {
                classes,
                colorscale,
                style,
                colorProp
            } = context.props.hideout; // get props from hideout
            const value = feature.properties[colorProp]; // get value the determines the color
            for (let i = 0; i < classes.length; ++i) {
                if (value >= classes[i]) {
                    style.fillColor = colorscale[i]; // set the fill color according to the class
                }
            }
            return style;
        },
        rivers_lc_points_handle: function rivers_sites_points_handle(feature, latlng, context) {
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
        rivers_lc_marae_handle: function(feature, latlng) {
            const flag = L.icon({
                iconUrl: '/assets/nzta-marae.svg',
                iconSize: [20, 30]
            });
            return L.marker(latlng, {
                icon: flag
            });
        },
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
        eco_reach_style_handle: function style2(feature, context) {
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
        eco_marae_handle: function(feature, latlng) {
            const flag = L.icon({
                iconUrl: '/assets/nzta-marae.svg',
                iconSize: [20, 30]
            });
            return L.marker(latlng, {
                icon: flag
            });
        },
        eco_sites_marae_handle: function(feature, latlng) {
            const flag = L.icon({
                iconUrl: '/assets/nzta-marae.svg',
                iconSize: [20, 30]
            });
            return L.marker(latlng, {
                icon: flag
            });
        },
        rivers_catch_style_handle_hfl: function style(feature) {
            return {
                fillColor: 'grey',
                weight: 2,
                opacity: 1,
                color: 'black',
                fillOpacity: 0.1
            };
        },
        rivers_base_reach_style_handle_hfl: function style3(feature) {
            return {
                weight: 2,
                opacity: 0.75,
                color: 'grey',
            };
        },
        rivers_reach_style_handle_hfl: function style2(feature, context) {
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
        rivers_sites_points_handle_hfl: function rivers_sites_points_handle(feature, latlng, context) {
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
        rivers_marae_handle_hfl: function(feature, latlng) {
            const flag = L.icon({
                iconUrl: '/assets/nzta-marae.svg',
                iconSize: [20, 30]
            });
            return L.marker(latlng, {
                icon: flag
            });
        },
        rivers_sites_points_handle: function rivers_sites_points_handle(feature, latlng, context) {
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
        rivers_marae_handle: function(feature, latlng) {
            const flag = L.icon({
                iconUrl: '/assets/nzta-marae.svg',
                iconSize: [20, 30]
            });
            return L.marker(latlng, {
                icon: flag
            });
        },
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
        },
        rivers_sites_marae_handle: function(feature, latlng) {
            const flag = L.icon({
                iconUrl: '/assets/nzta-marae.svg',
                iconSize: [20, 30]
            });
            return L.marker(latlng, {
                icon: flag
            });
        },
        rivers_catch_style_handle_sites_change: function style(feature) {
            return {
                fillColor: 'grey',
                weight: 2,
                opacity: 1,
                color: 'black',
                fillOpacity: 0.1
            };
        },
        rivers_base_reach_style_handle_sites_change: function style3(feature) {
            return {
                weight: 2,
                opacity: 0.75,
                color: 'grey',
            };
        },
        rivers_sites_points_handle_sites_change: function rivers_sites_points_handle(feature, latlng, context) {
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
        rivers_sites_change_marae_handle: function(feature, latlng) {
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