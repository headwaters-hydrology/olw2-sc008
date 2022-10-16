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
        }
    }
});