(function(window, document, $, undefined) {
    "use strict";
    $(function() {

        if ($('#c3chart_area').length) {
            var chart = c3.generate({
                bindto: "#c3chart_area",
                data: {
                    columns: [
                        ['sensor 0', 1, 143, 243, 80, 188, 192, 162, 155, 105, 128, 61, 131, 102, 222, 35, 245, 42, 116, 121, 179, 240, 29, 111, 56, 84],
                        ['sensor 1', 118, 183, 159, 104, 25, 72, 115, 208, 88, 197, 239, 193, 163, 213, 128, 190, 3, 67, 215, 51, 239, 88, 42, 90, 159],
                        ['sensor 2', 67, 91, 215, 162, 75, 9, 215, 89, 21, 4, 19, 71, 212, 63, 67, 160, 70, 175, 142, 119, 196, 219, 66, 196, 44],
                        ['sensor 3', 215, 239, 23, 229, 91, 101, 5, 4, 71, 229, 18, 114, 122, 171, 134, 242, 184, 60, 215, 130, 103, 147, 128, 70, 47],
                        ['sensor 4', 216, 3, 244, 196, 216, 213, 45, 185, 196, 163, 220, 23, 166, 145, 54, 208, 232, 193, 24, 119, 132, 222, 176, 6, 113],
                        ['sensor 5', 48, 232, 1, 29, 121, 71, 139, 219, 103, 31, 106, 107, 248, 157, 61, 160, 123, 126, 17, 183, 149, 82, 111, 242, 118],
                        ['sensor 6', 217, 22, 121, 234, 189, 127, 130, 18, 31, 101, 202, 19, 231, 25, 103, 55, 106, 38, 182, 47, 42, 179, 239, 61, 165]
                    ],
                    types: {
                        sensor0: 'area',
                        sensor1: 'area',
                        sensor2: 'area-spline',
                        sensor3: 'area',
                        sensor4: 'area-spline',
                        sensor5: 'area',
                        sensor6: 'area-spline'
                    }

                },
                axis: {

                    y: {
                        show: true




                    },
                    x: {
                        show: true
                    }
                }

            });
        }


        if ($('#c3chart_spline').length) {
            var chart = c3.generate({
                bindto: "#c3chart_spline",
                data: {
                    columns: [
                        ['left',  1, 143, 243, 80, 188, 192, 162, 155, 105, 128, 61, 131, 102, 222, 35, 245, 42, 116, 121, 179, 240, 29, 111, 56, 84],
                        ['right', 217, 22, 121, 234, 189, 127, 130, 18, 31, 101, 202, 19, 231, 25, 103, 55, 106, 38, 182, 47, 42, 179, 239, 61, 165]
                    ],
                    type: 'spline',
                    colors: {
                         data1: '#5969ff',
                        data2: '#ff407b',

                    }
                },
                axis: {
                    y: {
                        show: true,


                    },
                    x: {
                        show: true,
                    }
                }
            });
        }

        if ($('#c3chart_zoom').length) {
            var chart = c3.generate({
                bindto: "#c3chart_zoom",
                data: {
                    columns: [
                        ['sample', 215, 239, 23, 229, 91, 101, 5, 4, 71, 229, 18, 114, 122, 171, 134, 242, 184, 60, 215, 130, 103, 147, 128, 70, 47]
                    ],
                    colors: {
                        sample: '#5969ff'


                    }
                },
                zoom: {
                    enabled: true
                },
                axis: {
                    y: {
                        show: true,


                    },
                    x: {
                        show: true,
                    }
                }

            });
        }


        if ($('#c3chart_scatter').length) {
            var chart = c3.generate({
                bindto: "#c3chart_scatter",
                data: {
                    xs: {
                        setosa: 'setosa_x',
                        versicolor: 'versicolor_x',
                    },
                    // iris data from R
                    columns: [
                        ["setosa_x", 3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1, 3.7, 3.4, 3.0, 3.0, 4.0, 4.4, 3.9, 3.5, 3.8, 3.8, 3.4, 3.7, 3.6, 3.3, 3.4, 3.0, 3.4, 3.5, 3.4, 3.2, 3.1, 3.4, 4.1, 4.2, 3.1, 3.2, 3.5, 3.6, 3.0, 3.4, 3.5, 2.3, 3.2, 3.5, 3.8, 3.0, 3.8, 3.2, 3.7, 3.3],
                        ["versicolor_x", 3.2, 3.2, 3.1, 2.3, 2.8, 2.8, 3.3, 2.4, 2.9, 2.7, 2.0, 3.0, 2.2, 2.9, 2.9, 3.1, 3.0, 2.7, 2.2, 2.5, 3.2, 2.8, 2.5, 2.8, 2.9, 3.0, 2.8, 3.0, 2.9, 2.6, 2.4, 2.4, 2.7, 2.7, 3.0, 3.4, 3.1, 2.3, 3.0, 2.5, 2.6, 3.0, 2.6, 2.3, 2.7, 3.0, 2.9, 2.9, 2.5, 2.8],
                        ["setosa", 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2],
                        ["versicolor", 1.4, 1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1.0, 1.3, 1.4, 1.0, 1.5, 1.0, 1.4, 1.3, 1.4, 1.5, 1.0, 1.5, 1.1, 1.8, 1.3, 1.5, 1.2, 1.3, 1.4, 1.4, 1.7, 1.5, 1.0, 1.1, 1.0, 1.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3, 1.3, 1.2, 1.4, 1.2, 1.0, 1.3, 1.2, 1.3, 1.3, 1.1, 1.3],
                    ],
                    type: 'scatter',
                    colors: {
                        setosa: '#5969ff',
                        versicolor: '#ff407b',

                    }
                },
                axis: {
                    y: {
                        show: true,


                    },
                    x: {
                        show: true,
                    }
                }
            });
            setTimeout(function() {
                chart.load({
                    xs: {
                        virginica: 'virginica_x'
                    },
                    columns: [
                        ["virginica_x", 3.3, 2.7, 3.0, 2.9, 3.0, 3.0, 2.5, 2.9, 2.5, 3.6, 3.2, 2.7, 3.0, 2.5, 2.8, 3.2, 3.0, 3.8, 2.6, 2.2, 3.2, 2.8, 2.8, 2.7, 3.3, 3.2, 2.8, 3.0, 2.8, 3.0, 2.8, 3.8, 2.8, 2.8, 2.6, 3.0, 3.4, 3.1, 3.0, 3.1, 3.1, 3.1, 2.7, 3.2, 3.3, 3.0, 2.5, 3.0, 3.4, 3.0],
                        ["virginica", 2.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.7, 1.8, 1.8, 2.5, 2.0, 1.9, 2.1, 2.0, 2.4, 2.3, 1.8, 2.2, 2.3, 1.5, 2.3, 2.0, 2.0, 1.8, 2.1, 1.8, 1.8, 1.8, 2.1, 1.6, 1.9, 2.0, 2.2, 1.5, 1.4, 2.3, 2.4, 1.8, 1.8, 2.1, 2.4, 2.3, 1.9, 2.3, 2.5, 2.3, 1.9, 2.0, 2.3, 1.8],
                    ]
                });
            }, 1000);
            setTimeout(function() {
                chart.unload({
                    ids: 'setosa'
                });
            }, 2000);
            setTimeout(function() {
                chart.load({
                    columns: [
                        ["virginica", 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2],
                    ]
                });
            }, 3000);
        }


        if ($('#c3chart_stacked').length) {
            var chart = c3.generate({
                bindto: "#c3chart_stacked",

                data: {
                    columns: [
                        ['Correct', 130, 200, 320, 400, 530, 750],
                        ['Error', -130, 10, 130, 200, 150, 250],
                    ],
                    type: 'bar',
                    groups: [
                        ['Correct', 'Error']
                    ],
                    order: 'desc', // stack order by sum of values descendantly. this is default.
                    //      order: 'asc'  // stack order by sum of values ascendantly.
                    //      order: null   // stack order by data definition.

                    colors: {
                        data1: '#5969ff',
                        data2: '#ff407b',
                    }
                },
                axis: {
                    y: {
                        show: true,


                    },
                    x: {
                        show: true,
                    }
                },
                grid: {
                    y: {
                        lines: [{ value: 0 }]
                    }
                }
            });
            setTimeout(function() {
                chart.load({
                    columns: [
                        ['data4', 1200, 1300, 1450, 1600, 1520, 1820],
                    ]
                });
            }, 1000);
            setTimeout(function() {
                chart.load({
                    columns: [
                        ['data5', 200, 300, 450, 600, 520, 820],
                    ]
                });
            }, 2000);
            setTimeout(function() {
                chart.groups([
                    ['data1', 'data2', 'data3', 'data4', 'data5']
                ])
            }, 3000);
        }


        if ($('#c3chart_combine').length) {
            var chart = c3.generate({
                bindto: "#c3chart_combine",
                data: {
                    columns: [
                        ['data1', 30, 20, 50, 40, 60, 50],
                        ['data2', 200, 130, 90, 240, 130, 220],
                        ['data3', 300, 200, 160, 400, 250, 250],
                        ['data4', 200, 130, 90, 240, 130, 220],
                        ['data5', 130, 120, 150, 140, 160, 150],
                        ['data6', 90, 70, 20, 50, 60, 120],
                    ],
                    type: 'bar',
                    types: {
                        data3: 'spline',
                        data4: 'line',
                        data6: 'area',
                    },
                    groups: [
                        ['data1', 'data2']
                    ],

                    colors: {
                        data1: '#5969ff',
                        data2: '#ff407b',
                        data3: '#25d5f2',
                        data4: '#ffc750',
                        data5: '#2ec551',
                        data6: '#1ba3b9',


                    }

                },
                axis: {
                    y: {
                        show: true,


                    },
                    x: {
                        show: true,
                    }
                }
            });
        }

        if ($('#c3chart_pie').length) {
            var chart = c3.generate({
                bindto: "#c3chart_pie",
                data: {
                    columns: [
                        ['data1', 30],
                        ['data2', 50]
                    ],
                    type: 'pie',

                    colors: {
                         data1: '#5969ff',
                        data2: '#ff407b'


                    }
                },
                pie: {
                    label: {
                        format: function(value, ratio, id) {
                            return d3.format('$')(value);
                        }
                    }
                }
            });
        }

        if ($('#c3chart_donut').length) {
            var chart = c3.generate({
                bindto: "#c3chart_donut",
                data: {
                    columns: [
                        ['data1', 30],
                        ['data2', 120],
                    ],
                    type: 'donut',
                    onclick: function(d, i) { console.log("onclick", d, i); },
                    onmouseover: function(d, i) { console.log("onmouseover", d, i); },
                    onmouseout: function(d, i) { console.log("onmouseout", d, i); },

                    colors: {
                         data1: '#5969ff',
                        data2: '#ff407b'


                    }
                },
                donut: {
                    title: "Iris Petal Width"


                }


            });

            setTimeout(function() {
                chart.load({
                    columns: [
                        ["setosa", 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.4, 0.3, 0.3, 0.3, 0.2, 0.4, 0.2, 0.5, 0.2, 0.2, 0.4, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2, 0.2, 0.3, 0.3, 0.2, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2],
                        ["versicolor", 1.4, 1.5, 1.5, 1.3, 1.5, 1.3, 1.6, 1.0, 1.3, 1.4, 1.0, 1.5, 1.0, 1.4, 1.3, 1.4, 1.5, 1.0, 1.5, 1.1, 1.8, 1.3, 1.5, 1.2, 1.3, 1.4, 1.4, 1.7, 1.5, 1.0, 1.1, 1.0, 1.2, 1.6, 1.5, 1.6, 1.5, 1.3, 1.3, 1.3, 1.2, 1.4, 1.2, 1.0, 1.3, 1.2, 1.3, 1.3, 1.1, 1.3],
                        ["virginica", 2.5, 1.9, 2.1, 1.8, 2.2, 2.1, 1.7, 1.8, 1.8, 2.5, 2.0, 1.9, 2.1, 2.0, 2.4, 2.3, 1.8, 2.2, 2.3, 1.5, 2.3, 2.0, 2.0, 1.8, 2.1, 1.8, 1.8, 1.8, 2.1, 1.6, 1.9, 2.0, 2.2, 1.5, 1.4, 2.3, 2.4, 1.8, 1.8, 2.1, 2.4, 2.3, 1.9, 2.3, 2.5, 2.3, 1.9, 2.0, 2.3, 1.8],
                    ]
                });
            }, 1500);

            setTimeout(function() {
                chart.unload({
                    ids: 'data1'
                });
                chart.unload({
                    ids: 'data2'
                });
            }, 2500);
        }

        if ($('#c3chart_gauge').length) {
            var chart = c3.generate({
                bindto: "#c3chart_gauge",
                data: {
                    columns: [
                        ['data1', 91.4]

                    ],
                    type: 'gauge',
                    onclick: function(d, i) { console.log("onclick", d, i); },
                    onmouseover: function(d, i) { console.log("onmouseover", d, i); },
                    onmouseout: function(d, i) { console.log("onmouseout", d, i); },
                    colors: {
                      data1: '#5969ff',
                        data2: '#ff407b',
                        data3: '#25d5f2',
                        data4: '#ffc750',
                        data5: '#2ec551',
                        data6: '#1ba3b9',

                    }
                },
                gauge: {
                    //        label: {
                    //            format: function(value, ratio) {
                    //                return value;
                    //            },
                    //            show: false // to turn off the min/max labels.
                    //        },
                    //    min: 0, // 0 is default, //can handle negative min e.g. vacuum / voltage / current flow / rate of change
                    //    max: 100, // 100 is default
                    //    units: ' %',
                    //    width: 39 // for adjusting arc thickness
                },

                size: {
                    height: 320
                }
            });



        }


    });

})(window, document, window.jQuery);