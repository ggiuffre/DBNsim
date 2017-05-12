var index = 0;
    var chart = undefined;
    var job_id = undefined;

    function newTraining() {
        $("#train_form").submit(function(e) {
            $.ajax({
                type: 'POST',
                url: 'train/',
                data: $('#train_form').serialize(),
                success: function(response) { job_id = response; }
            });
            setupChart();
            e.preventDefault(); // do not submit the form
        });
    }

    function setupChart() {
        index = 0;
        chart = Highcharts.chart({
            chart: {
                renderTo: 'container',
                defaultSeriesType: 'areaspline',
                animation: {
                    duration: 200
                }
            },
            title: {
                text: 'Reconstruction error over time'
            },
            xAxis: {
                min: 0,
                title: {
                    text: 'Epoch number'
                },
                tickInterval: 1,
                maxZoom: 20
            },
            yAxis: {
                min: 0,
                max: 1, // TODO corretto?
                title: {
                    text: 'Error',
                    margin: 80
                }
            },
            series: [{
                name: 'Training error',
                data: []
            }]
        });
    }

    function updateError(autoContinue = false) {
        var parameters = { 'job_id': job_id };
        $.ajax({
            type: 'POST',
            url: 'getError/',
            data: JSON.stringify(parameters),
            contentType: 'application/json; charset=utf-8',
            dataType: 'json',
            success: function(response) {
                if (!response.stop) {
                    var point = response.error;

                    var shift = chart.series[0].data.length > 20; // shift if the series is longer than 20
                    chart.series[0].addPoint([index, point], true, shift);
                    index++;

                    if (autoContinue)
                        setTimeout(updateError, 500); // call it again
                }
            }
        });
    }
