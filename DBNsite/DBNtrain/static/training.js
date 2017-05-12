var epoch  = 0;
var job_id = undefined;
var chart  = undefined;

/**
 * Sets up the chart settings.
 * To be called _after_ the page has loaded!
 */
function setupChart() {
	chart = Highcharts.chart({
		chart: {
			renderTo: 'train_plot_container',
			defaultSeriesType: 'spline',
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
				text: 'Mean unit error',
				margin: 80
			}
		},
		series: [{
			name: 'Training error layer 1',
			data: []
		}]
	});
}

/**
 * Submits the training hyper-parameters to the
 * server for starting a new training session.
 */
function newTraining() {
	$("#train_form").submit(function(e) {
		$.ajax({
			type: 'POST',
			url: 'train/',
			data: $('#train_form').serialize(),
			success: function(response) { job_id = response; }
		});
		e.preventDefault(); // do not submit the form
	});
}

/**
 * Asks the server to train the network for
 * one epoch, then updates the reconstruction
 * error.
 * @param  {Boolean} autoContinue  whether to automate the update
 */
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
				chart.series[0].addPoint([epoch, point], true, shift);
				epoch++;

				if (autoContinue)
					setTimeout(updateError.bind(this, autoContinue), 500); // call it again
			}
		}
	});
}
