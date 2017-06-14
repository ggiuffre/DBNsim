/**
 * The current training epoch number.
 * @type {Number}
 */
let curr_epoch;

/**
 * The currently training RBM number index.
 * @type {Number}
 */
let curr_rbm;

/**
 * A unique identifier for the training job on the server.
 * @type {String}
 */
let job_id;

/**
 * The DBN graph.
 * @type {Cytoscape}
 */
let networkGraph;

/**
 * The chart for plotting the reconstruction
 * error over time, while training the net.
 * @type {Highcharts.Chart}
 */
let chart;





/**
 * Updates the architecture form after after the `tab` key
 * is pressed but _before_ its default behaviour is applied.
 */
$(function() {
	$('#net_form').keydown(function (e) {
		if (e.which == 8) updateArchitecture();
	});
});

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
		credits: {
			enabled: false
		},
		xAxis: {
			min: 1,
			max: $('#epochs').val(),
			title: {
				text: 'Epoch number'
			},
			tickInterval: 1
		},
		yAxis: {
			min: 0,
			max: 1,
			title: {
				text: 'Mean unit error',
				margin: 80
			}
		},
		series: [{
			name: 'RBM 1',
			data: []
		}]
	});
}

/**
 * Binds the submission of the training form to an AJAX request
 * that submits the training hyper-parameters to the server.
 */
function setupTrainForm() {
	$('#train_form').submit(function(e) {

		// reset counters and chart series:
		curr_epoch = 0;
		curr_rbm = -1;
		updateSeries();

		const epochs = $('#epochs').val();
		const num_layers = +$('#num_hid_layers').val() + 1;
		chart.xAxis[0].setExtremes(1, epochs);
		for (let i = 1; i < num_layers; i++)
			networkGraph.$('.rbm' + i).style('line-color', '#ACD');
		networkGraph.$('.rbm1').style('line-color', '#D89');

		const net_form_data = $('#net_form').serialize();
		const train_form_data = $('#train_form').serialize();
		let forms_data = net_form_data + '&' + train_form_data;
		forms_data += '&last_job_id=' + job_id;

		$.ajax({
			type: 'POST',
			url: 'train/',
			data: forms_data,
			success: function(response) {
				job_id = response;
				retrieveError(true);
			}
		});

		e.preventDefault(); // do not submit the form
	});
}





/**
 * Updates the field for defining the visible size,
 * based on the size of the training examples.
 */
function updateVisibleSize() {
	const curr_dataset = $('#dataset').val();
	const input_size = data_info[curr_dataset];
	$('#vis_sz').val(input_size);
}

/**
 * Updates the form for defining the DBN layers, based on
 * the number of layers that the user wants to create.
 */
function updateArchitecture() {
	const num_hid_layers = +$('#num_hid_layers').val();
	if (num_hid_layers === '')
		return;
	if (num_hid_layers < 1) {
		alert('You must have at least one hidden layer: defaulting to 1.');
		$('#num_hid_layers').val(1);
		num_hid_layers = 1;
	}
	if (num_hid_layers > 5) {
		alert('Too many hidden layers (' + num_hid_layers + '): defaulting to 5.');
		$('#num_hid_layers').val(5);
		num_hid_layers = 5;
	}

	// how many HTML inputs do we already have?
	const curr_layers = +$('.lay_sz').length;

	// add missing hidden layers:
	for (let i = curr_layers; i < num_hid_layers + 1; i++)
		$('#layers_sz').append(
			'<li class="lay_sz">' +
			'<label for="hid_sz_' + i + '">h' + i + '&nbsp;&nbsp;</label>' +
			'<input type="text" name="hid_sz_' + i + '" id="hid_sz_' + i + '" onchange="updateGraph();" />' +
			'</li>');

	// remove exceeding hidden layers:
	for (let i = curr_layers; i > num_hid_layers + 1; i--)
		$('.lay_sz').last().remove();
}

/**
 * Updates the time series for the error plot, matching
 * the number of RBMs defined in the architecture form.
 */
function updateSeries() {
	const num_layers = +$('#num_hid_layers').val() + 1;
	const num_rbms = num_layers - 1;

	// remove all the series:
	for (let i = chart.series.length - 1; i >= 0; i--)
		chart.series[i].remove();

	// add the necessary series:
	for (let i = 0; i < num_rbms; i++)
		chart.addSeries({
			name: 'RBM ' + (i + 1),
			data: []
		});
}

/**
 * Repaints the Cytoscape graph, checking
 * for changes in the HTML form.
 */
function updateGraph() {
	const num_layers = +$('#num_hid_layers').val() + 1;
	let prec_layer_nodes = 0;
	networkGraph = cytoscape({
		container: document.getElementById('DBNgraph')
	});

	// gather new nodes and edges:
	let graphElements = [];
	for (let layer = 0; layer < num_layers; layer++) {
		let label, selector;
		if (layer == 0) {
			label = 'Visible layer';
			selector = $('#vis_sz');
		} else {
			label = 'Hidden layer ' + layer;
			selector = $('#hid_sz_' + layer);
		}
		let num_nodes = +selector.val();

		if (num_nodes > 0) {
			const max_real_nodes = 10000;
			const max_rendered_nodes = 15;
			const edgeThickness = 1.3 + 2 * max_rendered_nodes / (prec_layer_nodes + num_nodes);

			if (num_nodes > max_real_nodes) {
				alert(label + ' has too many nodes (' + num_nodes + '): aborting; defaulting to ' + max_real_nodes + '.');
				selector.val(max_real_nodes);
				num_nodes = max_real_nodes;
			}

			graphElements.push({
				group: 'nodes',
				data: {
					id: label,
					layer: layer,
					placeholder: num_nodes + ' nodes'
				}
			});

			if (num_nodes > 10) {
				const scale_base = Math.pow(max_real_nodes, 1 / max_rendered_nodes);
				num_nodes = baseLog(num_nodes, scale_base) + 1;
			}

			for (let node = 1; node <= num_nodes; node++) {
				graphElements.push({
					group: 'nodes',
					data: {
						id: nodeId(layer, node),
						layer: layer,
						parent: label
					},
					position: { // X is horizontal, Y is vertical.
						x: ((num_nodes / 2) - node + 0.5) * networkGraph.width() / num_nodes,
						y: ((num_layers / 2) - layer + 0.5) * networkGraph.height() / num_layers
					}
				});

				for (let prec_node = 1; prec_node <= prec_layer_nodes; prec_node++) {
					const source_id = nodeId(layer, node);
					const target_id = nodeId(layer - 1, prec_node);
					graphElements.push({
						group: 'edges',
						classes: 'rbm' + layer,
						data: {
							id: edgeId(source_id, target_id),
							thickness: edgeThickness,
							rbm: layer - 1,
							source: source_id,
							target: target_id
						}
					});
				}
			}

			prec_layer_nodes = num_nodes;
		}
	}

	// build and render the graph:
	networkGraph = getGraphFrom(graphElements);

	// bind click events to the graph:
	networkGraph.$('node').on('tap', function(event) {
		const layer = event.target.data('layer');
		dissect(layer);
	});
	networkGraph.$('edge').on('tap', function(event) {
		const rbm = event.target.data('rbm');
		plotHistogram(rbm);
	});
}

/**
 * Returns a Cytoscape graph built from an array
 * of Cytoscape nodes and edges.
 * @param  {Array} elements  an array of nodes and edges
 * @return {Cytoscape}       a Cytoscape graph
 */
function getGraphFrom(elements) {
	return cytoscape({
		container: document.getElementById('DBNgraph'),
		elements: elements,
		userZoomingEnabled: false,
		style: [
			{
				selector: 'node',
				style: {
					'width': 10,
					'height': 10,
					'background-color': '#46A'
				}
			},
			{
				selector: ':parent',
				style: {
					'z-compound-depth': 'top',
					'background-color': '#FFF',
					'label': 'data(placeholder)',
					'text-valign': 'center',
					'text-opacity': 0.6
				}
			},
			{
				selector: 'edge',
				style: {
					'width': 'data(thickness)',
					'line-color': '#ACD'
				}
			},
			{
				selector: 'edge:active',
				style: {
					'overlay-color': '#ACD',
					'overlay-padding': 0
				}
			}
		],
		layout: {
			name: 'preset'
		}
	});
}

/**
 * Shows a random input image or the receptive
 * fields of one particular layer of the DBN.
 * @param {Number} layer  the layer position in the DBN
 */
function dissect(layer) {
	if (layer == 0) {
		const dataset = $('#dataset').val();
		const vis_sz = $('#vis_sz').val();
		$.ajax({
			url: 'getInput/',
			data: {dataset: dataset, index: -1},
			dataType: 'json',
			success: function(response) {
				$('#input_image').show();
				const title = 'Random input image from the "' + dataset + '" dataset.';
				heatmap('input_image', response, title);
				$('<p id="input_image_caption">' + title + '</p>').insertAfter('#input_image');
			}
		});
	} else {
		if (!job_id) return;
		$('#receptive_fields').empty();
		const neurons = $('#hid_sz_' + layer).val();
		const rec_fields_displayed = Math.min(neurons, 25);
		for (let i = 0; i < rec_fields_displayed; i++) {
			const rc_id = 'rec_field_' + i;
			$('#receptive_fields').append('<div id="' + rc_id + '" class="rec_field"></div>');
			$.ajax({
				url: 'getReceptiveField/',
				data: {
					job_id: job_id,
					layer: layer,
					neuron: Math.floor(i * neurons / rec_fields_displayed)
				},
				async: false,
				dataType: 'json',
				success: function(response) {
					$('#receptive_fields').show();
					heatmap(rc_id, response, null);
				}
			});
		}
	}
}

/**
 * Shows the histogram for the distribution of the
 * weights in a specific RBM of the DBN.
 * @param {Number} rbm  the RBM position in the DBN
 */
function plotHistogram(rbm) {
	if (!job_id) return;
	$.ajax({
		url: 'getHistogram/',
		data: {
			job_id: job_id,
			rbm: rbm
		},
		dataType: 'json',
		success: function(response) {
			$('#weights_histogram').show();
			Highcharts.chart('weights_histogram', {
				chart: {
					type: 'column'
				},
				title: {
					text: 'Histogram of the weights in RBM ' + (rbm + 1)
				},
				xAxis: {
					title: {
						text: 'Weights value'
					}
				},
				yAxis: {
					title: {
						text: 'Weights count'
					}
				},
				legend: { enabled: false },
				credits: { enabled: false },
				series: [{
					name: 'Weights count',
					data: response,
				}]
			});
		}
	});
}

/**
 * Renders and returns a Highcharts heatmap located
 * in a HTML div identified by `container`, containing
 * the given data and having the given title.
 * @param {String} container the id of the target HTML div
 * @param {Array} data       the heatmap data
 * @param {String} title     the heatmap title
 * @return {Highcharts}      a reference to the heatmap
 */
function heatmap(container, data, title) {
	return Highcharts.chart(container, {
		chart: {
			type: 'heatmap',
			borderWidth: 0
		},
		xAxis: {
			min: 0,
			max: Math.sqrt(data.length) - 1,
			visible: false
		},
		yAxis: {
			min: 0,
			max: Math.sqrt(data.length) - 1,
			visible: false
		},
		//tooltip: { enabled: false },
		title: { visible: false, text: null },
		legend: { enabled: false },
		credits: { enabled: false },
		exporting: { enabled: false },
		colorAxis: {
			minColor: '#FFF',
			maxColor: '#000'
		},
		series: [{
			name: 'Image plot',
			data: data,
			borderWidth: 1,
			borderColor: '#EEE'
		}]
	});
}

/**
 * Returns a standard string identifier
 * for the given node in the given layer.
 * @param  {Number} layer the layer position in the DBN
 * @param  {Number} node  the node position in the layer
 * @return {String}       a string identifier
 */
function nodeId(layer, node) {
	return 'l' + layer + 'n' + node;
}

/**
 * Returns a standard string identifier
 * for the given edge in the network graph.
 * @param  {Number} node1 the identifier for the first node
 * @param  {Number} node2 the identifier for the second node
 * @return {String}       a string identifier
 */
function edgeId(node1, node2) {
	return node1 + '_' + node2;
}

/**
 * Returns the logarithm of `x` base `base`.
 * @param  {Number} x    the argument of the logarithm
 * @param  {Number} base the base
 * @return {Number}      log_base(x)
 */
function baseLog(x, base) {
	return Math.log(x) / Math.log(base);
}





/**
 * Asks the server to train the network for one epoch,
 * then updates the reconstruction error on the chart.
 * @param {Boolean} autoContinue  whether to automate the training
 */
function retrieveError(autoContinue) {
	$('#train_plot_loading').css('opacity', 1);

	const parameters = { 'job_id': job_id };
	$.ajax({
		type: 'POST',
		url: 'getError/',
		data: JSON.stringify(parameters),
		dataType: 'json',
		success: function(response) {
			$('#train_plot_loading').css('opacity', 0);

			if (response.stop) {
				// announce training has ended for the last RBM:
				networkGraph.$('.rbm' + (curr_rbm + 1)).style('line-color', '#ACD');
			} else {
				const point = response.error;
				if (response.curr_rbm != curr_rbm) {
					curr_epoch = 0;
					curr_rbm = response.curr_rbm;

					// announce training has ended for the last RBM:
					networkGraph.$('.rbm' + curr_rbm).style('line-color', '#ACD');
					// announce training has started for the current RBM:
					networkGraph.$('.rbm' + (curr_rbm + 1)).style('line-color', '#D89');
				}

				curr_epoch++;
				chart.series[curr_rbm].addPoint([curr_epoch, point], true);

				if (autoContinue)
					retrieveError(true);
			}
		}
	});
}
