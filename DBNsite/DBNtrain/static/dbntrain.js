/**
 * The current training epoch number.
 * @type {Number}
 */
var curr_epoch;

/**
 * The currently training RBM number index.
 * @type {Number}
 */
var curr_rbm;

/**
 * A unique identifier for the training
 * job running on the server.
 * @type {String}
 */
var job_id;

/**
 * The DBN graph.
 * @type {Cytoscape}
 */
var networkGraph;

/**
 * The chart for plotting the reconstruction
 * error over time, while training the net.
 * @type {Highcharts.Chart}
 */
var chart;





/**
 * Updates the architecture form after
 * after the `tab` key is pressed but
 * _before_ its default behaviour is applied.
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
			name: 'Training error for RBM 1',
			data: []
		}]
	});
}

/**
 * Binds the submission of the training form to
 * an AJAX request that submits the training
 * hyper-parameters to the server.
 */
function setupTrainForm() {
	$('#train_form').submit(function(e) {

		// reset counters and chart series:
		curr_epoch = 0;
		curr_rbm = -1;
		updateSeries();

		var epochs = $('#epochs').val();
		var num_layers = $('#num_layers').val();
		chart.xAxis[0].setExtremes(1, epochs);
		for (var i = 1; i < num_layers; i++)
			networkGraph.$('.rbm' + i).style('line-color', '#ACD');

		var net_form_data = $('#net_form').serialize();
		var train_form_data = $('#train_form').serialize();
		var forms_data = net_form_data + '&' + train_form_data;

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
 * Updates the form for defining the DBN layers,
 * based on the number of layers that the user
 * wants to create.
 */
function updateArchitecture() {
	var num_layers = $('#num_layers').val();
	if (num_layers == '')
		return;
	if (num_layers < 1) {
		alert('You must have at least one layer (the visible layer): defaulting to 1.');
		$('#num_layers').val(1);
		num_layers = 1;
	}
	if (num_layers > 6) {
		alert('Too many layers (' + num_layers + '): defaulting to 6.');
		$('#num_layers').val(6);
		num_layers = 6;
	}

	// how many HTML inputs do we already have?
	var curr_layers = $('.lay_sz').length;

	// add missing hidden layers:
	for (var i = curr_layers; i < num_layers; i++)
		$('#layers_sz').append('<li class="lay_sz"><input type="text" name="hid_sz_' + i + '" id="hid_sz_' + i + '" onchange="updateGraph();" /></li>');

	// remove exceeding hidden layers:
	for (var i = curr_layers; i > num_layers; i--)
		$('.lay_sz').last().remove();
}

/**
 * Updates the time series for the error plot,
 * matching the number of RBMs defined in the
 * architecture form.
 */
function updateSeries() {
	var num_layers = $('#num_layers').val();
	var num_rbms = num_layers - 1

	// remove all the series:
	for (var i = chart.series.length - 1; i >= 0; i--)
		chart.series[i].remove();

	// add the necessary series:
	for (var i = 0; i < num_rbms; i++)
		chart.addSeries({
			name: 'Training error for RBM ' + (i + 1),
			data: []
		});
}

/**
 * Repaints the Cytoscape graph, checking
 * for changes in the HTML form.
 */
function updateGraph() {
	var num_layers = $('#num_layers').val();
	var prec_layer_nodes = 0;
	networkGraph = cytoscape({
		container: document.getElementById('DBNgraph')
	});

	// gather new nodes and edges:
	var graphElements = [];
	for (var layer = 0; layer < num_layers; layer++) {
		if (layer == 0) {
			var label = 'Visible layer';
			var selector = $('#vis_sz');
		} else {
			var label = 'Hidden layer ' + layer;
			var selector = $('#hid_sz_' + layer);
		}
		var num_nodes = (+selector.val())

		if (num_nodes > 0) {
			var max_real_nodes = 10000;
			var max_visible_nodes = 20;
			var max_rendered_nodes = 30;
			var edgeThickness = max_rendered_nodes / (prec_layer_nodes + num_nodes);

			if (num_nodes > max_real_nodes) {
				alert(label + ' has too many nodes (' + num_nodes + '): aborting; defaulting to ' + max_real_nodes + '.');
				selector.val(max_real_nodes);
				num_nodes = max_real_nodes;
			}

			var parent = undefined;
			if (num_nodes > max_visible_nodes) {
				parent = label;
				graphElements.push({
					group: 'nodes',
					data: {
						id: label,
						placeholder: '(' + num_nodes + ' nodes)'
					}
				});
				var scale_base = Math.pow(max_real_nodes, 1 / max_rendered_nodes);
				num_nodes = baseLog(num_nodes, scale_base);
			}

			for (var node = 1; node <= num_nodes; node++) {
				graphElements.push({
					group: 'nodes',
					data: {
						id: nodeId(layer, node),
						parent: parent
					},
					position: { // X is horizontal, Y is vertical.
						x: ((num_nodes / 2) - node + 0.5) * networkGraph.width() / num_nodes,
						y: ((num_layers / 2) - layer + 0.5) * networkGraph.height() / num_layers
					}
				});

				for (var prec_node = 1; prec_node <= prec_layer_nodes; prec_node++) {
					var source_id = nodeId(layer, node);
					var target_id = nodeId(layer - 1, prec_node);
					graphElements.push({
						group: 'edges',
						classes: 'rbm' + layer,
						data: {
							id: edgeId(source_id, target_id),
							thickness: edgeThickness,
							source: source_id,
							target: target_id
						}
					});
				}
			}

			prec_layer_nodes = num_nodes;
		}
	}

	networkGraph = cytoscape({
		container: document.getElementById('DBNgraph'),
		elements: graphElements,
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
			}
		],
		layout: {
			name: 'preset'
		}
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
 * Asks the server to train the network for
 * one epoch, then updates the reconstruction
 * error on the chart.
 */
function retrieveError(autoContinue) {
	$('#train_plot_loading').css('opacity', 1);

	var parameters = { 'job_id': job_id };
	$.ajax({
		type: 'POST',
		url: 'getError/',
		data: JSON.stringify(parameters),
		contentType: 'application/json; charset=utf-8',
		dataType: 'json',
		success: function(response) {
			$('#train_plot_loading').css('opacity', 0);

			if (response.stop) {
				// announce training has ended for the last RBM:
				networkGraph.$('.rbm' + (curr_rbm + 1)).style('line-color', '#ACD');
			} else {
				var point = response.error;
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
