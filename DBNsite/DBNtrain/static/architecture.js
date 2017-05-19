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
		$('#layers_sz').append('<li class="lay_sz"><input type="text" name="hid_sz_' + i + '" id="hid_sz_' + i + '" onchange="updateGraph();"></li>');

	// remove exceeding hidden layers:
	for (var i = curr_layers; i > num_layers; i--)
		$('.lay_sz').last().remove();
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
 * Repaints the Cytoscape graph, checking
 * for changes in the HTML form.
 */
function updateGraph() {
	var num_layers = $('#num_layers').val();
	var prec_layer_nodes = 0;

	// gather new nodes and edges:
	var graphElements = [];
	for (var layer = 0; layer < num_layers; layer++) {
		var num_nodes = 0;
		if (layer == 0)
			num_nodes = $('#vis_sz').val();
		else
			num_nodes = $('#hid_sz_' + layer).val();

		if (num_nodes != '' && num_nodes > 0) {
			var nodesClass = '';
			var edgesClass = '';
			if (num_nodes > 2999)
				return alert('Too many nodes at layer ' + layer + ' (' + num_nodes + '): aborting.');
			if (num_nodes > 99) {
				nodesClass = 'large_1';
				if (num_nodes > 499)
					nodesClass = 'large_2';
			}
			if (prec_layer_nodes + num_nodes > 149)
				edgesClass = 'dense';
			for (var node = 1; node <= num_nodes; node++) {
				graphElements.push({
					group: 'nodes',
					classes: nodesClass,
					data: { id: nodeId(layer, node) },
					position: {
						x: ((node - 0.5) / num_nodes) * 600 + 50,
						y: (num_layers - layer + 0.5) * (300 / num_layers)
					}
				});
				for (var prec_node = 1; prec_node <= prec_layer_nodes; prec_node++) {
					var source_id = nodeId(layer, node);
					var target_id = nodeId(layer - 1, prec_node);
					graphElements.push({
						group: 'edges',
						classes: edgesClass,
						data: {
							id: edgeId(source_id, target_id),
							source: source_id,
							target: target_id
						}
					});
				}
			}
			prec_layer_nodes = num_nodes;
		}
	}

	var networkGraph = cytoscape({
		container: document.getElementById('DBNgraph'),
		elements: graphElements,
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
				selector: '.large_1',
				style: {
					'width': 7,
					'height': 7,
				}
			},
			{
				selector: '.large_2',
				style: {
					'width': 4,
					'height': 4,
				}
			},
			{
				selector: 'edge',
				style: {
					'width': 0.8,
					'line-color': '#BCE'
				}
			},
			{
				selector: '.dense',
				style: {
					'width': 0.32
				}
			}
		],
		layout: {
			name: 'preset'
		}
	});
}

/**
 * Prevents the submission of the form
 * when the `enter` key is pressed.
 */
$(function() {
	$("#train_form").keydown(function (e) {
		if (e.which == 13) e.preventDefault();
	});
});

/**
 * Updates the architecture form after
 * after the `tab` key is pressed but
 * _before_ its default behaviour is applied.
 */
$(function() {
	$("#net_form").keydown(function (e) {
		updateArchitecture();
		if (e.which == 8) updateArchitecture();
	});
});
