/**
 * The maximum number of RBMs allowed in a DBN.
 * Feel free to change this number.
 * @type {Number}
 */
var max_rbms = 6;

/**
 * Updates the form for defining the DBN layers,
 * based on the number of layers that the user
 * wants to create.
 */
function updateArchitecture() {
	var num_rbms = $('#num_rbms').val();
	num_rbms = Math.min(num_rbms, max_rbms);

	var curr_rbms = $('.hid_sz').length;

	// add missing layers:
	for (var i = curr_rbms + 1; i <= num_rbms; i++) {
		var rbm_id = 'hid_sz_' + i;
		$('#arch_form').append('<br />');
		$('#arch_form').append('<label for="' + rbm_id + '">n. of hidden units (RBM ' + i + '):</label>');
		$('#arch_form').append('<br />');
		$('#arch_form').append('<input type="text" name="' + rbm_id + '" id="' + rbm_id + '" class="hid_sz" onchange="updateGraph();" />');
	}

	// remove exceeding layers:
	for (var i = num_rbms + 1; i <= curr_rbms; i++) {
		$('#hid_sz_' + i).remove();
		$('#arch_form br').last().remove();
		$('#arch_form label').last().remove();
		$('#arch_form br').last().remove();
	}
}

/**
 * Returns a standard string identifier
 * for the given node in the given RBM.
 * @param  {Number} rbm  the RBM position in the DBN
 * @param  {Number} node the node position in the RBM
 * @return {String}      a string identifier
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
	var num_rbms = $('#num_rbms').val();
	var prec_layer_nodes = 0;

	// gather new nodes and edges:
	var graphElements = [];
	for (var rbm = 1; rbm <= num_rbms; rbm++) {
		var num_nodes = $('#hid_sz_' + rbm).val();
		if (num_nodes != '' && num_nodes > 0) {
			graphElements.push({
				group: 'nodes',
				data: { id: 'Layer ' + rbm }
			});
			var nodesClass = '';
			if (num_nodes > 100)
				nodesClass = 'large';
			for (var node = 1; node <= num_nodes; node++) {
				graphElements.push({
					group: 'nodes',
					data: { id: nodeId(rbm, node), parent: 'Layer ' + rbm },
					classes: nodesClass,
					position: {
						x: ((node - 0.5) / num_nodes) * 600 + 50,
						y: (num_rbms - rbm + 0.5) * (300 / num_rbms)
					}
				});
				for (var prec_node = 1; prec_node <= prec_layer_nodes; prec_node++) {
					var source_id = nodeId(rbm, node);
					var target_id = nodeId(rbm-1, prec_node);
					graphElements.push({
						group: 'edges',
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
					'width': 15,
					'height': 15,
					'background-color': '#46A'
				}
			},
			{
				selector: '.large',
				style: {
					'width': 6,
					'height': 6,
				}
			},
			{
				selector: ':parent',
				style: {
					'background-opacity': '0.2'
				}
			},
			{
				selector: 'edge',
				style: {
					'width': 1,
					'line-color': '#BCE'
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
		updateArchitecture();
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
