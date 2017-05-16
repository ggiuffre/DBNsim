/**
 * The maximum number of RBMs allowed in a DBN.
 * Feel free to change this number.
 * @type {Number}
 */
var max_rbms = 8;

/**
 * The stylesheet for the DBN graph
 * (with Cytoscape).
 * @type {Array}
 */
var networkStyle = [
	{
		selector: 'node',
		style: {
			'background-color': '#46A',
			'label': 'data(id)'
		}
	},
	{
		selector: 'edge',
		style: {
			'width': 1,
			'line-color': '#AAC'
		}
	}
];

/**
 * The Cytoscape layout for
 * the DBN architecture
 * @type {Object}
 */
var networkLayout = {
	name: 'cose'
}

/**
 * Sets up the graph for the
 * DBN architecture.
 */
function setupGraph() {
	var networkGraph = cytoscape({
		container: document.getElementById('DBNgraph'),
		elements: [],
		style: networkStyle,
		layout: networkLayout
	});
}

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
		$('#arch_form').append('<input type="text" name="' + rbm_id + '" id="' + rbm_id + '" class="hid_sz" onchange="updateLayers();" />');
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
function nodeId(rbm, node) {
	return 'rbm' + rbm + 'n' + node;
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
function updateLayers() {
	var num_rbms = $('.hid_sz').length;
	var prec_layer_nodes = 0;

	// gather new nodes and set up the edges:
	var graphElements = []
	for (var rbm = 1; rbm <= num_rbms; rbm++) {
		var num_nodes = $('#hid_sz_' + rbm).val();
		if (num_nodes != '') {
			for (var node = 1; node <= num_nodes; node++) {
				graphElements.push({
					group: 'nodes',
					data: { id: nodeId(rbm, node) }/*,
					position: {
						x: node,
						y: (num_rbms - rbm) * 10
					}*/
				});
				for (var prec_node = 1; prec_node <= prec_layer_nodes; prec_node++) {
					graphElements.push({
						group: 'edges',
						data: {
							id: edgeId(node, prec_node),
							source: nodeId(rbm, node),
							target: nodeId(rbm-1, prec_node)
						}
					});
				}
			}
			prec_layer_nodes = num_nodes;
		}
	}

	// repaint the graph:
	networkGraph = cytoscape({
		container: document.getElementById('DBNgraph'),
		elements: graphElements,
		style: networkStyle,
		layout: networkLayout
	});

	for (var rbm = 1; rbm <= num_rbms; rbm++) {
		var num_nodes = $('#hid_sz_' + rbm).val();
		if (num_nodes != '') {
			for (var node = 1; node <= num_nodes; node++) {
				networkGraph.$('#' + nodeId(rbm, node)).renderedPosition({
					x: node,
					y: (num_rbms - rbm) * 10
				});
			}
		}
	}
}

/**
 * Prevents the submission of the form
 * when the `enter` key is pressed.
 */
$(function() {
	$("#train_form").keydown(function (e) {
		updateArchitecture();
		// updateLayers();
		if (e.which == 13) e.preventDefault();
	});
});
