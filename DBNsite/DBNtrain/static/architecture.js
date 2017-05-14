var max_rbms = 8;

/**
 * Updates the form for defining the layers,
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
		$('#arch_form').append('<input type="text" name="' + rbm_id + '" id="' + rbm_id + '" class="hid_sz" />');
		$('#arch_form').append('<label for="' + rbm_id + '">n. of hidden units (rbm ' + i + ')</label>');
	}

	// remove exceeding layers:
	for (var i = num_rbms + 1; i <= curr_rbms; i++) {
		$('#hid_sz_' + i).remove();
		$('#arch_form label').last().remove();
		$('#arch_form br').last().remove();
	}
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
