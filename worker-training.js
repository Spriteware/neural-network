
if (typeof importScripts !== "undefined")
    importScripts("http://localhost/machinelearning/lib/neural_network.js");

onmessage = function(e) {

    if (!e.data.params || !e.data.weights)
        throw new NetException("Invalid params or weights in order to build a Neural Network copy", {params: e.data.params, weights: e.data.weights});

    var training_data = e.data.training_data;
    var epochs = e.data.epochs;

    console.info("Training imported data in processing... Brain copy below:");    

    // Create copy of our current Network
    var brain = new Network(e.data.params);
    brain.weights = e.data.weights;

    ///////////////////// Training //////////////////////////////

    var i, curr_epoch, training_size = training_data.length;

    var output_errors = [];
    var output_errors_mean = 0 ;
    var sum = 0, global_sum = 0;
    var mean, global_mean;

    // Feeforward NN
    for (curr_epoch = 0; curr_epoch < epochs; curr_epoch++)
    {
        for (sum = 0, i = 0; i < training_size; i++)
        {
            brain.feed(training_data[i].inputs);
            brain.backpropagate(training_data[i].targets);

            sum += brain.outputError;

            if (epochs <= _CANVAS_GRAPH_EPOCHS_TRESHOLD)
                output_errors.push( brain.outputError );
        }
        
        global_sum += sum;
        mean = sum / training_size; 
        global_mean = global_sum / ((curr_epoch+1) * training_size); 

        if (epochs > _CANVAS_GRAPH_EPOCHS_TRESHOLD)
            output_errors.push( Math.sqrt(mean) );

        // Send updates back to real thread
        self.postMessage({
            type: WORKER_TRAINING_PENDING,
            curr_epoch: curr_epoch,
            output_errors: output_errors,
            global_mean: global_mean,
        });
    }

    console.info("Training done. Gone through all epochs", {epochs: epochs, global_mean: global_mean});

    self.postMessage({
        type: WORKER_TRAINING_OVER,
        weights: brain.exportWeights(),
        biais: brain.exportBiais()
    });
};