const _SVG_STROKE_WIDTH  = 4;
const _SVG_CIRCLE_RADIUS = 15;
const _SVG_CIRCLE_COLOR_DEFAULT = "#ffe5e5";
const _SVG_CIRCLE_COLOR_DROPPED = "#c7c7c7";
const _SVG_MAX_WEIGHTS_DISPLAY_TEXT = 4;

const _CANVAS_GRAPH_WIDTH  = 400;
const _CANVAS_GRAPH_HEIGHT = 100;
const _CANVAS_GRAPH_EPOCHS_TRESHOLD = 50;

const _ERROR_VALUE_TOO_HIGH  = 100000;
const _WEIGHT_VALUE_TOO_HIGH = 10000;

const WORKER_TRAINING_PENDING = 0;
const WORKER_TRAINING_OVER    = 1;

/*
    TODO
    * trainRecurrent la fonction qui ajoute en input la récurrence
        Créer un type de Network special recurrent ? 
    * au lieu de faire une nouvelle fonction, passer un objet de paramètre à train. ça sera plus parlant
*/


function randomBiais() {
    return Math.random() * 2 - 1;
}

function randomWeight() {
    return Math.random() * 2 - 1;
}

//////////////////////////////////////////////

function tooltipOn(tooltip, event, object) {
    
    tooltip.object = object;
    tooltip.setAttribute("class", "");
    tooltip.style.left = (event.pageX+10) + "px";
    tooltip.style.top = (event.pageY+5) + "px";

    tooltipUpdate(object);
}

function tooltipUpdate(tooltip, object) {

    if (typeof object !== "object") {
        tooltip.object = object;
        return;
    }

    var buffer = "";

    for (var key in object) 
        if (object.hasOwnProperty(key) && key !== "object")
            buffer += key + ": " + object[key] + "<br />";

    tooltip.innerHTML = buffer;
}    

function tooltipOff(tooltip, event, object) {
    
    tooltip.object = undefined;
    tooltip.setAttribute("class", "off");
}

////////////////////////////////////////////


var Neuron = function(id, layer, biais) {

    this.id = id;
    this.layer = layer;
    this.biais = biais || 0;
    this.dropped = false;

    this.output = undefined;
    this.error = undefined;

    this.activation = undefined;
    this.derivative = undefined;
};

////////////////////////////////////////////

var Network = function(params) {

    // Required variables: lr, layers
    this.lr = undefined; // Learning rate
    this.momentum = undefined;
    this.layers = undefined;
    this.hiddenLayerFunction = undefined; // activation function for hidden layer

    this.neurons    = undefined;
    this.weights    = undefined;
    this.weightsTm1 = undefined; // weights at T-1 

    // Caching variables:
    this.layersSum = undefined;
    this.layersMul = undefined;
    this.nbLayers  = undefined;
    this.nbNeurons = undefined;
    this.nbWeights = undefined;

    // Stats-purpose:
    this.maxWeight = 0;
    this.outputError = 0;
    this.globalError = 0;
    this.weightsPerNeuron = 0;

    // Visualization:
    this.svgVisualization = false;
    this.DOM = {
        svg: undefined,
        tooltip: undefined,

        neuronsCircles: undefined,
        weightTexts: undefined,
        inputTexts: undefined,
        outputTexts: undefined,
        weightCurves: undefined
    };

    this.loadParams(params);
    this.initialize();
};

Network.prototype.loadParams = function(params) {

    for (var key in params)
        if (this.hasOwnProperty(key) && this[key] === undefined)
            this[key] = params[key];

    console.log("loaded params", this);    
};

Network.prototype.exportParams = function() {

    return {
        lr: this.lr,
        momentum: this.momentum,
        layers: this.layers,
        hiddenLayerFunction: this.hiddenLayerFunction
    };
};

Network.prototype.exportWeights = function() {
    return this.weights;
};

Network.prototype.importWeights = function(values) {
    this.weights = values;
};

Network.prototype.exportBiais = function() {
    
    var values = Array(this.nbNeurons);

    for (var i = 0; i < this.nbNeurons; i++)
        values[i] = this.neurons[i].biais;

    return values;
};

Network.prototype.importBiais = function(values) {

    for (var i = 0; i < this.nbNeurons; i++)
        this.neurons[i].biais = values[i];
};

Network.prototype.initialize = function() {

    if (this.lr === undefined || this.lr <= 0)
        throw new NetException("Undefined or invalid learning rate", {lr: this.lr});

    if (this.momentum === undefined || this.momentum < 0 || this.momentum > 1)
        throw new NetException("Undefined or invalid momentum (must be between 0 and 1 both included)", {momentum: this.momentum});

    if (this.layers === undefined || this.layers.length <= 0)
        throw new NetException("Undefined or unsificient layers", {layers: this.layers});
    
    var i, sum, mul;
    var curr_layer = 0;

    // Initialization
    this.nbLayers = this.layers.length;
    this.layersSum = [];
    this.layersMul = [];
    this.neurons = [];
    this.weights = [];
    this.weightsTm1 = [];

    // Prepare layers relative computation
    for (i = 0, sum = 0, mul = 1; i < this.nbLayers; i++) {
        sum += this.layers[i];
        mul = (this.layers[i-1] || 0) * this.layers[i];
        this.layersSum.push(sum);
        this.layersMul.push(mul + (this.layersMul[i-1] || 0)); 
        // [0] will be 0, Because layerMul is used to know how many weights there is before a layer, and there is no before layer 0
    }

    // Create neurons
    this.nbNeurons = sum;

    for (i = 0; i < sum; i++) {
        this.neurons.push( new Neuron(i, i >= this.layersSum[curr_layer] ? ++curr_layer : curr_layer, randomBiais()) );
        this.neurons[i].activation = this.static_linearActivation;
        this.neurons[i].derivative = this.static_linearDerivative;
    }

    // Set hidden layer activation functions
    console.log( "activation", this.hiddenLayerFunction);

    switch (this.hiddenLayerFunction)
    {
        case "tanh":
            this.setHiddenLayerToActivation(this.static_tanhActivation, this.static_tanhDerivative);
            break;

        case "sigmoid":
            this.setHiddenLayerToActivation(this.static_sigmoidActivation, this.static_sigmoidDerivative);
            break;

        case "relu":
            this.setHiddenLayerToActivation(this.static_reluActivation, this.static_reluDerivative);
            break;
        
        default:
            this.setHiddenLayerToActivation(this.static_linearActivation, this.static_linearDerivative);
    }

    // Create weights
    this.nbWeights = this.layersMul[this.layersMul.length-1];

    for (i = 0; i < this.nbWeights; i++) {
        this.weights.push( randomWeight() );
        this.weightsTm1.push( this.weights[0] );        
    }

    this.weightsPerNeuron = this.nbWeights / this.nbNeurons;
};

Network.prototype.createVisualization = function() {

    var i, l, l2, n, index;
    var x1, y1, x2, y2, max_y1 = 0;
    var neuron1, neuron2, is_input;
    var DOM_tmp, DOM_weight;

    var _MARGIN_X = 150;
    var _MARGIN_Y = 75;
    var that = this;

    // Create DOM elements
    var container = document.createElement("div");
    this.DOM.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    this.DOM.tooltip = document.createElement("div");
    this.DOM.tooltip.setAttribute("id", "tooltip");
    this.DOM.tooltip.setAttribute("class", "off");
    container.appendChild(this.DOM.svg);
    container.appendChild(this.DOM.tooltip);

    this.DOM.neuronsCircles = []; 
    this.DOM.weightTexts = []; 
    this.DOM.inputTexts = []; 
    this.DOM.outputTexts = []; 
    this.DOM.weightCurves = [];

    // Computing functions & listeners callbacks
    function calcX(neuron) {
        return (neuron.layer + 1) * _MARGIN_X;
    }
    
    function calcY(neuron) {
        return (neuron.id - (that.layersSum[neuron.layer-1] || 0)) * _MARGIN_Y + _MARGIN_Y / 2;
    }

    function neuronTooltip(event) {
        tooltipOn( that.DOM.tooltip, event, that.neurons[event.target.getAttribute("data-object")] );
    } 

    function weightTooltip(event) {
        tooltipOn( that.DOM.tooltip, event, that.weights[event.target.getAttribute("data-object")] );
    } 

    function neuronNweightTooltipOff(event) {
        tooltipOff( that.DOM.tooltip, event );
    }

    // Fetching every neuron
    for (i = 0, l = this.neurons.length; i < l; i++)
    {
        neuron1 = this.neurons[i];
        x1 = calcX(neuron1);
        y1 = calcY(neuron1);
        
        // Fetching neurons from next layer for weights
        for (n = 0, l2 = (this.layers[neuron1.layer + 1] || 0); n < l2; n++)
        {
            neuron2 = this.neurons[this.layersSum[ neuron1.layer ] + n];
            index = this.getWeightIndex(neuron1, neuron2);
            x2 = calcX(neuron2);
            y2 = calcY(neuron2);

            // Creating SVG weights
            DOM_tmp = document.createElementNS("http://www.w3.org/2000/svg", "path");
            DOM_tmp.setAttribute("class", "weight");
            DOM_tmp.setAttribute("data-object", index);
            DOM_tmp.setAttribute("d", "M" + x1 + "," + y1 +" C" + (x1 + _MARGIN_X/2) + "," + y1 + " " + (x1 + _MARGIN_X/2) + "," + y2 + " " + x2 + "," + y2);
            DOM_tmp.setAttribute("stroke-width", _SVG_STROKE_WIDTH);
            DOM_tmp.addEventListener("mousemove", weightTooltip);
            DOM_tmp.addEventListener("mouseout", neuronNweightTooltipOff);

            this.DOM.svg.appendChild(DOM_tmp);
            this.DOM.weightCurves.push(DOM_tmp);
            
            // Creating SVG weight Text
            DOM_tmp = document.createElementNS("http://www.w3.org/2000/svg", "text");
            DOM_tmp.setAttribute("class", "weight-text");
            DOM_tmp.setAttribute("data-object", index);
            DOM_tmp.setAttribute("x", x1 + (x2 - x1) * 0.2);
            DOM_tmp.setAttribute("y", y1 + (y2 - y1) * 0.2);

            this.DOM.weightTexts.push(DOM_tmp);
        }

        // Creating SVG input/output lines and text
        if (neuron1.layer === 0 || neuron1.layer === this.nbLayers-1)
        {
            is_input = neuron1.layer === 0 ? 1 : -1;
            
            DOM_tmp = document.createElementNS("http://www.w3.org/2000/svg", "path");
            DOM_tmp.setAttribute("class", "weight");
            DOM_tmp.setAttribute("d", "M" + x1 + "," + y1 +" L" + (x1 - _MARGIN_X / 4 * is_input) + "," + y1);

            this.DOM.svg.appendChild(DOM_tmp);

            DOM_tmp = document.createElementNS("http://www.w3.org/2000/svg", "text");
            DOM_tmp.setAttribute("class", is_input === 1 ? "input-text" : "output-text");
            DOM_tmp.setAttribute("x", is_input === 1 ? x1 - _MARGIN_X / 1.8 : x1 + _MARGIN_X / 3);
            DOM_tmp.setAttribute("y", y1 + 5);

            if (is_input === 1)
                this.DOM.inputTexts.push(DOM_tmp);
            else
                this.DOM.outputTexts.push(DOM_tmp);
        }

        // Creating SVG neuron
        DOM_tmp = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        DOM_tmp.setAttribute("class", "neuron");
        DOM_tmp.setAttribute("data-object", neuron1.id);
        DOM_tmp.setAttribute("cx", x1);
        DOM_tmp.setAttribute("cy", y1);
        DOM_tmp.setAttribute("r", _SVG_CIRCLE_RADIUS);
        DOM_tmp.setAttribute("fill", _SVG_CIRCLE_COLOR_DEFAULT);
        DOM_tmp.addEventListener("mousemove", neuronTooltip);
        DOM_tmp.addEventListener("mouseout", neuronNweightTooltipOff);

        this.DOM.svg.appendChild(DOM_tmp);
        this.DOM.neuronsCircles.push(DOM_tmp);
        
        max_y1 = max_y1 < y1 ? y1 : max_y1;
    }

    // We stretch our svg document (here x2 is supposed to be the maximum possible)
    var width = x2 + _MARGIN_X, height = max_y1 + _MARGIN_Y / 2, scale = 1.5;
    this.DOM.svg.setAttribute("width", width >= window.innerWidth ? width/scale : width);
    this.DOM.svg.setAttribute("height", width >= window.innerWidth ? height/scale : height);
    this.DOM.svg.setAttribute("viewBox", "0 0 " + width + " " + height);

    // Push text elements on top of everything
    var svg_texts = this.DOM.outputTexts.concat( this.DOM.inputTexts.concat( this.DOM.weightTexts ));

    for (i = 0, l = svg_texts.length; i < l; i++)
        this.DOM.svg.appendChild( svg_texts[i] );

    this.svgVisualization = true;
    console.info("SVG visualization ready");

    return container;
};    

Network.prototype.visualize = function(inputs, scales) {

    if (!this.svgVisualization)
        throw new NetException("SVG Visualization is not available", {network: this});

    if (!inputs || inputs.length !== this.layers[0])
        throw new NetException("Incorrect inputs (undefined or incorrect length)", {inputs: inputs, layer: this.layers[0]});

    var i, l;
    var output_neurons = this.getNeuronsInLayer( this.nbLayers-1 );

    if (scales && scales.length !== output_neurons.length)
        throw new NetException("Incorrect scales which is not about the same size as outputs", {scales: scales, outputs_neurons: output_neurons});

    // Update SVG text inputs
    for (i = 0, l = this.DOM.inputTexts.length; i < l; i++)
        this.DOM.inputTexts[i].innerHTML = (inputs[i] * (scales ? scales[i] : 1)).toFixed(1);

    // Update SVG text outputs
    for (i = 0, l = this.DOM.outputTexts.length; i < l; i++)
        this.DOM.outputTexts[i].innerHTML = (output_neurons[i].output * (scales ? scales[i] : 1)).toFixed(1);

    // Update SVG weights
    for (i = 0, l = this.nbWeights; i < l; i++) {
        this.DOM.weightCurves[i].setAttribute("stroke-width", Math.abs(this.weights[i]) / this.maxWeight * _SVG_STROKE_WIDTH);
        if (this.weightsPerNeuron < _SVG_MAX_WEIGHTS_DISPLAY_TEXT) 
            this.DOM.weightTexts[i].innerHTML = this.weights[i].toFixed(4);
    }

    // Update tooltip
    if (this.DOM.tooltip.object !== undefined)
        tooltipUpdate(this.DOM.tooltip, this.DOM.tooltip.object);
};

Network.prototype.feed = function(inputs, scales) {

    if (!inputs || inputs.length !== this.layers[0])
        throw new NetException("Incorrect inputs", {inputs: inputs, layer: this.layers[0]});

    var index, n, l, sum, neuron, prev_neurons; // neurons from previous layer
    var curr_layer = 0;

    // Input layer filling
    for (index = 0; index < this.layers[0]; index++)
        this.neurons[index].output = inputs[index];

    // Fetching neurons from second layer (even if curr_layer equals 0, it'll be changed directly)
    for (index = this.layers[0]; index < this.nbNeurons; index++)
    {
        neuron = this.neurons[index];

        if (neuron.dropped)
            continue;

        // Update if necessary all previous layer neurons
        if (prev_neurons === undefined || neuron.layer !== curr_layer)
            prev_neurons = this.getNeuronsInLayer(curr_layer++);

        // Computing w1*x1 + ... + wn*xn
        for (sum = 0, n = 0, l = prev_neurons.length; n < l; n++) {
            if (!prev_neurons[n].dropped)
                sum += this.getWeight(prev_neurons[n], neuron) * prev_neurons[n].output;
        }

        // Updating output    
        neuron.output = neuron.activation(sum + neuron.biais); 

        // if it's output layer, we apply the scales. Finally we have to do it inside our model if we want the error to detect differencies
        // if (index >= this.layersSum[this.nbLayers-2])
        //     neuron.output *= scales[index - this.layersSum[this.nbLayers-2]];

        if (!isFinite(neuron.output)) {
            
            for (sum = 0, n = 0, l = prev_neurons.length; n < l; n++)
                console.log(n, this.getWeight(prev_neurons[n], neuron));
            throw new NetException("non finite or too high output", {neuron: neuron});
        }
    }

    // console.log("neurons:");
    // console.table(this.neurons);

    return this.getNeuronsInLayer(this.nbLayers-1);
};

Network.prototype.backpropagate = function(targets) {

    var outputs_neurons = this.getNeuronsInLayer(this.nbLayers-1);

    if (!targets || !outputs_neurons || targets.length !== outputs_neurons.length)
        throw new NetException("Incoherent targets for current outputs", {targets: targets, outputs_neurons: outputs_neurons});

    // Computing output error
    // https://fr.wikipedia.org/wiki/R%C3%A9tropropagation_du_gradient

    var index, weight_index, n, l, sum, err, grad, weight, weightTm1, tmp, max_weight = 0;
    var output_error = 0, curr_layer = this.nbLayers-1;
    var neuron, next_neurons;

    this.globalError = 0;

    // Output layer filling: err = (expected-obtained) - and normalize;
    for (n = l = outputs_neurons.length; n > 0; n--)
    {
        neuron = outputs_neurons[l - n];
        grad = neuron.derivative(neuron.output);
        err = targets[l - n] - neuron.output;

        // var err2 = 1/2 * (targets[l - n] - neuron.output)*(targets[l - n] - neuron.output);

        // console.log( "output err1 & err2", err, err2 );
        // err = 1/2 * (targets[l - n] - neuron.output) * (targets[l - n] - neuron.output); // ne marche pas du tout

        neuron.error = grad * err;
        output_error += Math.abs(neuron.error);

        if (!isFinite(neuron.error))
            throw new NetException(" Non finite error on output neuron", {neuron: neuron});
    }

    this.outputError = output_error;

    // if (this.outputError < 0)
        // throw new NetException("Output error is under 0", {output_error: this.outputError, targets: targets});
    
    // if (this.outputError > 1)
        // console.error("Output error over 1", {output_error: output_error});
        // throw new NetException("Output error is over 1. Please try to find better targets", {output_error: this.outputError, targets: targets});

    // Fetching neurons from last layer
    for (index = this.layersSum[curr_layer-1] - 1; index >= 0; index--)
    {
        neuron = this.neurons[index];

        if (neuron.dropped)
            continue;

        // Update if necessary all next layer neurons
        if (next_neurons === undefined || neuron.layer !== curr_layer)
            next_neurons = this.getNeuronsInLayer(curr_layer--);

        // Computing w1*e1 + ... + wn*en
        for (sum = 0, n = 0, l = next_neurons.length; n < l; n++) {
            if (!next_neurons[n].dropped)
                sum += this.getWeight(neuron, next_neurons[n]) * next_neurons[n].error;
                // sum += 1/2 * (this.getWeight(neuron, next_neurons[n]) * next_neurons[n].error) * (this.getWeight(neuron, next_neurons[n]) * next_neurons[n].error);
        }

        // Updating error    
        neuron.error = sum * neuron.derivative(neuron.output); 
        this.globalError += Math.abs(neuron.error); 
        
        if (!isFinite(neuron.error)) {
            throw new NetException("non finite error", {neuron: neuron});
        } else if (Math.abs(neuron.error) > _ERROR_VALUE_TOO_HIGH) {
            console.info("scaling down error to a max", {neuron: neuron, error: neuron.error});
            neuron.error = neuron.error < 0 ? - _ERROR_VALUE_TOO_HIGH : _ERROR_VALUE_TOO_HIGH;
            throw new NetException("computed error is too high", {neuron: neuron});
        }

        // Updating weights w = w + lr * en * output
        for (n = 0, l = next_neurons.length; n < l; n++)
        {
            if (next_neurons[n].dropped)
                continue;

            if (next_neurons[n].error >= _ERROR_VALUE_TOO_HIGH) {
                console.info("avoiding backpropagation on weight due to error too high", {error: next_neurons[n].error});
                continue;
            }

            weight_index = this.getWeightIndex(neuron, next_neurons[n]); 

            // We introduce momentum to escape local minimums
            tmp = this.weights[weight_index];
            weightTm1 = this.weightsTm1[weight_index];

            if (this.momentum !== 0)
                weight = tmp + this.lr * next_neurons[n].error * neuron.output + this.momentum * weightTm1;
            else
                weight = tmp + this.lr * next_neurons[n].error * neuron.output;

            // Update maxWeight (for visualisation)
            max_weight = max_weight < Math.abs(weight) ? Math.abs(weight) : max_weight;

            if (!isFinite(weight)) {
                throw new NetException("non finite weight", {neuron: neuron, weight: weight});
            } else if (Math.abs(weight) > _WEIGHT_VALUE_TOO_HIGH) {
                console.info("scaling down weight to a max", {neuron: neuron, weight: weight});
                weight = weight < 0 ? - _WEIGHT_VALUE_TOO_HIGH : _WEIGHT_VALUE_TOO_HIGH;
                // throw new NetException("non finite or too high weight", {neuron: neuron, weight: weight});
            }

            // Finally update weights
            this.weights[weight_index] = weight;
            this.weightsTm1[weight_index] = tmp;
        }

        // Update biais (really important...)
        neuron.biais = neuron.biais + this.lr * neuron.error;

        if (!isFinite(neuron.biais))
            throw new NetException("non finite biais", {neuron: neuron});
    }

    this.maxWeight = max_weight;

    // console.clear();
    // console.log( this.weights );
    // console.table( this.neurons );
};

Network.prototype.dropout = function(completely_random, drop_inputs) {

    // Dropping out random neurons allows to push out our network of a bad solution
    // We usually start from first hidden layer, but could be possible to start from inputs layer

    var i, l, n, neurons, shot, p = 0.6;

    for (i = drop_inputs ? 0 : 1; i < this.nbLayers-1; i++)
    {
        neurons = this.getNeuronsInLayer(i);
        shot = completely_random === true ? undefined : Math.round( Math.random() * (this.layers[i] - 1) );

        for (n = 0, l = neurons.length; n < l; n++)
        {
            if (shot === n || (shot === undefined && Math.random() >= p))
            {
                if (neurons[n].dropped === false && this.svgVisualization === true) // update vizualisation {
                    this.DOM.neuronsCircles[this.getNeuronIndex(i, n)].setAttribute("fill", _SVG_CIRCLE_COLOR_DROPPED);
                neurons[n].dropped = true;
            }
            else 
            {
                if (neurons[n].dropped === true && this.svgVisualization === true) // update vizualisation
                    this.DOM.neuronsCircles[this.getNeuronIndex(i, n)].setAttribute("fill", _SVG_CIRCLE_COLOR_DEFAULT);
                neurons[n].dropped = false;
            }
        }
    } 
};

Network.prototype.train = function(training_data, epochs, visualise) {

    if (!training_data)
        throw new NetException("Invalid training data", {training_data: training_data});

    if (!epochs || isNaN(epochs))
        throw new NetException("Invalid epochs number for training", {epochs: epochs});
        
    if (typeof window.Worker === "undefined" || !window.Worker)
        throw new NetException("Web Worker is not supported by your client. Please upgrade in order to train as background operation");

    var container, graph, graph_ctx, text_output;
    var scaled_width, training_size = training_data.length;

    if (visualise)
    {
        // Create canvas
        container = document.createElement("div");
        container.setAttribute("style", "margin: 10px;");

        graph = document.createElement("canvas");
        graph.setAttribute("width", _CANVAS_GRAPH_WIDTH);
        graph.setAttribute("height", _CANVAS_GRAPH_HEIGHT);
        container.appendChild( graph );

        // Create global error mean output
        text_output = document.createElement("samp");
        container.appendChild( text_output ); 

        // We don't want to display too much data futilely
        if (epochs <= _CANVAS_GRAPH_EPOCHS_TRESHOLD)
            scaled_width = _CANVAS_GRAPH_WIDTH / (epochs * training_size);
        else
            scaled_width = _CANVAS_GRAPH_WIDTH / epochs;

        graph_ctx = graph.getContext("2d");
        graph_ctx.translate(0, _CANVAS_GRAPH_HEIGHT);
        graph_ctx.scale(scaled_width, - _CANVAS_GRAPH_HEIGHT);
        graph_ctx.globalAlpha = 0.5;
        graph_ctx.lineWidth = 0.03;
    }

    //////////////// Worker below ////////////////////////////

    var worker = new Worker("http://localhost/machinelearning/lib/worker-training.js");
    var that = this;

    worker.addEventListener("message", function(e) {
        
        if (typeof e.data.type === "undefined")
            throw new NetException("Worker message needs to contain message type (WORKER_TRAINING_X)", {data: e.data});

        // Training is over : we display our output_errors curves
        if (e.data.type === WORKER_TRAINING_PENDING)
        {
            if (!visualise)
                return;

            window.requestAnimationFrame(function() {
    
                var output_errors = e.data.output_errors;
                var global_mean = e.data.global_mean;
                var i, l, o, oel = output_errors.length;
                var tmp, sum = 0, values = [], moving_averages = [], _AVERAGES_SIZE = Math.round(oel / 10);

                /*
                    Warning! depending on the asked number of epochs, output_errors contains all output_errors
                    for all training set inputs of all epochs, or can be only one mean value for each epoch
                    Be careful by modifying all the computing stuff, be aware of the lengths. Use oel
                */
                
                // Display error curve
                graph_ctx.clearRect(0, 0, _CANVAS_GRAPH_WIDTH / scaled_width, 1);
                graph_ctx.beginPath();
                graph_ctx.moveTo(0, 0);
        
                for (o = 0; o < oel; o++) {
        
                    graph_ctx.lineTo(o, output_errors[o] / global_mean * 0.2);
    
                    // Graphically separate epochs
                    if (epochs < 10 && o > 0 && o % training_size === 0) {
                        graph_ctx.save();
                        graph_ctx.fillStyle = "#d65f42";
                        graph_ctx.fillRect(o, 0, 2 / scaled_width, 1);
                        graph_ctx.restore();
                    }

                    // Compute moving averages for a smooth curve
                    tmp = epochs <= _CANVAS_GRAPH_EPOCHS_TRESHOLD ? Math.sqrt(output_errors[o]) || 0 :  output_errors[o] || 0;
                    values.push(tmp);
                    sum += tmp;
                    moving_averages.push(sum / values.length);

                    if (o >= _AVERAGES_SIZE)
                        sum -= values.shift(values);
                }
        
                graph_ctx.lineTo(o, 0);
                graph_ctx.closePath();
                graph_ctx.fill();
                // End displaying curves

                // Display smoother error curve, only based on means :
                graph_ctx.save();
                graph_ctx.strokeStyle = "#42b9f4";
                graph_ctx.beginPath();
                graph_ctx.moveTo(0, output_errors[0]);
                
                for (i = 0, l = moving_averages.length; i < l; i++)
                    graph_ctx.lineTo(i, moving_averages[i] / global_mean * 0.2);

                graph_ctx.lineTo(oel, moving_averages[i-1] / global_mean * 0.2);
                graph_ctx.stroke();
                graph_ctx.restore();
                // End display smoother curves

                // Update output text display
                text_output.innerHTML = "epoch " + (e.data.curr_epoch+1) + "/" + epochs + " | output error mean: " + global_mean.toFixed(5);
            });
        }

        // Training is over : we update our weights an biais
        else if (e.data.type === WORKER_TRAINING_OVER)
        {
            that.importWeights( e.data.weights );
            that.importBiais( e.data.biais );

            // Feeding and bping in order to have updated values (as error) into neurons or others
            that.feed( training_data[0].inputs );
            that.backpropagate( training_data[0].targets );
        }
    });

    // Start web worker with training data through epochs
    worker.postMessage({
        params: this.exportParams(),
        weights: this.exportWeights(),
        biais: this.exportBiais(),
        training_data: training_data,
        epochs: epochs
    });

    return container || null;
};

Network.prototype.getNeuronIndex = function(layer, n) {

    if (layer === undefined || layer < 0 || layer >= this.nbLayers)
        throw new NetException("Invalid layer access", {layer: layer, n: n});

    if (n ===undefined || n >= this.layers[layer])
        throw new NetException("Invalid neuron access", {layer: layer, n: n});

    return (this.layersSum[layer-1] || 0) + n;
};

Network.prototype.getNeuron = function(layer, n) {

    return this.neurons[this.getNeuronIndex(layer, n)];
};    

Network.prototype.getNeuronsInLayer = function(layer) {

    if (layer === undefined || layer < 0 || layer >= this.nbLayers)
        throw new NetException("Invalid layer access", {layer: layer});

    return this.neurons.slice( this.layersSum[layer] - this.layers[layer], this.layersSum[layer]);
};

Network.prototype.getWeightIndex = function(from, to, debug) {

    if (!from || !to)
        throw new NetException("Invalid weight access, wrong neurons", {from: from, to: to});

    if (to.layer - from.layer !== 1 || to.layer <= 0 || to.layer >= this.nbLayers)
        throw new NetException("Invalid weight access, layers are not incorrect", {from: from, to: to});

    // How to explain this formula ? IT'S RIGHT FROM MY BRAIN
    var part1 = this.layersMul[from.layer]; // How many weights there is before from.layer
    var part2 = (from.id - (this.layersSum[from.layer-1] || 0)) * this.layers[to.layer]; // How many weights there is in from.layer, but before our neuron
    var part3 = to.id - this.layersSum[from.layer]; // How many weights there is from our neuron, which are not going to our second neuron
    var index = part1 + part2 + part3;

    if (debug || isNaN(this.weights[index]) || part1 < 0 || part2 < 0 || part3 < 0 || index < from.id)
    {
        console.log(from, to);
        console.log("index: ", index);
        console.log("#1", part1);
        console.log("#2", part2);
        console.log("#3", part3);

        if (isNaN(this.weights[index]))
            throw new NetException("NaN detected for computing weight index");
        else if (part1 < 0 || part2 < 0 || part3 < 0)
            throw new NetException("Parts calculus is incorrect: negatives values");
            else if (index < from.id)
            throw new NetException("Incoherent index inferior to from.id");
        else 
            throw new NetException("Error: debug launched", {debug: debug});
    }

    return index;
};

Network.prototype.getWeight = function(from, to) {
    
    return this.weights[this.getWeightIndex(from, to)];
};

Network.prototype.getWeightTm1 = function(from, to) {
    
    return this.weightsTm1[this.getWeightIndex(from, to)];
};
     
Network.prototype.setWeight = function(from, to, value) {

    this.weights[this.getWeightIndex(from, to)] = value;
};

Network.prototype.setWeightTm1 = function(from, to, value) {
    
    this.weights[this.getWeightIndex(from, to)] = value;
};

Network.prototype.setHiddenLayerToActivation = function(activation, derivation) {
 
    if (!activation || !derivation)
        throw new NetException("Invalid activation and/or derivation assignment", {activation: activation, derivation: derivation});
    
    for (var i = this.layers[0]; i < this.layersSum[this.nbLayers-2]; i++) {
        this.neurons[i].activation = activation;
        this.neurons[i].derivative = derivation;
    }
};


/////////////////////////// Statics activation functions 


Network.prototype.static_linearActivation = function(x) {
    return x;
};

Network.prototype.static_linearDerivative = function(x) {
    return 1;
};

Network.prototype.static_tanhActivation = function(x) {
    return Math.tanh(x);
};

Network.prototype.static_tanhDerivative = function(x) {
    return 1 - (Math.tanh(x) * Math.tanh(x));
};

Network.prototype.static_sigmoidActivation = function(x) {
    return 1 / (1 + Math.exp(-x));
};

Network.prototype.static_sigmoidDerivative = function(x) {
    return 1 / (1 + Math.exp(-x)) * (1 - 1 / (1 + Math.exp(-x)));
};

Network.prototype.static_reluActivation = function(x) {
    return x < 0 ? 0 : x;
};

Network.prototype.static_reluDerivative = function(x) {
    return x < 0 ? 0 : 1;
};
    
//////////////////////////////////////////////

function NetException(message, variables) {
    console.error("ERROR: " + message, variables);
}
