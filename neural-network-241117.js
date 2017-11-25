"use strict";

const _AVAILABLE_OPTIMIZERS = ["momentum", "nag", "adagrad", "adadelta", "adam"];
const _WEIGHT_RANDOM_COEFF  = 1;    // must be one if we want to keep a normal distributation centered in 0
const _BIAIS_RANDOM_COEFF   = 0.0;  // usually, can be 0 or 0.1. See http: //cs231n.github.io/neural-networks-2/
const _DROPOUT_PROBABILITY  = 0.5;  // usually a good value also
const _EPSILON = 1e-8;

const _TRAINING_GATHER_ALL_THRESHOLD = 100000;
const _TRAINING_DROPOUT_EPOCHS_THRESHOLD = 200;
const _TRAINING_DROPOUT_MEAN_THRESHOLD = 0.001;

const _DEFAULT_TRAINING_BACKPROPAGATE = true;
const _DEFAULT_TRAINING_DROPOUT = false;
const _DEFAULT_TRAINING_SHUFFLE = true;

const _WORKER_TRAINING_PENDING = 0;
const _WORKER_TRAINING_OVER = 1;

const _ERROR_VALUE_TOO_HIGH = 100000;
const _WEIGHT_VALUE_TOO_HIGH = 10000;

const _CANVAS_GRAPH_DEFAULT_WIDTH = 600;
const _CANVAS_GRAPH_DEFAULT_HEIGHT = 100;
const _CANVAS_GRAPH_WINDOW_FACTOR = 1 / 0.9;
const _CANVAS_GRAPH_SMOOTH_FACTOR = 1 / 20;
const _CANVAS_GRAPH_SEPARATE_EPOCHS_THRESHOLD = 20;

const _SVG_STROKE_WIDTH = 4;
const _SVG_CIRCLE_RADIUS = 15;
const _SVG_CIRCLE_COLOR_DEFAULT = "#ffe5e5";
const _SVG_CIRCLE_COLOR_DROPPED = "#c7c7c7";
const _SVG_MAX_WEIGHTS_DISPLAY_TEXT = 4;

const _COLOR_ASPHALT = "rgb(52, 73, 94)";
const _COLOR_PURPLE = "rgb(142, 68, 173)";
const _COLOR_BLUE = "rgb(52, 152, 219)";
const _COLOR_GREEN = "rgb(26, 188, 156)";

/////////////////////////////// Utils - various functions 

var Utils = {
    static: {}, // yes, it's just sugar for a good looking in console....
    trainingData: "",
    trainingSize: 0,
    trainingMaxSize: 10000
};

Utils.static.tooltipOn = function(tooltip, event, object) {
    
    tooltip.object = object;
    tooltip.setAttribute("class", "");
    tooltip.style.left = (event.pageX+10) + "px";
    tooltip.style.top = (event.pageY+5) + "px";

    Utils.static.tooltipUpdate(object);
};

Utils.static.tooltipUpdate = function(tooltip, object) {

    if (typeof object !== "object") {
        tooltip.object = object;
        return;
    }

    var buffer = "";

    for (var key in object) 
        if (object.hasOwnProperty(key) && key !== "object")
            buffer += key + ": " + object[key] + "<br />";

    tooltip.innerHTML = buffer;
};    

Utils.static.tooltipOff = function(tooltip) {
    
    tooltip.object = undefined;
    tooltip.setAttribute("class", "off");
};

////////////

Utils.static.setTrainingSize = function(size) {

    Utils.trainingMaxSize = size;
};

Utils.static.addIntoTraining = function(inputs, targets) {

    // Build training data (as string) for future exportation
    if (Utils.trainingSize <= Utils.trainingMaxSize) {
        Utils.trainingData += inputs.join(" ") + " : " + targets.join(" ") + ";\\\n"; 
        Utils.trainingSize++;
        return true;
    }

    return false;
};

Utils.static.exportTrainingData = function() {

    console.info("Saving training data...", "Reading 'training_data'");

    var output = document.createElement("textarea");
    output.innerHTML = "var imported_training_set = \"" + Utils.trainingData + "\";";
    document.body.appendChild( output );

    return "Export completed for " + Utils.trainingSize + " entries.";
};

Utils.static.getTrainingData = function() {
    
    return Utils.trainingData;
};

Utils.static.clearTrainingData = function() {
    
    Utils.trainingData = "";
};

Utils.static.parseTrainingData = function(raw) {

    // Parse training data
    var i, l, entry, splitted = raw.split(";");
    var training_data = [], training_size;

    for (i = 0, l = splitted.length; i < l; i++)
    {
        entry = splitted[i].trim().split(":");
        if (entry.length !== 2)
            break;

        training_data.push({
            inputs: entry[0].trim().split(" ").map(parseFloat),
            targets: entry[1].trim().split(" ").map(parseFloat)
        });
    }

    return training_data;
};

////////////////////////////////// Neural Network core

function Neuron(id, layer, biais) {

    this.id = id;
    this.layer = layer;
    this.biais = biais || 0;
    this.biaisMomentum = 0;
    this.biaisGradient = 0;
    this.dropped = false;

    this.agregation = undefined;
    this.output = undefined;
    this.error = undefined;

    this.network = undefined; // link to its network, indispensable for special activation & derivation
    this.activation = undefined;
    this.derivative = undefined;

    // Input/output weights as cache (because Network.getWeight method is repeated a lot in feed and backprop, it takes time)
    this.inputWeightsIndex = undefined;
    this.outputWeightsIndex = undefined;
}

function Network(params) {

    // Required variables: lr, layers
    this.params = params;

    this.lr = undefined; // Learning rate
    this.layers = undefined;
    this.optimizer = undefined; // must bin included in _AVAILABLE_OPTIMIZER
    this.optimizerParams = undefined; // example: momentum rate will be {alpha: X}
    this.activation = undefined; // activation function for hidden layer
    this.activationParams = undefined;

    this.neurons   = undefined;
    this.weights   = undefined;
    this.momentums = undefined; // momentums coefficients a t-1
    this.gradients = undefined; // gradients squared for Adagrad 
    this.output    = undefined; // current output array

    // Caching variables:
    this.layersSum = undefined;
    this.layersMul = undefined;
    this.nbLayers  = undefined;
    this.nbNeurons = undefined;
    this.nbWeights = undefined;

    // Stats-purpose:
    this.iterations = 0;
    this.maxWeight = 0;
    this.outputError = 0;
    this.globalError = 0;
    this.avgWeightsPerNeuron = 0;

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

    // Necessary for avoiding problems with Cross Origin (Web Worker)
    this.libURI = undefined;

    this.loadParams(params);
    this.initialize();
}

Network.prototype.loadParams = function(params) {

    for (var key in params)
        if (this.hasOwnProperty(key) && this[key] === undefined)
            this[key] = params[key];

    console.log("Loaded params", this);    
};

Network.prototype.exportParams = function() {

    // Ensure to update params if they were modified on live
    for (var key in this.params)
        if (this.hasOwnProperty(key) && this[key] !== undefined)
            this.params[key] = this[key];

    return this.params;
};

Network.prototype.exportWeights = function() {
    return this.weights;
};

Network.prototype.importWeights = function(values) {
    
    this.weights = values;
    this.momentums.fill(0);
    this.gradients.fill(0);
    this.iterations = 0;
};

Network.prototype.exportBiais = function() {
    
    // We ensure to make a copy and not a reference here
    var values = Array(this.nbNeurons);

    for (var i = 0; i < this.nbNeurons; i++)
        values[i] = this.neurons[i].biais;

    return values;
};

Network.prototype.importBiais = function(values) {

    for (var i = 0; i < this.nbNeurons; i++) {
        this.neurons[i].biais = values[i];
        this.neurons[i].biaisMomentum = 0;
        this.neurons[i].biaisGradient = 0;
    }
};

Network.prototype.initialize = function() {

    if (this.libURI === undefined)
        throw new NetException("Undefined or invalid lib URI. Necessary for avoiding Cross Origin problems. Use https://domain.com/.../neural-net.js notation", {libURI: this.libURI});
 
    if (this.lr === undefined || this.lr <= 0)
        throw new NetException("Undefined or invalid learning rate", {lr: this.lr});

    if (this.layers === undefined || this.layers.length <= 1)
        throw new NetException("Undefined or unsufficient layers. At least, you must have a input and a output layer.", {layers: this.layers});

    if (this.optimizer !== undefined && !_AVAILABLE_OPTIMIZERS.includes(this.optimizer))
        throw new NetException("Invalid optimizer. Available optimizers = ", { available: _AVAILABLE_OPTIMIZERS, optimizer: this.optimizer });
        
    if ((this.optimizer === "momentum" || this.optimizer === "nag") && (this.optimizerParams === undefined || this.optimizerParams.alpha === undefined || this.optimizerParams.alpha < 0 || this.optimizerParams.alpha > 1))
        throw new NetException("Undefined or invalid momentum rate (must be between 0 and 1 both included) ", {optimizer: this.optimizer, optimizerParams: this.optimizerParams});
        
    if (this.optimizer === "adam" && (this.optimizerParams === undefined || this.optimizerParams.beta1 === undefined || this.optimizerParams.beta2 === undefined || this.optimizerParams.beta1 < 0 || this.optimizerParams.beta1 > 1 || this.optimizerParams.beta2 < 0 || this.optimizerParams.beta2 > 1)) 
        throw new NetException("Undefined or invalid (beta1,beta2) for Adam optimizer", {optimizer: this.optimizer, optimizerParams: this.optimizerParams});

    var i, j, l, sum, mul, tmp;
    var curr_layer = 0;

    // Initialization
    this.iterations = 0;
    this.nbLayers   = this.layers.length;
    this.layersSum  = [];
    this.layersMul  = [];
    this.neurons    = [];
    this.weights    = [];
    this.momentums  = [];
    this.gradients  = [];

    // Prepare layers relative computation
    for (i = 0, sum = 0, mul = 1; i < this.nbLayers; i++) {
        sum += this.layers[i];
        mul = (this.layers[i-1] || 0) * this.layers[i];
        this.layersSum.push(sum);
        this.layersMul.push(mul + (this.layersMul[i-1] || 0)); 
        // [0] will be 0, Because layerMul is used to know how many weights there is before a layer, and there is no before layer 0
    }
    
    // Compute and put lengths in cache
    this.nbNeurons = sum;
    this.nbWeights = this.layersMul[this.layersMul.length-1];
    this.avgWeightsPerNeuron = this.nbWeights / this.nbNeurons;
    
    // Create weights, momentum and gradients
    for (i = 0; i < this.nbWeights; i++) {
        this.weights[i] = this.static_randomWeight();
        this.momentums.push(0);
        this.gradients.push(0);
    }

    // Create neurons
    var index, neuron, prev_neurons = [], next_neurons = [];

    for (curr_layer = 0, i = 0; i < this.nbNeurons; i++)
    {
        neuron = new Neuron(i, i >= this.layersSum[curr_layer] ? ++curr_layer : curr_layer, this.static_randomBiais());
        neuron.network = this;
        neuron.activation = this.static_linearActivation;
        neuron.derivative = this.static_linearDerivative;
        this.neurons.push(neuron);
    }

    // Set hidden layer activation functions 
    // (separated from loop above because we don't want input and output layers to have an activation function -by default)
    switch (this.activation) {
        case "tanh":
            this.setHiddenLayerToActivation(this.static_tanhActivation, this.static_tanhDerivative);
            break;

        case "sigmoid":
            this.setHiddenLayerToActivation(this.static_sigmoidActivation, this.static_sigmoidDerivative);
            break;

        case "relu":
            this.setHiddenLayerToActivation(this.static_reluActivation, this.static_reluDerivative);
            break;

        case "prelu":
            this.setHiddenLayerToActivation(this.static_preluActivation, this.static_preluDerivative);
            break;

        default:
            this.setHiddenLayerToActivation(this.static_linearActivation, this.static_linearDerivative);
    }

    // 1- Assign weights index into neuron's cache
    // 2- Improve the weight initialization by ensuring that the variance is equal to 1
    for (curr_layer = -1, i = 0; i < this.nbNeurons; i++)
    {
        neuron = this.neurons[i];

        if (neuron.layer !== curr_layer) {
            curr_layer++;
            prev_neurons = curr_layer > 0 ? this.getNeuronsInLayer(curr_layer-1) : [];
            next_neurons = curr_layer < this.nbLayers-1 ? this.getNeuronsInLayer(curr_layer+1) : [];
        }

        neuron.inputWeightsIndex = Array(prev_neurons.length);        
        neuron.outputWeightsIndex = Array(next_neurons.length);       

        // Input weights
        for (j = 0, l = prev_neurons.length; j < l; j++) {
            neuron.inputWeightsIndex[j] = this.getWeightIndex(prev_neurons[j], neuron);
            this.weights[neuron.inputWeightsIndex[j]] *= Math.sqrt(2 / l);
        }

        // Output weights
        for (j = 0, l = next_neurons.length; j < l; j++)
             neuron.outputWeightsIndex[j] = this.getWeightIndex(neuron, next_neurons[j]);
    }  

    // Initialize brain.output to zeros, to avoid training problems
    this.output = Array(this.layers[this.nbLayers - 1]);
    this.output.fill(0);

    // Display the complexity of this new NN (weights + biais)
    var parameters = this.weights.length + this.nbNeurons;
    console.info("This neural network has %d parameters.", parameters);
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

    function neuronTooltipOn(event) {
        Utils.static.tooltipOn( that.DOM.tooltip, event, that.neurons[event.target.getAttribute("data-object")] );
    } 

    function neuronTooltipOff(event) {
        Utils.static.tooltipOff( that.DOM.tooltip );
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
        DOM_tmp.addEventListener("mousemove", neuronTooltipOn);
        DOM_tmp.addEventListener("mouseout", neuronTooltipOff);

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

Network.prototype.visualize = function(inputs, precision) {


    if (!this.svgVisualization)
        throw new NetException("SVG Visualization is not available", {network: this});

    if (!inputs || inputs.length !== this.layers[0])
        throw new NetException("Incorrect inputs (undefined or incorrect length)", {inputs: inputs, layer: this.layers[0]});

    var i, l;
    var output_neurons = this.getNeuronsInLayer( this.nbLayers-1 );
    precision = precision || 1;

    // Update SVG text inputs
    for (i = 0, l = this.DOM.inputTexts.length; i < l; i++)
        this.DOM.inputTexts[i].innerHTML = inputs[i].toFixed(precision);

    // Update SVG text outputs
    for (i = 0, l = this.DOM.outputTexts.length; i < l; i++)
        this.DOM.outputTexts[i].innerHTML = output_neurons[i].output ? output_neurons[i].output.toFixed(precision) : output_neurons[i].output;

    // Update SVG weights
    for (i = 0, l = this.nbWeights; i < l; i++) {
        this.DOM.weightCurves[i].setAttribute("stroke-width", Math.abs(this.weights[i]) / this.maxWeight * _SVG_STROKE_WIDTH);
        if (this.avgWeightsPerNeuron < _SVG_MAX_WEIGHTS_DISPLAY_TEXT) 
            this.DOM.weightTexts[i].innerHTML = this.weights[i].toFixed(4);
    }

    // Update tooltip
    if (this.DOM.tooltip.object !== undefined)
        Utils.static.tooltipUpdate(this.DOM.tooltip, this.DOM.tooltip.object);
};

Network.prototype.feed = function(inputs) {

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
        for (sum = 0, n = 0, l = prev_neurons.length; n < l; n++)
            if (!prev_neurons[n].dropped)
                sum += this.weights[neuron.inputWeightsIndex[n]] * prev_neurons[n].output;

        // Updating output    
        neuron.agregation = sum + neuron.biais;
        neuron.output = neuron.activation(neuron.agregation); 

        if (!isFinite(neuron.output))
            throw new NetException("Non finite or too high output. You may have a problem in your code", {neuron: neuron});
    }

    // Update network output
    var neurons = this.getNeuronsInLayer(this.nbLayers-1);
    for (n = 0, l = this.layers[this.nbLayers-1]; n < l; n++)
        this.output[n] = neurons[n].output;            

    // Return output neurons
    return neurons;
};

Network.prototype.loss = function(targets) {
    
    var outputs_neurons = this.getNeuronsInLayer(this.nbLayers - 1);

    if (!targets || !outputs_neurons || targets.length !== outputs_neurons.length)
        throw new NetException("Incoherent targets for current outputs", { targets: targets, outputs_neurons: outputs_neurons });

    // Compute output error with our loss function
    // https://en.wikipedia.org/wiki/Backpropagation

    var n, l, neuron;
    this.outputError = 0;

    // Output layer filling: err = (expected-obtained)
    for (n = 0, l = outputs_neurons.length; n < l; n++) {
        neuron = outputs_neurons[n];
        neuron.error = (targets[n] - neuron.output) * neuron.derivative(neuron.agregation);
        this.outputError += 1 / 2 * neuron.error * neuron.error;

        if (!isFinite(neuron.error))
            throw new NetException("Non finite error on output neuron. You may have a problem in your code", { neuron: neuron });
    }
};

Network.prototype.backpropagate = function(targets) {

    // Compute current output error with our loss function 
    this.loss(targets);

    var index, weight_index, n, l, sum, calc, grad, weight, max_weight = 0;
    var output_error = 0, curr_layer = this.nbLayers - 1;
    var neuron, next_neurons;

    this.iterations++; // need to be 1 in first for Adam computing
    this.globalError = 0;

    // Fetching neurons from last layer: backpropagate error & update weights
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
                sum += this.weights[neuron.outputWeightsIndex[n]] * next_neurons[n].error;
        }

        // Updating error    
        neuron.error = sum * neuron.derivative(neuron.agregation); 
        this.globalError += Math.abs(neuron.error); 
        
        if (!isFinite(neuron.error)) {
            throw new NetException("Non finite error. You may have a problem in your code", {neuron: neuron});
        } else if (Math.abs(neuron.error) > _ERROR_VALUE_TOO_HIGH) {
            console.info("Scaling down error to a max", {neuron: neuron, error: neuron.error});
            neuron.error = neuron.error < 0 ? - _ERROR_VALUE_TOO_HIGH : _ERROR_VALUE_TOO_HIGH;
            throw new NetException("Computed error is too high. Try a smaller learning rate?", {neuron: neuron});
        }

        // Updating weights w = w + lr * en * output
        for (n = 0, l = next_neurons.length; n < l; n++)
        {
            if (next_neurons[n].dropped)
                continue;

            weight_index = neuron.outputWeightsIndex[n]; 

            // Compute new values w.r.t gradient optimizer
            grad = next_neurons[n].error * neuron.output;
            calc = this.optimizeGradient(this.weights[weight_index], grad, this.momentums[weight_index], this.gradients[weight_index]);
            
            // Updates values
            this.weights[weight_index] = weight = calc.value;
            this.momentums[weight_index] = calc.momentum;
            this.gradients[weight_index] = calc.gradients;

            // Update maxWeight (for visualisation)
            max_weight = max_weight < Math.abs(weight) ? Math.abs(weight) : max_weight;

            if (!isFinite(weight)) {
                throw new NetException("Non finite weight. You may have a problem in your code", {neuron: neuron, weight: weight});
            } else if (Math.abs(weight) > _WEIGHT_VALUE_TOO_HIGH) {
                console.info("Scaling down weight to a max.", {neuron: neuron, weight: weight});
                weight = weight < 0 ? - _WEIGHT_VALUE_TOO_HIGH : _WEIGHT_VALUE_TOO_HIGH;
            }
        }

        // Compute biais with gradient optimizer
        grad = neuron.error;
        calc = this.optimizeGradient(neuron.biais, grad, neuron.biaisMomentum, neuron.biaisGradient);

        // Updates values
        neuron.biais = calc.value;
        neuron.biaisMomentum = calc.momentum;
        neuron.biaisGradient = calc.gradients;

        if (!isFinite(neuron.biais))
            throw new NetException("Non finite biais. You may have a problem in your code", {neuron: neuron});
    }

    this.maxWeight = max_weight;
};

Network.prototype.optimizeGradient = function(value, grad, momentum, gradients) {

    var p = this.optimizerParams, prev_momentum = momentum;

    if (value === undefined || grad === undefined || momentum === undefined || gradients === undefined)
        throw new NetException("Invalid parameters for gradient optimization", { value: value, grad: grad, momentum: momentum, gradients: gradients });

    // Momentum helps to escape local minimums, 
    // Nesterov accelerated gradient is smarter than momentum because inertia is predicted
    // Adagrad aims to automatically decrease the learning rate 
    // Adadelta correct the too aggressive learning rate reduction of Adagrad

    switch (this.optimizer)
    {
        case "momentum":
            momentum = (1 - p.alpha) * this.lr * grad + p.alpha * momentum;
            value += momentum;
            break;
        
        case "nag":
            momentum = p.alpha * momentum + (1 - p.alpha) * this.lr * grad;
            value += -p.alpha * prev_momentum + (1 + p.alpha) * momentum;
            break;

        case "adagrad":
            gradients += grad * grad; // this contains the sum of all past squared gradients
            value += this.lr * grad / (Math.sqrt(gradients) + _EPSILON);
            break;

        case "adadelta":
            gradients = p.alpha * gradients + (1 - p.alpha) * grad * grad; // this contains the decaying average of all past squared gradients
            value += this.lr * grad / (Math.sqrt(gradients) + _EPSILON);
            break;

        case "adam":
            momentum = p.beta1 * momentum + (1 - p.beta1) * grad;
            gradients = p.beta2 * gradients + (1 - p.beta2) * grad * grad;
            
            var mt = momentum / (1 - Math.pow(p.beta1, this.iterations)); // momentum biais correction
            var gt = gradients / (1 - Math.pow(p.beta2, this.iterations)); // gradients biais correction

            value += this.lr * mt / (Math.sqrt(gt) + _EPSILON);
            break;

        default: // good-old vanilla SGD
            value += this.lr * grad;
    }

    return { value: value, grad: grad, momentum: momentum, gradients: gradients };
};

Network.prototype.dropout = function(completely_random, drop_inputs) {

    // Dropping out random neurons allows to push out our network of a bad solution
    // If completely_random === true, the same neuron can be dropped again. 
    // We usually start from first hidden layer, but could be possible to start from inputs layer if drop_inputs === true

    var i, l, n, neurons, shot;
    completely_random = typeof completely_random === "undefined" ? true : completely_random;

    for (i = drop_inputs === true ? 0 : 1; i < this.nbLayers-1; i++)
    {
        neurons = this.getNeuronsInLayer(i);
        shot = completely_random ? undefined : Math.round( Math.random() * (this.layers[i] - 1) );

        for (n = 0, l = neurons.length; n < l; n++)
        {
            if (shot === n || (shot === undefined && Math.random() >= _DROPOUT_PROBABILITY))
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

Network.prototype.validate = function (params) {

    if (!params)
        throw new NetException("Invalid parameters object for validation", { params: params });

    params.backpropagate = false;
    params.epochs = 1;
    params.dropout = false;

    return this.train(params);
};

Network.prototype.train = function(params) {

    if (!params)
        throw new NetException("Invalid parameters object for training", {params: params});

    var training_data = params.trainingSet || undefined;
    var validation_data = params.validationSet || [];
    var test_data = params.testSet || [];

    var epochs = params.epochs || undefined;

    if (!training_data || training_data.length <= 0)
        throw new NetException("Invalid raw training data (object)", {training_data: training_data});

    if (!epochs || isNaN(epochs))
        throw new NetException("Invalid epochs number for training", {epochs: epochs});
        
    if (typeof window.Worker === "undefined" || !window.Worker)
        throw new NetException("Web Worker is not supported by your client. Please upgrade in order to train as background operation");
        
    // Important to register these here (accessible in worker callback)
    var training_size = training_data.length;
    var validation_size = validation_data.length;
    var test_size = test_data.length;
    var gather_all = epochs * training_size <= _TRAINING_GATHER_ALL_THRESHOLD;

    console.info("Training: trying to handle %d extracted inputs/targets", training_size);
    console.info("Validation: trying to handle %d extracted inputs/targets", validation_size);
    console.info("Test: trying to handle %d extracted inputs/targets", test_size);

    // Create visualisation (these one are also behond the scope)
    var container, graph, graph_ctx, text_output;
    var graph_width = params.graphWidth ? params.graphWidth : _CANVAS_GRAPH_DEFAULT_WIDTH;
    var graph_height = params.graphHeight ? params.graphHeight : _CANVAS_GRAPH_DEFAULT_HEIGHT;
    var scaled_width;

    if (params.visualize === true)
    {
        // Create canvas
        container = document.createElement("div");
        container.setAttribute("style", "margin: 10px;");

        graph = document.createElement("canvas");
        graph.setAttribute("width", graph_width);
        graph.setAttribute("height", graph_height);
        container.appendChild( graph );

        // Create global error mean output
        text_output = document.createElement("samp");
        container.appendChild( text_output ); 

        // We don't want to display too much data futilely
        if (gather_all)
            scaled_width = graph_width / (epochs * training_data.length);
        else
            scaled_width = graph_width / epochs;

        graph_ctx = graph.getContext("2d");
        graph_ctx.translate(0, graph_height);
        graph_ctx.scale(scaled_width, - graph_height);
        // graph_ctx.scale(1, - _CANVAS_GRAPH_HEIGHT);
        graph_ctx.globalAlpha = 0.8;
        graph_ctx.lineWidth = 0.03;

        // Following functions will be called in our requestAnimFrame
        var display_curves = function (data, window_width, fill, stroke, fill_style, stroke_style)
        {
            if (!data || data.length === 0)
                return;

            var ratio = window_width / (data.length-1);
            var l = data.length;

            graph_ctx.fillStyle = fill_style;
            graph_ctx.strokeStyle = stroke_style;
            graph_ctx.beginPath();
            graph_ctx.moveTo(0, 0);

            for (var i = 0; i < l; i++)
                graph_ctx.lineTo(i * ratio, Math.sqrt(data[i] + _EPSILON) * _CANVAS_GRAPH_WINDOW_FACTOR);

            if (fill) {
                // graph_ctx.lineTo(i * ratio, Math.sqrt(data[i-1] + _EPSILON) * _CANVAS_GRAPH_WINDOW_FACTOR);
                graph_ctx.lineTo((i-1) * ratio, 0);
                graph_ctx.closePath();
                graph_ctx.fill();
            }

            if (stroke) {
                graph_ctx.stroke();
                graph_ctx.closePath();
            }
        };

        var Stats = function (losses, epoch_mean_loss, global_mean_loss) {

            this.size = losses.length;
            this.losses = losses;
            this.epoch_mean_loss = epoch_mean_loss;
            this.global_mean_loss = global_mean_loss;
        };
    }

    //////////////// Worker below ////////////////////////////

    var blob = new Blob(['(' + this.workerHandler.toString() + ')()' ], { type: "text/javascript" });
    var worker = new Worker(window.URL.createObjectURL(blob));
    var that = this;

    worker.addEventListener("message", function(e) {
        
        if (typeof e.data.type === "undefined")
            throw new NetException("Worker message needs to contain message type (WORKER_TRAINING_X)", {data: e.data});

        // Training is over for the current epoch: we display our losses
        if (e.data.type === _WORKER_TRAINING_PENDING)
        {
            if (params.visualize !== true)
                return;

            window.requestAnimationFrame(function() {

                var training = new Stats(e.data.training_stats.losses, e.data.training_stats.epoch_mean_loss, e.data.training_stats.global_mean_loss);
                var validation = new Stats(e.data.validation_stats.losses, e.data.validation_stats.epoch_mean_loss, e.data.validation_stats.global_mean_loss);
                var test = new Stats(e.data.test_stats.losses, e.data.test_stats.epoch_mean_loss, e.data.test_stats.global_mean_loss);

                var smooth_size = graph_width * _CANVAS_GRAPH_SMOOTH_FACTOR;

                ////////////////////////////

                graph_ctx.clearRect(0, 0, graph_width / scaled_width, 1);

                // Graphically separate epochs (only with a small amount of epochs)
                if (epochs <= _CANVAS_GRAPH_SEPARATE_EPOCHS_THRESHOLD) {
                    graph_ctx.fillStyle = "#c7cbe0";
                    for (var i = 1; i < epochs; i++)
                        graph_ctx.fillRect(i * graph_width / scaled_width / epochs, 0, 1 / scaled_width, 1);
                }
                
                // Display the training set losses curve
                display_curves(training.losses.average(graph_width), training.size, true, false, _COLOR_ASPHALT, _COLOR_BLUE);
                
                // Display smoother mean if necessary
                if (gather_all)
                    display_curves(training.losses.average(graph_width * _CANVAS_GRAPH_SMOOTH_FACTOR), training.size, false, true, _COLOR_ASPHALT, _COLOR_BLUE);

                // Display the validation set and test set smoothly 
                display_curves(validation.losses.average(graph_width), training.size, false, true, "pink", _COLOR_PURPLE);
                display_curves(test.losses.average(graph_width), training.size, false, true, "pink", _COLOR_GREEN);

                // Update output text display
                text_output.innerHTML = "epoch " + (e.data.curr_epoch+1) + "/" + epochs + " | curr error mean: " + training.epoch_mean_loss.toFixed(5);
            });
        }

        // Training is over : we update our weights an biais
        else if (e.data.type === _WORKER_TRAINING_OVER)
        {
            that.importWeights( e.data.weights );
            that.importBiais( e.data.biais );

            // Feeding and bring in order to have updated values (as error) into neurons or others
            that.feed( training_data[0].inputs );
            that.loss( training_data[0].targets );

            // Free space
            training_data = null;
            validation_data = null;
            test_data = null;
            worker.terminate();
        }
    });

    // Start web worker with training data through epochs
    worker.postMessage({
        lib: this.libURI,
        params: this.exportParams(),
        weights: this.exportWeights(),
        biais: this.exportBiais(),

        trainingData: training_data,
        validationData: validation_data, 
        testData: test_data, 

        epochs: epochs,
        options: {
            backpropagate: params.backpropagate !== undefined ? params.backpropagate : _DEFAULT_TRAINING_BACKPROPAGATE,
            dropout: params.dropout !== undefined ? params.dropout : _DEFAULT_TRAINING_DROPOUT,
            shuffle: params.shuffle !== undefined ? params.shuffle : _DEFAULT_TRAINING_SHUFFLE
        }
    });

    // You can disable worker (for exemple: analyze peformance thanks to developement utils)
    // this.disabledWorkerHandler({
    //     ... same params ...
    // });

    return container || null;
};

Network.prototype.workerHandler = function() {

    // Inside onmessage here's the core training, what will be executed by our webworker
    onmessage = function(e) {
        
        if (typeof importScripts !== "undefined")
            importScripts(e.data.lib);

        if (!e.data.lib || !e.data.params || !e.data.weights)
            throw new NetException("Invalid lib_url, params or weights in order to build a Neural Network copy", {lib: e.data.lib, params: e.data.params, weights: e.data.weights});

        var epochs = e.data.epochs;
        var training_data = e.data.trainingData;
        var validation_data = e.data.validationData;
        var test_data = e.data.testData;
        var options = {
            backpropagate: e.data.options.backpropagate,
            dropout: e.data.options.dropout,
            shuffle: e.data.options.shuffle
        };

        console.info("Training imported data in processing... "+ epochs + "requested. Options: ", options);
        console.info("Brain copy below:");    

        // Create copy of our current Network
        var brain = new Network(e.data.params);
        brain.importWeights(e.data.weights);
        brain.importBiais(e.data.biais);

        ///////////////////// Training - validation - test  //////////////////////////////

        var datasetHandler = function(data) {

            this.data = data;
            this.size = data.length;

            this.losses = [];
            this.lossesMean = 0;
            this.lossesSum = 0;
            this.globalLossesSum = 0;

            this.epochMeanLoss = undefined;
            this.globalMeanLoss = undefined;
        };

        datasetHandler.prototype.fetch = function(options, backpropagate) {

            // At a threshold, we only collect back the mean of every epoch. It enhance display performance (on the canvas)
            // and avoid passing oversized arrays back to the main thread
            var gather_all = epochs * this.size <= _TRAINING_GATHER_ALL_THRESHOLD;

            // Shuffling data can improve learning
            if (options.shuffle === true)
                this.data = this.data.shuffle();

            // Feeforward NN thought the training dataset
            for (this.lossesSum = 0, i = 0; i < this.size; i++) {
                try {
                    brain.feed(this.data[i].inputs);

                    if (backpropagate === false)
                        brain.loss(this.data[i].targets);
                    else
                        brain.backpropagate(this.data[i].targets);
                }

                catch (ex) {
                    console.error(ex);
                    return false;
                }

                this.lossesSum += brain.outputError;

                // Display every loss of every epochs
                if (gather_all)
                    this.losses.push(brain.outputError);
            }

            this.globalLossesSum += this.lossesSum;
            this.epochMeanLoss = this.lossesSum / this.size;
            this.globalMeanLoss = this.globalLossesSum / ((curr_epoch + 1) * this.size); 

            // Display the loss mean for every epoch
            if (!gather_all)
                this.losses.push(this.epochMeanLoss);

            return true;
        };

        var i, n, curr_epoch;

        var training_handler = new datasetHandler(training_data);
        var validation_handler = new datasetHandler(validation_data);
        var test_handler = new datasetHandler(test_data);

        // Variables that will store means of the current training in order to fire a dropout if requested. See below the dropout execution
        var last_means = [];
        var last_means_sum = 0;

        // Repeat the feedforward & backpropagation process for 'epochs' epochs
        for (curr_epoch = 0; curr_epoch < epochs; curr_epoch++)
        {
            // Train by using the training set
            if (!training_handler.fetch(options, options.backpropagate))
                return;

            // Feed the NN with the validation set
            if (!validation_handler.fetch(options, false))
                return;

            // Feed the NN with the test set
            if (!test_handler.fetch(options, false))
                return;

            options.dropout = options.dropout === true ? _TRAINING_DROPOUT_EPOCHS_THRESHOLD : options.dropout;

            // Introducing dynamic dropout every "options.dropout" epochs,
            // if the mean difference is below _TRAINING_DROPOUT_MEAN_THRESHOLD
            if (options.dropout !== false)
            {
                last_means_sum += training_handler.epochMeanLoss;
                last_means.push( training_handler.epochMeanLoss );
    
                if (last_means.length >= options.dropout)
                {
                    last_means_sum -= last_means.shift();
                    var local_mean = last_means_sum / options.dropout; 
                   
                    if (local_mean - training_handler.epochMeanLoss <= _TRAINING_DROPOUT_MEAN_THRESHOLD) {
                        console.info("EVENT: Dropout at epoch #%d", curr_epoch);
                        brain.dropout(false);
                        last_means = [];
                        last_means_sum = 0;
                    }
                }
            }

            // Send updates back to real thread
            self.postMessage({
                type: _WORKER_TRAINING_PENDING,
                curr_epoch: curr_epoch,

                training_stats:  {
                    losses: training_handler.losses,
                    epoch_mean_loss: training_handler.epochMeanLoss,
                    global_mean_loss: training_handler.globalMeanLoss,
                },

                validation_stats: {
                    losses: validation_handler.losses,
                    epoch_mean_loss: validation_handler.epochMeanLoss,
                    global_mean_loss: validation_handler.globalMeanLoss,
                },

                 test_stats: {
                    losses: test_handler.losses,
                    epoch_mean_loss: test_handler.epochMeanLoss,
                    global_mean_loss: test_handler.globalMeanLoss,
                }
            });
        }

        console.info("Training done. Gone through all epochs", {epochs: epochs, global_mean_loss: training_handler.global_mean_loss});

        self.postMessage({
            type: _WORKER_TRAINING_OVER,
            weights: brain.exportWeights(),
            biais: brain.exportBiais()
        });

        self.close();
    };

    return onmessage; // allows fallback for Network.disabledWorkerHandler
};

Network.prototype.disabledWorkerHandler = function(data) {
    
    if (!data)
        throw new NetException("Invalid data for disabledWorkerHandler", {data: data});

    // Override self.postMessage (doesn't exist outside of webWorker, but we create it here to avoid error and to monitor what's happening)
    self.postMessage = function(data) {
        console.log(data);
    };

    this.workerHandler()({data: data});
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

    if ((!layer && layer !== 0) || layer < 0 || layer >= this.nbLayers)
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

Network.prototype.setWeight = function(from, to, value) {

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


/////////////////////////// Statics network methods & activation functions

Network.prototype.static_randomBiais = function() {
    return Math.uniform() * _BIAIS_RANDOM_COEFF;
};

Network.prototype.static_randomWeight = function() {
    return Math.uniform() * _WEIGHT_RANDOM_COEFF;
};

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
    return this.network.static_sigmoidActivation(x) * (1 - this.network.static_sigmoidActivation(x));
};

Network.prototype.static_reluActivation = function(x) {
    return x < 0 ? 0 : x;
};

Network.prototype.static_reluDerivative = function(x) {
    return x < 0 ? 0 : 1;
};

Network.prototype.static_preluActivation = function(x) {
    return x < 0 ? this.network.activationParams.alpha * x : x;
};

Network.prototype.static_preluDerivative = function(x) {
    return x < 0 ? this.network.activationParams.alpha : 1;
};
    
/////////////////////////// Network Exception

function NetException(message, variables) {
    console.error("ERROR: " + message, variables);
}

Array.prototype.hash = function() {
    return { hash: btoa(this.join()), size: this.length };
};

Array.prototype.shuffle = function() {
    
    var j, x, i;

    for (i = this.length - 1; i > 0; i--) {
        j = Math.floor(Math.random() * (i + 1));
        x = this[i];
        this[i] = this[j];
        this[j] = x;
    }

    return this;
};

Array.prototype.average = function(size) {

    if (size >= this.length)
        return this;

    var ratio = this.length / size;
    var index, i, j, l = this.length, n = Math.ceil(ratio);
    var sum, last_sum = 0, mean, avgs = [];

    for (i = 0; i < size; i++)
    {
        index = index = Math.floor(i * ratio);
        sum = 0;

        for (j = 0; j < n && index+j < l; j++)
            sum += this[index + j];
        
        avgs.push((sum + last_sum) / (n * 2));
        last_sum = sum;
    }
    
    return avgs;
};

Math.uniform = function() {
    return ((Math.random() + Math.random() + Math.random() + Math.random() + Math.random() + Math.random()) - 3) / 3;
};