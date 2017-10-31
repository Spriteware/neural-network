
////////////// Neural network initialization ////////////


var _params = {
        
    // Since there is a webWorker inside, we need to provide the local URI.
    // This is used for our WebWorker to include the whole library into itself.
    // You can provide a distant URL w
    libURI: "http://localhost/neural-network.js",

    // The learning rate factor is really important. Try a few ones to get the right one.
    // The more you have layers and neurons, the more you need to specify a small one
    // Currently, momentum is not working. Specify it as 0
    lr: 0.005,          
    momentum: 0,
    
    // The topology of our neural network. Ex: 4 input neurons, 1 hidden layer (3 neurons) and 2 output neurons 
    layers: [4, 3, 2],  
    // Be creative: (bur remember to whatch the console and to adapt your learning rate)
    // layers: [2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 2],
    // layers: [5, 4, 3, 2, 1],
    // layers: [2, 40, 2]

    // Activation function used for the hidden layers. Inputs and outputs neurons have a linear activation function
    // If not specified, "linear" is de default value. activationParams are just factors that impacts some activation function (etc: PReLu)
    // Currently possible values:
    // linear, sigmoid, tanh, relu, prelu 
    activation: "linear",
    activationParams: {alpha: 0.1}
};

// Initialize the neural network
var brain = new Network(_params);

// NeuralNetwork.createVisualization() will returns the whole SVG visualisation as a DOM element
// As an example, you can directly append it to your current document
document.body.appendChild( brain.createVisualization() );  

// If you provide a training dataset, you can specify it to the Neural Net and train it with it
if (typeof training_data_imported !== 'undefined' && training_data_imported !== undefined)
{
    // The epochs is 'how many times do you want to feed the same training dataset'
    var _epochs = 500;

    // NeuralNetwork.train accept a string which contains your whole dataset, and returns a DOM object for visualizating the training.
    // Please see Utils.static.xxxx to know how to produce a dataset, and how are formatted a dataset.
    // You can also visit the examples on franpapers.com to see what a dataset looks like
    // It accept something that looks like a CSV file.
    // You can deactivate visualization if you don't need it 

    var training_visu = brain.train({
        data: training_data_imported,
        epochs: _epochs,
        visualize: true
    });

    // Add the current training visualization to your document
    document.body.appendChild(training_visu);
}


////////////// Neural network core ////////////


// Be careful to have normalized inputs and targets. If not you will see the errors jumping
// Here is a inputs/targets example of values
var inputs = [0.3, 0.5, 0.1, 0.9];  
var targets = [0, 1];      

// NeuralNetwork.feed() returns the output layer
var output_neurons = brain.feed(inputs);
brain.backpropagate(targets);

// And that's it ! For sure, if you already trained your NN,
// maybe you don't have to use NeuralNetwork.backpropagation() anymore

