# A up-to-date implementation of a Neural Network in JavaScript 

> This library provides neural networks models that can be trained and
> visualized with various optimizers, into its own thread. You can see
> an example of what it does here:   
> **Article:** https://franpapers.com/en/2017-neural-network-implementation-in-javascript-by-an-example/  
> **Video:** https://www.youtube.com/watch?v=tIdTulicm9M  

Summary:

 1. [Introduction](https://github.com/Spriteware/neural-network#1--introduction) 
 2. [Basic example](https://github.com/Spriteware/neural-network#2--basic-example)
 3. [Data example](https://github.com/Spriteware/neural-network#3--data-example)
 4. [What's next](https://github.com/Spriteware/neural-network#4--whats-next)

## 1 • Introduction

The main idea was to do something that can **help to visualize the network and its evolutuion** through backpropagation. Here's an example:

![Visualization example](https://franpapers.com/wp-content/uploads/2017/10/Capture.png)

**SVG** is used to provide you a clean visualisation, and a **simple Web Worker** is used for the training part (for avoiding blocking UI thread).

This library was not meant for "distribution" purpose, so it may have thousand bugs and may be not working as you want. **Fork it !**

## 2 • Basic example
It's simple, there are just way too much comments
```javascript


////////////// Neural network initialization ////////////


var _params = {
        
    // Since there is a webWorker inside, we need to provide the local URI of the script itself.
    // This is used for our WebWorker to include the whole library into itself.
    // You can provide a distant or local URL
    libURI: "http://localhost/neural-network.js",

    // The learning rate factor is really important. Try a few ones to get the right one.
    // The more you have layers and neurons, the more you need to specify a small one
    lr: 0.05,          
    
    // The topology of our neural network. The library can handle large number of neurons, it will just be slow.
    // Ex: 4 input neurons, 1 hidden layer (3 neurons) and 2 output neurons 
    layers: [4, 3, 2],  
    // Be creative: (bur remember to whatch the console and to adapt your learning rate)
    // layers: [2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 2],
    // layers: [2, 150, 2]

    // Activation function used for the hidden layers. Inputs and outputs neurons have a linear activation function
    // If not specified, "linear" is the default value. activationParams are just factors that impacts some activation function (etc: PReLu)
    // Currently possible values:
    // linear, sigmoid, tanh, relu, prelu 
    activation: "linear",
    activationParams: {alpha: 0.1}, // since the linear activation doesn't depends upon any variable, this property is useless here
    
    // Optimizers used. The library currently implements Momentum, Nesterov Accelerated Gradient, AdaGrad, AdaDelta, AdaM
    // Best results are found with 'nag' and 'adam'
    optimizer: "nag", // momentum, nag, adagrad, adadelta or adam
    optimizerParams: { alpha: 0.7, beta1: 0.9, beta2: 0.99 }, // alpha for nag and adadelta, betas for adam
};

// Initialize the neural network
var brain = new Network(_params);

// NeuralNetwork.createVisualization() will returns the whole SVG visualisation as a DOM element
// As an example, you can directly append it to your current document
document.body.appendChild( brain.createVisualization() );  

// If you provide a training dataset, you can specify it to the Neural Net and train it with it
if (typeof _imported_training_set !== 'undefined' && _imported_training_set !== undefined)
{
    // The epochs is 'how many times do you want to feed the same training dataset'
    var _epochs = 500;

    // NeuralNetwork.train accept a string which contains your whole dataset, and returns a DOM object for visualizating the training.
    // It accept something that looks like a CSV file.
    // You can also visit the examples on franpapers.com to see what a dataset looks like
    
    var training_set = typeof _imported_training_set !== "undefined" ? Utils.static.parseTrainingData(_imported_training_set) : undefined;
    var validation_set = typeof _imported_validation_set !== "undefined" ? Utils.static.parseTrainingData(_imported_validation_set) : undefined;
    var test_set = typeof _imported_test_set !== "undefined" ? Utils.static.parseTrainingData(_imported_test_set) : undefined;

    // Launch training
    var graph = _brain.train({
        training_set: training_set,
        validation_set: validation_set,
        test_set: test_set,

        epochs: _epochs,
        dropout: false,    // Automatic dropout
        shuffle: _shuffle, // Shuffle the training set at each epoch
        visualize: true    // If false, it doesn't return the DOM element
    });

    // Add the current training visualization to your document
    document.body.appendChild(graph);
}


////////////// Neural Network Core ////////////


// Be careful to have normalized inputs and targets. If not you will see the errors jumping
// Here is a inputs/targets example of values
var inputs = [0.3, 0.5, 0.1, 0.9];  
var targets = [0, 1];      

// NeuralNetwork.feed() returns the output layer
var output_neurons = brain.feed(inputs);
brain.backpropagate(targets);

// And that's it ! For sure, if you already trained your NN,
// You don't have to use NeuralNetwork.backpropagation() anymore


```

## 3 • Training/validation/test dataset example 
The datasets looks like a CSV file, but with a few differencies. Every `inputs/targets` couple is separated by a `;`. `inputs `and `targets` are separated by a `:`. `inputs` or `targets` values are seperated by a `space`. Here is an example:
```
input1 input2 input3 input4 : target1 target2 ;
input1 input2 input3 input4 : target1 target2 ;
```

Usually I save everything into a JS variable, and I put the JS script as a normal script in my page. This is why you can see this "_imported_training_set" variable in the code above.
```javascript
var _imported_training_set = "0 0.004012032080192407 0 0.004012032080192407 : 6.123233995736767e-17 -1;\
0.003901877324468339 0.0009301870879891982 0.00034437034884393647 -0.0009193694393909713 : -0.9726239602750568 -0.23238466364815152;\
0.003968835720993713 0.0005847808693595358 0.00006695839652537394 -0.0003454062186296625 : -0.9892652288476347 -0.14613112944556783;\
0.00380007257485393 0.0012799354131641794 -0.00016876314613978316 0.0006951545438046436 : -0.9473190861218562 0.32029135028790773;\
";
```

### Helpers for creating a training dataset 
There are a few helpers functions if you want to generate your dataset meanwhile you train (or not) your neural network. They are under `Utils.static.xxx` and you can call them in your script or your console.
```javascript

// Add inputs and associated targets in your training dataset. 
// Both parameters are arrays. That will contribute to create the dataset as a string
Utils.static.addIntoTraining(inputs, targets);

// Get the created dataset as a string
var training_data_imported = Utils.static.getTrainingData(inputs, targets);

// Or if you want to export it directly in the document (there's an 'document.body.appendChild' inside)
Utils.static.exportTrainingData();

// Clear the current dataset
Utils.static.clearTrainingData();
```

## 4 • What's next?

And that's it. 
You can check my other github repo "[Spriteware/machinelearning](https://github.com/Spriteware/machinelearning)" to see more implementations examples. 

If you like this work, don't hesitate to send me a friendly message on twitter [@Spriteware](https://twitter.com/Spriteware).
You can also visit my blog: [franpapers.com](https://franpapers.com)

