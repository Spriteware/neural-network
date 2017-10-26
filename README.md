# It's for training but also visualisation

This piece of code is a neural network implementation in Javascript. 
You can see an example of what it does here : https://franpapers.com/en/2017-neural-network-implementation-in-javascript-by-an-example/

The main idea was to do something that can **help to visualize the network and its modification** through backpropagation. Here's an example:

![Visualization example](https://franpapers.com/wp-content/uploads/2017/10/Capture.png)

**SVG** is used to provide you a clean visualisation, and a **simple Web Worker** is used for the training part (for avoiding blocking UI thread).

This library was not meant for "distribution" purpose, so it may have thousand bugs and may be not working as you want. **Fork it !**

## Basic example
It's simple, there are just way too much comments
```javascript


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


```

## Training data example 
The training looks like a CSV file. Except that it is not. Every `inputs/targets` couple is separated by a `;`. `inputs `and `targets` are separated by a `:`. `inputs` or `targets` values are seperated by a `space`. Here is an example:
```
input1 input2 input3 input4 : target1 target2 ;
input1 input2 input3 input4 : target1 target2 ;
```

Usually I save everything into a JS variable, and I put the JS script as a normal script in my page. This is why you can see this "training_data_imported" variable in the code above.
```javascript
var training_data_imported = "0 0.004012032080192407 0 0.004012032080192407 : 6.123233995736767e-17 -1;\
0.003901877324468339 0.0009301870879891982 0.00034437034884393647 -0.0009193694393909713 : -0.9726239602750568 -0.23238466364815152;\
0.003968835720993713 0.0005847808693595358 0.00006695839652537394 -0.0003454062186296625 : -0.9892652288476347 -0.14613112944556783;\
0.00380007257485393 0.0012799354131641794 -0.00016876314613978316 0.0006951545438046436 : -0.9473190861218562 0.32029135028790773;\
";
```

## Helpers for creating a dataset 
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

## And ?

And that's it. There are no other things to learn about this.
You can check my other github repo "[Spriteware/machinelearning](https://github.com/Spriteware/machinelearning)" to see more implementations examples. 

If you like this work, don't hesitate to send me a friendly message on twitter [@Spriteware](https://twitter.com/Spriteware).
You can also visit my blog: [franpapers.com](https://franpapers.com)
