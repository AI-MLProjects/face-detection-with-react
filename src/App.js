import React, { useRef } from "react";
import './App.css';

import * as tf from "@tensorflow/tfjs";
import * as facemesh from "@tensorflow-models/facemesh";
import Webcam from "react-webcam";
import { drawMesh, addData } from "./utilities";
import * as ml5 from "ml5";

function App() {

  // Setup Refrences
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  let brain;
  const initBrain = () => {
    const options = {
      inputs: 1404,
      outputs: 2,
      task: 'classification',
      debug: true
    };
    brain = ml5.neuralNetwork(options);
  }
  initBrain();
  // Load Facemesh
  const runFacemesh = async () => {
    const net = await facemesh.load({
      inputResolution: {width: 640, height: 480},
      scale: 0.8
    });

    setInterval(() => {
      detect(net);
    }, 100);
  }

  const addHappyData = () => {
    console.log('Happy data is called --- ');
    state = 'collecting';
    label = '1';
  };
  const addSadData = () => {
    console.log('Sad data is called --- ');
    state = 'collecting';
    label = '2';
  };
  const saveData = () => {
    console.log('save data is called --- ');
    brain.saveData('facedata');
  };
  const loadDataAndTrainModel = () => {
    console.log('trainModel is called ---');
    brain.loadData('./facedata.json', () => {
      console.log('Data loaded --- ');
      brain.normalizeData();
      brain.train({epochs: 50}, saveModel);
    });
  };
  const trainModel = () => {
    brain.normalizeData();
    brain.train({epochs: 50}, saveModel);
  }
  const saveModel = () => {
    console.log('Model trained');
    brain.save();
  }

  const predictModel = () => {

  };
  let state = 'waiting';
  let label = '';

  // Detect function
  const detect = async(net) => {
    if(typeof webcamRef.current !== "undefined" && 
      webcamRef.current !== null && 
      webcamRef.current.video.readyState === 4) {
        // Get video properties
        const video = webcamRef.current.video;
        const {videoWidth, videoHeight} = video;
        
        // set video width
        webcamRef.current.video.width = videoWidth;
        webcamRef.current.video.height = videoHeight;
        
        // set canvas width
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;
        // Make detection
        const face = await net.estimateFaces(video);
        // console.log(face);
        
        // Get canvas context for drawing
        const ctx = canvasRef.current.getContext("2d");
        drawMesh(face, ctx);
        if(state === 'collecting') {
          addData(brain, face, label);
          state = 'waiting';
        }
    }
  }
  runFacemesh();
  
  return (
    <div className="App">
      <header className="App-header">
        <Webcam
          ref={webcamRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 9,
            width: 640,
            height: 480,
          }}
        ></Webcam>
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 9,
            width: 640,
            height: 480,
        }}
        ></canvas>
      </header>
      <div>
        <button onClick={addHappyData} >Colledt Happy Data</button>
        <button onClick={addSadData}>Colledt Sad Data</button>
        <button onClick={saveData}>Save Data</button>
        <button onClick={loadDataAndTrainModel}>Train Model</button>
        <button onClick={predictModel}>Predict Model</button>
      </div>
    </div>
  );
}

export default App;
