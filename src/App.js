import React, { useRef, useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import * as bodyPix from "@tensorflow-models/body-pix";
import "./App.css";

function App() {
  const imageRef = useRef(null);
  const canvasRef = useRef(null);
  const [imageURL, setImageURL] = useState(null); // State to hold the uploaded image URL

  const handleImageUpload = event => {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
      setImageURL(e.target.result);
    };

    reader.readAsDataURL(file);
  };

  const runBodysegment = async () => {
    if (imageRef.current) {
      const net = await bodyPix.load();
      console.log("BodyPix model loaded.");
      detect(net);
    }
  };

  const detect = async (net) => {
    const image = imageRef.current;
    const imageWidth = image.width;
    const imageHeight = image.height;

    // Set canvas height and width
    canvasRef.current.width = imageWidth;
    canvasRef.current.height = imageHeight;

    // Make detections
    const person = await net.segmentPersonParts(image);
    console.log(person);
    
    // const torsoMask = person.data.map(value => value === 0 ? 255 : 0); // Creates a binary mask for the torso
    // const torsoImageData = new ImageData(new Uint8ClampedArray(torsoMask), imageWidth, imageHeight);
    // const torsoCanvas = document.createElement('canvas');
    // torsoCanvas.width = imageWidth;
    // torsoCanvas.height = imageHeight;
    // torsoCanvas.getContext('2d').putImageData(torsoImageData, 0, 0);

    // Visualization code
    const coloredPartImage = bodyPix.toColoredPartMask(person);
    const opacity = 0.7;
    const flipHorizontal = false;
    const maskBlurAmount = 0;
    const canvas = canvasRef.current;

    bodyPix.drawMask(
      canvas,
      image,
      coloredPartImage,
      opacity,
      maskBlurAmount,
      flipHorizontal
    );
  };

  useEffect(() => {
    if (imageURL) {
      runBodysegment();
    }
  }, [imageURL]); // Re-run the effect when imageURL changes

  return (
    <div className="App">
      <header className="App-header">
        <input 
          type="file" 
          onChange={handleImageUpload} 
          accept="image/*" 
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            zIndex: 10,
            width: '100%',
            opacity: 1,
            cursor: 'pointer',
          }}
        />
        {imageURL && (
          <div style={{
            position: "relative",
            maxWidth: "100%",
            maxHeight: "100%",
          }}>
            <img
              ref={imageRef}
              src={imageURL}
              alt="Upload Preview"
              onLoad={runBodysegment}
              style={{
                maxWidth: "100%",
                maxHeight: "100%",
                display: "block", // Ensure it fills the container
              }}
            />
            <canvas
              ref={canvasRef}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
              }}
            />
          </div>
        )}
      </header>
    </div>
  ); 
}

export default App;
