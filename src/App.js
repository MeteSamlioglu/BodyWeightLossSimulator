import React, { useRef, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import * as bodyPix from "@tensorflow-models/body-pix";
import "./App.css";

function App() {
  const imageRef = useRef(null);
  const canvasRef = useRef(null);

  const runBodysegment = async () => {
    const net = await bodyPix.load();
    console.log("BodyPix model loaded.");
    detect(net);
  };

  const detect = async (net) => {
    if (imageRef.current) {
      const image = imageRef.current;
      const imageWidth = image.width;
      const imageHeight = image.height;

      // Set canvas height and width
      canvasRef.current.width = imageWidth;
      canvasRef.current.height = imageHeight;
      console.log(imageWidth)
      console.log(imageHeight)
      // Make detections
      const person = await net.segmentPersonParts(image);
      console.log(person.allPoses);

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
    }
  };

  useEffect(() => {
    runBodysegment();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <img
          ref={imageRef}
          src="sample.png"
          alt="Sample"
          onLoad={() => runBodysegment()}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            maxWidth: "100%",
            maxHeight: "100%",
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
          }}
        />
      </header>
    </div>
  );
}

export default App;