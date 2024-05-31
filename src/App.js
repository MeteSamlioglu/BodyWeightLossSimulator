import React, { useRef, useState, useEffect, useCallback } from 'react';
import * as bodyPix from '@tensorflow-models/body-pix';
import '@tensorflow/tfjs-backend-webgl';
import './App.css'; // Import the CSS file

function App() {
  const imageRef = useRef(null);
  const [imageURL, setImageURL] = useState(null);
  const [returnedImageURL, setReturnedImageURL] = useState(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (!file) {
      // No file selected, return early
      return;
    }

    const reader = new FileReader();

    reader.onload = function (e) {
      setImageURL(e.target.result);
    };

    reader.readAsDataURL(file);
  };

  const runBodysegment = useCallback(async () => {
    if (imageRef.current) {
      const net = await bodyPix.load();
      console.log("BodyPix model loaded.");
      detect(net);
    }
  }, []);

  const detect = async (net) => {
    const image = imageRef.current;

    // Use the image's rendered dimensions
    const imageWidth = image.clientWidth;
    const imageHeight = image.clientHeight;

    // Set a temporary canvas to match the image's displayed dimensions
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = imageWidth;
    tempCanvas.height = imageHeight;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(image, 0, 0, imageWidth, imageHeight);

    // Convert the canvas to a Blob (resized image)
    tempCanvas.toBlob(async (blob) => {
      // Make detections
      const person = await net.segmentPersonParts(tempCanvas);
      console.log(person);

      // Extract and format the required data
      const bodyPartsData = {
        score: person.allPoses[0].score,
        keypoints: person.allPoses[0].keypoints.reduce((acc, keypoint) => {
          acc[keypoint.part] = keypoint.position;
          return acc;
        }, {})
      };

      // Send the JSON data and the resized image to the Flask server
      postBodyPartsData(bodyPartsData, blob);
    }, 'image/png');
  };

  const postBodyPartsData = async (bodyPartsData, imageBlob) => {
    const formData = new FormData();
    formData.append('data', JSON.stringify(bodyPartsData));
    formData.append('image', imageBlob, 'resized_image.png');

    try {
      const response = await fetch("http://localhost:5000/BodyWeightLoss", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setReturnedImageURL(url);
      console.log("Data successfully sent and received:", url);
    } catch (error) {
      console.error("Error sending data:", error);
    }
  };

  useEffect(() => {
    if (imageURL) {
      runBodysegment();
    }
  }, [imageURL, runBodysegment]);

  const handleReset = () => {
    if (imageURL) {
      runBodysegment();
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <div className="image-section">
          <div className="image-container">
            {imageURL && (
              <img
                ref={imageRef}
                src={imageURL}
                alt="Upload Preview"
                className="uploaded-image"
              />
            )}
          </div>
          <div className="image-container">
            {returnedImageURL && (
              <img
                src={returnedImageURL}
                alt="Returned Image"
                className="uploaded-image"
              />
            )}
          </div>
        </div>
        <input
          type="file"
          onChange={handleImageUpload}
          accept="image/*"
          className="file-input"
        />
        {imageURL && (
          <button onClick={handleReset} className="reset-button">
            Reset
          </button>
        )}
      </header>
    </div>
  );
}

export default App;
