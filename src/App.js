import React, { useRef, useState, useEffect, useCallback } from 'react';
import * as bodyPix from '@tensorflow-models/body-pix';
import '@tensorflow/tfjs-backend-webgl';
import './App.css'; // Import the CSS file

function App() {
  const imageRef = useRef(null);
  const [imageURL, setImageURL] = useState(null);
  const [returnedImageURL, setReturnedImageURL] = useState(null);
  const [activeButton, setActiveButton] = useState(null);
  const [height, setHeight] = useState('');
  const [weight, setWeight] = useState('');
  const [bmi, setBmi] = useState(null);
  const [showTable, setShowTable] = useState(false);
  const [weightLossData, setWeightLossData] = useState([]);
  const [bodyPartsData, setBodyPartsData] = useState(null);
  const [face, setFace] = useState('');
  const [torso, setTorso] = useState('');
  const [upperLegs, setUpperLegs] = useState('');
  const [hips, setHips] = useState('');
  const [arms, setArms] = useState('');
  const [lowerLegs, setLowerLegs] = useState('');

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
          acc[keypoint.part] = {
            position: keypoint.position,
            score: keypoint.score
          };
          return acc;
        }, {})
      };

      setBodyPartsData(bodyPartsData);

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

  const handleButtonClick = (button) => {
    setActiveButton((prevActiveButton) => (prevActiveButton === button ? null : button));
    if (button !== 'button1') {
      setHeight('');
      setWeight('');
      setBmi(null);
      setShowTable(false);
      setWeightLossData([]);
    }
  };

  const handleHeightChange = (event) => {
    setHeight(event.target.value);
  };

  const handleWeightChange = (event) => {
    setWeight(event.target.value);
  };

  const calculateBmi = (weight) => {
    const heightInMeters = height / 100;
    return (weight / (heightInMeters * heightInMeters)).toFixed(2);
  };

  const generateWeightLossData = () => {
    const data = [];
    let currentWeight = parseFloat(weight);

    const decreasePercentages = {
      torso: 0.04,
      hipAndThigh: 0.02,
      legs: 0.02,
      faceAndNeck: 0.01,
      arms: 0.005
    };

    const months = [1, 3, 6, 12, 15, 18, 24];

    const getCumulativeLoss = (initialWeight, percentage, period) => {
      return (initialWeight * percentage * period).toFixed(2);
    };

    for (const month of months) {
      if (calculateBmi(currentWeight) <= 22.5) break;

      const period = month === 1 ? 1 : month / 3;

      data.push({
        month: `${month}th`,
        torso: getCumulativeLoss(weight, decreasePercentages.torso, period),
        hipAndThigh: getCumulativeLoss(weight, decreasePercentages.hipAndThigh, period),
        legs: getCumulativeLoss(weight, decreasePercentages.legs, period),
        faceAndNeck: getCumulativeLoss(weight, decreasePercentages.faceAndNeck, period),
        arms: getCumulativeLoss(weight, decreasePercentages.arms, period),
        totalWeight: currentWeight.toFixed(2),
        bmi: calculateBmi(currentWeight),
      });
      currentWeight *= 0.9;
    }

    setWeightLossData(data);
    setShowTable(true);
  };

  const handleCalculateBmi = () => {
    const bmiValue = calculateBmi(parseFloat(weight));
    setBmi(bmiValue);
    generateWeightLossData();
  };

  const handleShowClick = async (month) => {
    console.log(month);
    if (imageURL && bodyPartsData) {
      try {
        const response = await fetch(imageURL);
        const blob = await response.blob();

        const formData = new FormData();
        formData.append('month', month);
        formData.append('image', blob, 'uploaded_image.png');
        formData.append('data', JSON.stringify(bodyPartsData));

        const uploadResponse = await fetch("http://localhost:5000/UploadImage", {
          method: "POST",
          body: formData,
        });

        if (!uploadResponse.ok) {
          throw new Error("Network response was not ok");
        }

        const blobResponse = await uploadResponse.blob();
        const url = URL.createObjectURL(blobResponse);
        setReturnedImageURL(url);
        console.log("Image, month, and body parts data successfully sent:", url);
      } catch (error) {
        console.error("Error sending image, month, and body parts data:", error);
      }
    }
  };

  const handleAdvancedSubmit = async () => {
    if (imageURL && bodyPartsData) {
      try {
        const response = await fetch(imageURL);
        const blob = await response.blob();

        const formData = new FormData();
        formData.append('face', face);
        formData.append('torso', torso);
        formData.append('upperLegs', upperLegs);
        formData.append('hips', hips);
        formData.append('arms', arms);
        formData.append('lowerLegs', lowerLegs);
        formData.append('image', blob, 'uploaded_image.png');
        formData.append('data', JSON.stringify(bodyPartsData));

        const uploadResponse = await fetch("http://localhost:5000/AdvancedWeightLoss", {
          method: "POST",
          body: formData,
        });

        if (!uploadResponse.ok) {
          throw new Error("Network response was not ok");
        }

        const blobResponse = await uploadResponse.blob();
        const url = URL.createObjectURL(blobResponse);
        setReturnedImageURL(url);
        console.log("Advanced weight loss data successfully sent:", url);
      } catch (error) {
        console.error("Error sending advanced weight loss data:", error);
      }
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
        <div className="controls">
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
        </div>
        <div className="content">
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
          <div className="right-section">
            <div className="button-group">
              <button
                onClick={() => handleButtonClick('button1')}
                className={`toggle-button ${activeButton === 'button1' ? 'active' : ''}`}
                disabled={activeButton && activeButton !== 'button1'}
              >
                BMI
              </button>
              <button
                onClick={() => handleButtonClick('button2')}
                className={`new-button ${activeButton === 'button2' ? 'active' : ''}`}
                disabled={activeButton && activeButton !== 'button2'}
              >
                Advanced
              </button>
            </div>
            {activeButton === 'button1' && (
              <>
                <div className="bmi-section">
                  <div className="bmi-form">
                    <h3>Calculate Your BMI</h3>
                    <input
                      type="number"
                      placeholder="Height (cm)"
                      value={height}
                      onChange={handleHeightChange}
                    />
                    <input
                      type="number"
                      placeholder="Weight (kg)"
                      value={weight}
                      onChange={handleWeightChange}
                    />
                    <button onClick={handleCalculateBmi} className="calculate-button">Calculate</button>
                    {bmi && (
                      <div className="bmi-result">
                        <h4>Your BMI: {bmi}</h4>
                      </div>
                    )}
                  </div>
                </div>
                {showTable && (
                  <div className="bmi-table">
                    <table>
                      <thead>
                        <tr>
                          <th>Month</th>
                          <th>Torso</th>
                          <th>Hip and Thigh</th>
                          <th>Legs</th>
                          <th>Face and Neck</th>
                          <th>Arms</th>
                          <th>Total Weight</th>
                          <th>BMI</th>
                          <th>Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {weightLossData.map((row, index) => (
                          <tr key={index}>
                            <td>{row.month}</td>
                            <td>{row.torso}</td>
                            <td>{row.hipAndThigh}</td>
                            <td>{row.legs}</td>
                            <td>{row.faceAndNeck}</td>
                            <td>{row.arms}</td>
                            <td>{row.totalWeight}</td>
                            <td>{row.bmi}</td>
                            <td><button onClick={() => handleShowClick(row.month)}>Show</button></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </>
            )}
            {activeButton === 'button2' && (
              <div className="advanced-section">
                <h3>Advanced Weight Loss</h3>
                <div className="advanced-form">
                  <label>
                    Face:
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="0.35"
                      value={face}
                      onChange={(e) => setFace(e.target.value)}
                    />
                  </label>
                  <label>
                    Torso:
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="0.35"
                      value={torso}
                      onChange={(e) => setTorso(e.target.value)}
                    />
                  </label>
                  <label>
                    Upper Legs:
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="0.35"
                      value={upperLegs}
                      onChange={(e) => setUpperLegs(e.target.value)}
                    />
                  </label>
                  <label>
                    Hips:
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="0.35"
                      value={hips}
                      onChange={(e) => setHips(e.target.value)}
                    />
                  </label>
                  <label>
                    Arms:
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="0.35"
                      value={arms}
                      onChange={(e) => setArms(e.target.value)}
                    />
                  </label>
                  <label>
                    Lower Legs:
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="0.35"
                      value={lowerLegs}
                      onChange={(e) => setLowerLegs(e.target.value)}
                    />
                  </label>
                  <button onClick={handleAdvancedSubmit} className="calculate-button">Compute</button>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>
    </div>
  );
}

export default App;
